import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, Subset

import crypten
import crypten.nn as cnn
import crypten.mpc as mpc
import crypten.communicator as comm
from crypten.config import cfg

from plaintext.models import *

import numpy as np
import pandas as pd
from typing import Callable, Union, Dict, Any, List
import random
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import os
import argparse

import pickle
from PIL import Image
import matplotlib.pyplot as plt

from config import *

## INITIATING FUNCTIONS

def set_seed(seed_value=42):
    """Sets seeds for reproducibility across multiple components."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    # 1. GPU specific settings (MOST IMPORTANT FOR YOUR ISSUE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # for multi-GPU
        
        # Forces the use of deterministic algorithms where possible
        # This is the line that typically fixes the CUDNN issue.
        torch.backends.cudnn.deterministic = True
        
        # Disables auto-tuning which can select non-deterministic algorithms
        torch.backends.cudnn.benchmark = False 

def parsing_all():
    parser = argparse.ArgumentParser(description="Run single inference on a specified dataset and model.")
    parser.add_argument('--dataset', type=str, required=True, choices=['MNIST', 'Tiny Imagenet', 'Imagenet'],
                        help="The dataset to use: 'MNIST' (downloads via torchvision) or 'Tiny Imagenet' (uses ./data/tiny-imagenet) r 'Imagenet' (uses ./data/imagenet).")
    parser.add_argument('--model', type=str, required=True, choices=['AlexNet', 'ResNet18', 'ResNet50', 'ResNet152', 'VGG11', 'VGG19', 'Google', 'AliceNet'],
                        help="The neural network model to use: Choose 'AlexNet' or 'ResNet18' or 'ResNet50' or 'ResNet152' or 'VGG11', 'VGG19', or 'Google' or 'AliceNet'. \
                            'AlexNet' is not available in Secure mode; will compute a Crypten-ized 'CustomAlexNet' instead.")
    parser.add_argument('--explain', type=str, default="vanilla", 
                        help='The explanation method (e.g., "vanilla, "xinput").')
    parser.add_argument('--mode', type=str, required=True, choices=['plaintext', 'secure'],
                    help="The inference mode to use: 'plaintext' or 'secure'.")
    return parser.parse_args()

def rename_inputs(args):
    if args.dataset == "MNIST":
        args.dataset_txt = "MNIST"
        args.dataset_str = "mnist"
    elif args.dataset == "Tiny Imagenet":
        args.dataset_txt = "Tiny ImageNet"
        args.dataset_str = "tiny-imagenet"
    else:
        args.dataset_txt = "Imagenet"
        args.dataset_str = "imagenet"
    
    return args

## CONFIGURATE DATASET

class TinyImageNetValDataset(Dataset):
    """
    Corrected custom dataset to load the 'val' split of Tiny-ImageNet,
    assuming images are located in class-specific subdirectories like:
    ./data/tiny-imagenet/images/nID/image_name.jpeg
    and labels are in ./data/tiny-imagenet/val/val_annotations.txt
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.base_img_dir = os.path.join(root_dir, 'val','images') 
        ann_file = os.path.join(root_dir, 'val', 'val_annotations.txt')
        
        self.nID_to_class = {}
        try:
            with open(os.path.join(root_dir, 'wnids.txt'), 'r') as f:
                for i, line in enumerate(f):
                    self.nID_to_class[line.strip()] = i
        except FileNotFoundError:
            raise ValueError("Warning: wnids.txt not found. Class mapping may be incorrect.")
            
        self.img_paths = []
        self.labels = []
        
        self.class_names = []
        try:
            with open(os.path.join(root_dir, 'words.txt'), 'r') as f:
                self.class_names = [line.strip().split('\t')[0] for line in f if line.strip()][:200]
        except FileNotFoundError:
            raise ValueError("Warning: words.txt not found. Using numeric class names.")
            self.class_names = [str(i) for i in range(200)]

        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.split('\t')
                img_name = parts[0]
                n_id = parts[1] 
                img_path = os.path.join(self.base_img_dir, n_id, img_name)
                label = self.nID_to_class.get(n_id, -1)
                if label != -1 and os.path.exists(img_path): 
                    self.img_paths.append(img_path)
                    self.labels.append(label)
                # if len(self.img_paths) >= 150:
                #     break

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, self.class_names[label]

class ImageNetTestDataset(Dataset):
    def __init__(self, root_dir="../data", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_map = []
        self.class_names = [f"Class {i}" for i in range(1000)]

        for i in range(10): 
            filename = f"test_{i}_image.jpg"
            img_path = os.path.join(self.root_dir, filename)
            
            if os.path.exists(img_path) and filename in IMAGENET_LABELS:
                self.data_map.append({
                    'path': img_path,
                    'label': IMAGENET_LABELS[filename],
                    'filename': filename
                })
        
        if not self.data_map:
            print(f"Warning: No images found in {root_dir} matching 'test_0_image.jpg' to 'test_9_image.jpg'.")

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        item = self.data_map[idx]
        img_path = item['path']
        label = item['label']
        image = Image.open(img_path).convert('RGB') 
        
        if self.transform:
            image = self.transform(image)

        return image, label, self.class_names[label]
 
def transform_datasets(args):
    norm_configs = {
        "AliceNet": {"mean": (0.1307,), "std": (0.3081,)},
        "DEFAULT":  {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    }
    if args.dataset_str == "mnist" and args.model == "AliceNet":
        t_list = [transforms.ToTensor()]
    else:
        t_list = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        ]
    if args.normalized == "yes":
        t_list.extend([
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-7)),
            transforms.Lambda(lambda x: x + 0.01),
        ])
    else:
        cfg = norm_configs.get(args.model, norm_configs["DEFAULT"])
        t_list.append(transforms.Normalize(**cfg))

    return transforms.Compose(t_list)
    
def preprocess_data(args):
    if args.dataset_str == 'tiny-imagenet':
        root_path = './data/tiny-imagenet'
        if not os.path.isdir(root_path):
            raise ValueError(f"Error: Tiny-ImageNet directory not found at {root_path}. \
                            Please ensure the dataset is physically downloaded and located there.")
        elif not os.path.exists(os.path.join(root_path, 'val', 'val_annotations.txt')):
            raise ValueError(f"Warning: Tiny-ImageNet validation annotation file missing. The custom dataloader may fail.")
        # Tiny-ImageNet is 3-channel (RGB), original size 64x64
        transform = transform_datasets(args)
        # Use the custom dataset class
        test_data = TinyImageNetValDataset(root_dir='./data/tiny-imagenet', transform=transform)
        num_classes = 200
        class_names = test_data.class_names
    elif args.dataset_str == 'imagenet':
        root_path = './data/imagenet' 
        if not any(os.path.exists(os.path.join(root_path, f"test_{i}_image.jpg")) for i in range(11)):
            raise ValueError(f"Error: ImageNet test images not found in {root_path}. "
                             f"Expected files like 'test_0_image.jpg'.")

        transform = transform_datasets(args)
        test_data = ImageNetTestDataset(root_dir=root_path, transform=transform) 
        num_classes = 1000 
        class_names = test_data.class_names
    elif args.dataset_str == 'mnist':
        transform = transform_datasets(args)
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        class_names = [str(i) for i in range(10)]            
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_str}. Choose 'mnist' or 'tiny-imagenet' or 'imagenet'.")
    limit = min(1, len(test_data))
    indices = list(range(limit))
    test_subset = Subset(test_data, indices)
    return test_subset, num_classes, class_names 
    

## CONFIGURATE MODEL

def load_and_configure_model(args, num_classes, MODEL_CONFIG):
    if args.model not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {args.model}. Choose from {list(MODEL_CONFIG.keys())}.")
    if args.mode not in ("plaintext", "secure"):
        raise ValueError(f"Unknown mode: {args.mode}. Choose 'secure' or 'plaintext'.")
    config = MODEL_CONFIG[args.model]    
    model_class = config[args.mode]
    model = model_class(args, num_classes)
    if args.mode == "secure":
        crypten.common.serial.register_safe_class(config['plaintext'])        
    pth_file = f'models/trained_{args.model}_{args.dataset_str}.pth'
    weights_path = getattr(args, 'weights_path', pth_file)
    
    if not os.path.exists(weights_path):
       raise ValueError(f"Trained model not found at {weights_path}")
    
    trained_state_dict = torch.load(weights_path, map_location='cpu')
    try:    
        model.load_state_dict(trained_state_dict, strict=False)
        print(f"\t✅ Uploaded weights from {pth_file}")
    except Exception as e:
        raise ValueError(f"Error loading state dict: {e}. Model may not be correctly configured.")
    return model


## RUN INFERENCES

def get_dummy_input(args):
    """
    Generates a dummy input tensor based on the specific 
    dimensions required by the dataset and model transforms.
    """
    if args.dataset_str == "tiny-imagenet":
        return torch.randn(1, 3, 224, 224)

    elif args.dataset_str == "mnist":
        if args.model == "AliceNet":
            return torch.randn(1, 1, 28, 28)
        elif args.model == "AlexNet":
            return torch.randn(1, 3, 224, 224)
        else:
            return torch.randn(1, 3, 224, 224)

    elif args.dataset_str == "imagenet":
        return torch.randn(1, 3, 224, 224)

    else:
        return torch.randn(1, 3, 224, 224)
        
def setup_secure_inference(args, model):
    """Sets up the crypten environment and encrypts the model."""
    
    cfg.communicator.verbose = True
    #crypten.communicator.get().reset_communication_stats()
    torch.set_num_threads(1)
    
    if isinstance(model, crypten.nn.Module): 
        name = type(model).__name__
        private_model = model
        str_model= "a Custom cnn.Module"
    elif isinstance(model, nn.Module):
        name = type(model).__name__
        dummy_input = get_dummy_input(args)
        private_model = cnn.from_pytorch(model, dummy_input)
        str_model= "imported from_pytorch"
    else:
        raise TypeError("Model must be a PyTorch nn.Module or CrypTen nn.Module.")
    private_model.encrypt(src=0)
    
    
    private_model.eval()
    return private_model, str_model

# def run_secure_inference(private_model, input_data, args):
#     """
#     Performs the secure forward pass for a single batch/input.
#     Handles encryption/decryption timing for single-sample mode.
#     """
#     input_tensor = input_data.unsqueeze(0)
    
#     #scale = 10
#     #input_scaled = input_tensor/scale
#     private_input_tensor = crypten.cryptensor(input_tensor, src=1, requires_grad=False)
#     with crypten.no_grad():
#         private_output = private_model(private_input_tensor)
        
#         output = private_output.get_plain_text() 

#     rounds_com_prep = comm.get().comm_rounds
#     bytes_com_prep = comm.get().comm_bytes
#     bytes_sent = bytes_com_prep
#     comm.get().broadcast_obj(bytes_sent, src=0)
#     comm.get().broadcast_obj(rounds_com_prep, src=0)

#     comm.get().reset_communication_stats()

#     return private_input_tensor

TensorType = Union[crypten.CrypTensor, torch.Tensor]

def calculate_saliency(
    attribution_function: Callable[[TensorType, int], TensorType],
    model,
    input_tensor: TensorType,
    target_class: int,
    args: argparse.Namespace):
    """
    Computes a saliency map using the provided function and saves the result in the pkl file.

    Args:
        attribution_function: The 'attribute' method from your XAI class (e.g., gc_alexnet.attribute).
        input_tensor: The input image tensor (B, C, H, W).
        target_class: The class index being explained matching the true label of the photo.
        args: The arguments of the input.
    """
    
    comm.get().barrier()
    
    comm.get().reset_communication_stats()
    
    saliency_map_tensor, output = attribution_function(args, input_tensor, target_class)
    saliency_map_tensor.get_plain_text()

    rounds_com_prep = comm.get().comm_rounds
    bytes_com_prep = comm.get().comm_bytes
    bytes_sent = bytes_com_prep
    comm.get().barrier()

    comm.get().broadcast_obj(bytes_sent, src=0)
    comm.get().broadcast_obj(rounds_com_prep, src=0)
    comm.get().barrier()
    comm.get().reset_communication_stats()
    
        
    comm.get().barrier()
    input_tensor.get_plain_text()
        
    if "integrated" in args.explain:
        comm.get().barrier()
        output = model(input_tensor)
    comm.get().barrier()
    output.get_plain_text()
    