import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, Subset

import crypten
import crypten.nn as cnn
import crypten.mpc as mpc
import crypten.communicator as comm

from plaintext.models import *

import numpy as np
import pandas as pd
from typing import Callable, Union, Dict, Any, List
import random
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import os
import argparse
import time
import logging
from datetime import datetime

import pickle
from PIL import Image
import matplotlib.pyplot as plt

from config import *

import warnings
warnings.filterwarnings("ignore")

TensorType = Union[crypten.CrypTensor, torch.Tensor]

suffix = "_bn"

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

def parsing_all(main_script=True):
    parser = argparse.ArgumentParser(description="Run single inference on a specified dataset and model.")
    parser.add_argument('--normalized', type=str, default="no", choices=['yes', 'no'], 
                        help="Whether to use input images with values > 0.")
    parser.add_argument('--map', type=str, default="yes", choices=['yes', 'no'],
                        help="Determines whether to generate a matplotlib plot of the comparison.")
    parser.add_argument('--steps', type=int, default=2, help="The number of times the gradient is calculated.")
    parser.add_argument('--baseline', type=str, default="zero", help="The type of baseline used for Integrated Gradient XAI.",
                        choices=["zero", "blur", "white", "gray", "random"])
    parser.add_argument('--epsilon', type=float, default=1e-6, help="The epsilon value to stabilize LRP to prevent division by 0; Most importantly, it determines the amount of filtering for relevance (The higher, the smoother the LRP map).")
    if main_script:
        parser.add_argument('--dataset', type=str, required=True, choices=['MNIST', 'Tiny Imagenet', 'Imagenet'],
                        help="The dataset to use: 'MNIST' (downloads via torchvision) or 'Tiny Imagenet' (uses ./data/tiny-imagenet).")
        parser.add_argument('--model', type=str, required=True, choices=['AlexNet', 'ResNet18', 'ResNet50', 'ResNet152', 'VGG19', 'VGG11', 'AliceNet', 'ShelfResNet18'],
                        help="The neural network model to use: Choose 'AlexNet' or 'ResNet18' or 'ResNet50' or 'ResNet152' or 'VGG11' or 'VGG19' or 'Google' or 'AliceNet' or 'ShelfResNet18'. \
                            'AlexNet' is not available in Secure mode; will compute a Crypten-ized 'CustomAlexNet' instead.")
        parser.add_argument('--mode', type=str, required=True, choices=['plaintext', 'secure'],
                        help="The inference mode to use: 'plaintext' or 'secure'.")
        parser.add_argument('--inference', type=str, default=None, choices=['single', 'multiple'],
                        help='Run 1 image ("single") or the entire testset ("multiple").')
        parser.add_argument('--idx', type=int, default=None, help='Index value of image chosen in testset, required for args.inference=="single".')
        parser.add_argument('--explain', type=str, default=None, 
                        help='The explanation method (e.g., "grad_simple, "grad_xinput"). Leave empty for no explanation.',
                        choices=["vanilla", "xinput", "integrated"])
        
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
    args.map = args.map.lower()
    args.normalized = args.normalized.lower()
    args.is_positive = "_positive-input" if args.normalized == "yes" else ""
    if "integrated" in args.explain:
        args.explain = f"{args.explain}_{args.steps}_{args.baseline}"
    elif "LRP" in args.explain:
        args.explain = f"{args.explain}_{args.epsilon:.0e}"
    else: 
        pass
    args.identity = f"{args.mode}_{args.model.lower()}_{args.dataset_str}{suffix}"
    return args

def logging_intro(filename):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
    filename=filename,
    level=logging.INFO,
    format='%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
    logging.info("\n=====================================================================")
    current_time = datetime.now()
    time_str = current_time.strftime('%Y-%m-%d %H:%M:%S') 
    logging.info(f"{time_str}")
    logging.info("=====================================================================")


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
                # if len(self.img_paths) >= 20:
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
        transform = transform_datasets(args)
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
    if args.inference == "single":
        limit = min(1, len(test_data))
        indices = list(range(limit))
        test_subset = Subset(test_data, indices)
        return test_subset, num_classes, class_names 
    
    elif args.inference == "multiple":
        test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
        return test_loader, num_classes, class_names
    else:
        raise ValueError(f"Unknown inference mode: {args.inference}.")
    

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
        print(f"✅ Uploaded weights from {pth_file}")
    except Exception as e:
        raise ValueError(f"Error loading state dict: {e}. Model may not be correctly configured.")
    return model


## RUN INFERENCES

def setup_plaintext_inference(args, model):
    """Sets up the model and device for plaintext inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device

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
        else:
            return torch.randn(1, 3, 224, 224)

    elif args.dataset_str == "imagenet":
        return torch.randn(1, 3, 224, 224)
        
def setup_secure_inference(args, model):
    """Sets up the crypten environment and encrypts the model."""
    logging.info("=====================================================================\n")
    
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
    if args.inference == "single":
        start = time.perf_counter()
    private_model.encrypt(src=0)
    if args.inference == "single":
        end = time.perf_counter()
        model_enc_time = end-start
    else: 
        model_enc_time = None
    private_model.eval()
    return private_model, str_model, model_enc_time

# def run_plaintext_inference(model, input_data, device, args):
#     """Performs the forward pass for a single batch/input."""

#     if args.inference == "single":
#         input_tensor = input_data.unsqueeze(0).to(device)
#     else:
#         input_tensor = input_data.to(device)
    
    
#     with torch.no_grad():
#         if args.inference == "single":
#             start = time.perf_counter()
        
#         output = model(input_tensor)
        
#         if args.inference == "single":
#             end = time.perf_counter()
#             duration = end-start
            
#     return input_tensor, output
    
# def run_secure_inference(private_model, private_input_tensor, args):
#     """
#     Performs the secure forward pass for a single batch/input.
#     Handles encryption/decryption timing for single-sample mode.
#     """
#     input_tensor = input_data.unsqueeze(0) if args.inference == "single" else input_data
    
#     if args.inference == "single":
#         start = time.perf_counter()
    

#     with crypten.no_grad():
#         if args.inference == "single":
#             end = time.perf_counter()
#             enc_time = (end - start)
#             logging.info(f"\nSECURE INFERENCE PROCESS ({args.model} for {args.dataset_txt})")
#             logging.info(f"\tinput encryption  : {(enc_time):.4f} s")
        
#         if args.inference == "single":
#             start = time.perf_counter()
                
#         private_output = private_model(private_input_tensor)
#         if args.inference == "single":
#             end = time.perf_counter()
#             inf_time = (end - start)
#             logging.info(f"\tsecure inference  : {(inf_time):.4f} s")
#         if args.inference == "single":
#             start = time.perf_counter()
            
#         output = private_output.get_plain_text() 
        
#         if args.inference == "single":
#             end = time.perf_counter()
#             dec_time = (end - start)
#             logging.info(f"\toutput decryption : {(dec_time):.4f} s")

    
    
#     bytes_sent = comm.get().broadcast_obj(None, src=0)
#     rounds_com_prep = comm.get().broadcast_obj(None, src=0)
#     total_inf = (enc_time+inf_time+dec_time)
#     print(f'\nTOTAL SECURE INFERENCE : {total_inf} seconds')
#     print(f"\tInference Secure Communication: {bytes_sent:.2f} GB ({bytes_sent * (1000 ** 3)} bytes)")
#     print(f"\tInference Secure ROUNDS: {rounds_com_prep}")
    
#     logging.info(f'\tTOTAL SECURE INFERENCE : {total_inf} seconds')
#     logging.info(f"\t\tInference Secure Communication: {bytes_sent:.2f} GB ({bytes_sent * (1000 ** 3)} bytes)")
#     logging.info(f"\t\tInference Secure ROUNDS: {rounds_com_prep}")
#     comm.get().reset_communication_stats()
       
#     #prior_value = crypten.CrypTensor.AUTOGRAD_ENABLED
#     #crypten.CrypTensor.set_grad_enabled(True)
#     return private_input_tensor, output
    
def accuracy_single_inference(args, output, label, class_names):
    """Calculates and logs the result for a single sample."""
    class_label_str = class_names[label]
    _, predicted = torch.max(output.data, 1)
    predicted_class_idx = predicted.item()
    predicted_class_name = class_names[predicted_class_idx]
    is_correct = "correct" if predicted_class_idx == label else "incorrect"
    
    logging.info(f"\tresult: {is_correct}. --> image classified as {predicted_class_name}; correct class was {class_label_str}")
    print(f"\tresult: {is_correct}. --> image classified as {predicted_class_name}; correct class was {class_label_str}")
    return is_correct, predicted_class_idx

# def run_multiple_inferences(args, model, test_loader, device, inference_func):
#     """
#     Generic execution loop for full test set inference.
    
#     Args:
#         model (nn.Module or crypten.nn.Module): The configured model (plaintext or encrypted).
#         test_loader (DataLoader): The DataLoader for the full test set.
#         device (torch.device or None): Device for plaintext, None for secure.
#         inference_func (function): The specific function to run on each batch 
#                                    (e.g., run_plaintext_inference or run_secure_inference).
                                   
#     Returns:
#         tuple: (all_labels, all_preds)
#     """
#     all_preds = []
#     all_labels = []
#     image_count = 0 
#     context_manager = torch.no_grad() if args.mode == "plaintext" else crypten.no_grad()
    
#     with context_manager:
#         for data in test_loader:
#             if args.dataset_str == 'tiny-imagenet':
#                 inputs, labels, _ = data
#             else:
#                 inputs, labels = data
#             _, output = inference_func(model, inputs, device)
#             _, predicted = torch.max(output.data, 1)
#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#             image_count += 256
#             print(image_count)
#     return all_labels, all_preds

# def calculate_and_log_metrics(args, all_labels, all_preds):
#     """Calculates and logs all batch metrics."""
#     accuracy = accuracy_score(all_labels, all_preds)
    
#     # Calculate weighted metrics
#     f1_w = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
#     recall_w = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
#     precision_w = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    
#     # Calculate micro and macro metrics
#     f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
#     recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
#     precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
#     f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
#     recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
#     precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    
#     log_mode = "SECURE BATCH INFERENCE" if args.mode == "secure" else "BATCH INFERENCE"
#     logging.info(f"\n{log_mode} RESULTS on full test set:")
#     logging.info(f"\tAccuracy: {accuracy:.4f}")
    
#     logging.info(f"\n--- Weighted Metrics ---")
#     logging.info(f"\tF1 Score (Weighted): {f1_w:.4f}")
#     logging.info(f"\tRecall (Weighted): {recall_w:.4f}")
#     logging.info(f"\tPrecision (Weighted): {precision_w:.4f}")

#     logging.info(f"\n--- Micro Metrics ---")
#     logging.info(f"\tF1 Score (Micro): {f1_micro:.4f}")
#     logging.info(f"\tRecall (Micro): {recall_micro:.4f}")
#     logging.info(f"\tPrecision (Micro): {precision_micro:.4f}")
    
#     logging.info(f"\n--- Macro Metrics ---")
#     logging.info(f"\tF1 Score (Macro): {f1_macro:.4f}")
#     logging.info(f"\tRecall (Macro): {recall_macro:.4f}")
#     logging.info(f"\tPrecision (Macro): {precision_macro:.4f}")
#     return accuracy
        

## FOR INDIVIDUAL MAPS

def calculate_saliency(
    attribution_function: Callable[[TensorType, int], TensorType],
    model,
    input_tensor: TensorType,
    target_class: int,
    class_names,
    args: argparse.Namespace, i: int, baseline
):
    """
    Computes a saliency map using the provided function and saves the result in the pkl file.

    Args:
        attribution_function: The 'attribute' method from your XAI class (e.g., gc_alexnet.attribute).
        input_tensor: The input image tensor (B, C, H, W).
        target_class: The class index being explained matching the true label of the photo.
        args: The arguments of the input.
    """
    if args.mode == "secure":
        comm.get().barrier()
        comm.get().reset_communication_stats()
    
    start_saliency = time.perf_counter()
    saliency_map_tensor, output, times = attribution_function(args, input_tensor, target_class, baseline)
    if isinstance(saliency_map_tensor, (crypten.CrypTensor, mpc.mpc.MPCTensor)):
        sss = time.perf_counter()
        final_torch_tensor = saliency_map_tensor.get_plain_text()
        eee = time.perf_counter()
        times["heatmap decryption"] = eee-sss
        
        comm.get().barrier()
        bytes_sent = comm.get().broadcast_obj(None, src=0)
        rounds_com_prep = comm.get().broadcast_obj(None, src=0)
        end_saliency = time.perf_counter()
        comm.get().barrier()
        comm.get().reset_communication_stats()
        
        comm.get().barrier()
        input_tensor_decrypted = input_tensor.get_plain_text()
        if input_tensor_decrypted.dim() == 4:
            original_np = input_tensor_decrypted[0].permute(1, 2, 0).cpu().numpy()
        elif input_tensor_decrypted.dim() == 3:
            original_np = input_tensor_decrypted.permute(1, 2, 0).cpu().numpy()
        else:
            raise ValueError(f"Unexpected input dimension: {input_tensor_decrypted.dim()}")


    else:
        end_saliency = time.perf_counter()
        original_np = input_tensor[0].permute(1, 2, 0).cpu().numpy()
        final_torch_tensor = saliency_map_tensor
    
    saliency_np = final_torch_tensor.cpu().numpy()
    
    times["TOTAL XAI"] = end_saliency-start_saliency
    
    
    for key, value in times.items():
        logging.info(f"\t {key:<25} : {value:.4f} s")
        print(f"\t {key:<25} : {value:.4f} s")
    
    if args.mode == "secure":
        logging.info(f"\t\tHeatmap Secure Communication: {bytes_sent:.2f} bytes ({bytes_sent / (1000 ** 3)} GB)")
        logging.info(f"\t\tHeatmap Secure ROUNDS: {rounds_com_prep}")
        print(f"\t\tHeatmap Secure Communication: {bytes_sent:.2f} GB ({bytes_sent / (1000 ** 3)} GB)")
        print(f"\t\tHeatmap Secure ROUNDS: {rounds_com_prep}")
    
    
    if "integrated" in args.explain:
        start_infer = time.perf_counter()
        if args.mode == "secure":
            comm.get().barrier()
        output = model(input_tensor)
        end_infer = time.perf_counter()
        times["inference"] = end_infer - start_infer

    if args.mode == "secure":
        out_start = time.perf_counter()
        comm.get().barrier()
        output = output.get_plain_text()
        out_end = time.perf_counter()
        times["output decryption"] = out_end-out_start
        
    is_correct, predicted_class_idx = accuracy_single_inference(
    args, output, target_class, class_names)

    
    filename = f'output_pkls/heatmap_values.pkl'
    start1 = time.perf_counter() 
    method = args.explain + args.is_positive + suffix
    save_explanation_arrays(filename, args.dataset_str, args.model, method, args.mode, i, original_np, saliency_np)
    end1 = time.perf_counter()
    logging.info(f"\tSaving XAI heatmap: {(end1-start1)} s") 

    return saliency_np, predicted_class_idx
        
def save_explanation_arrays(filename: str, dataset: str, model: str, method: str, mode: str, idx: int, original: np.ndarray, array_data: np.ndarray):
    """
    Appends or overwrites a single structured data record (NumPy array + metadata) to the
    specified pickle file based on matching metadata.
    """

    new_record: Dict[str, Any] = {
        'dataset': dataset,
        'model': model,
        'method': method,
        'mode': mode,
        'idx': idx,
        'original': original,
        'data': array_data,
    }

    IDENTIFIER_KEYS = ['dataset', 'model', 'method', 'mode', 'idx']

    data_records: List[Dict[str, Any]] = []
    
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                data_records = pickle.load(f)

                
            if not isinstance(data_records, list):
                raise Warning(f"Existing file {filename} content is not a list. Overwriting file with new record.")
                data_records = []
        except Exception as e:
            raise Warning(f"Could not read existing file {filename} ({e}). Overwriting file with new record.")
            data_records = []
    len1 = len(data_records)
    record_found = False
    for record in reversed(data_records):
        is_match = all(record.get(key) == new_record[key] for key in IDENTIFIER_KEYS)
        if is_match:
            record['original'] = original
            record['data'] = array_data
            record_found = True
            
            break

    if not record_found:
        data_records.append(new_record)

    with open(filename, 'wb') as f:
        pickle.dump(data_records, f)
    len2 = len(data_records)
    status = "Updated existing" if record_found else "Added new"
    print(f"{status} record | Heatmaps: {len1} -> {len2})")
    return

def plot_saliency_map(
    final_torch_input: torch.Tensor,
    saliency_np: np.array,
    target_class: int,
    predicted_class: int,
    file_path: str,
    title: str,
    short_title: str,
    args: argparse.Namespace
):
    """
    Plots a Matplotlib figure (original image + heatmap).

    Args:
        input_tensor: The input image tensor (B, C, H, W).
        saliency_np: Numpy array of the saliency values
        target_class: The class index being explained matching the true label of the photo.
        predicted_class: The class index predicted by the model.
        file_path: The full path to save the output image (e.g., 'saliency_output.png').
        title: Title for the overall figure.
    """
    if target_class == predicted_class:
        results = "correctly"
    else:
        results = "incorrectly"

    img_np = final_torch_input.permute(1, 2, 0).cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original Input (Class {target_class})")
    axes[0].axis('off')
    im = axes[1].imshow(saliency_np, cmap='inferno') # 'hot' 
    axes[1].set_title(f"{short_title} HeatMap")
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
    plt.suptitle(f"{args.model} for {args.dataset_txt}\nExplainability using {title} for Class {target_class} ({results} predicted Class {predicted_class})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    plt.close(fig) 
  

## FOR COMPARATIVE MAPS

def calculate_all_metrics(row):
    """Calculates metrics based on specific pairings: 
    - Sparsity: Individual
    - Robustness: Original vs Heatmap
    - Consistency/Corr/L1/MSE: Plaintext vs Secure
    """
    # Extract arrays
    original = row['original']
    plaintext = row['plaintext_normed']
    secure = row['secure_normed']

    
    # --- 1. Comparison Metrics (Calculated ONCE: Plaintext vs Secure) ---
    l1_norm, mse, corr = compare_metrics(plaintext, secure)
    consistency = calculate_consistency(plaintext, secure)

    consistency_plaintext = calculate_consistency(plaintext, original)
    consistency_secure = calculate_consistency(secure, original)

    # --- 2. Sparsity (Calculated TWICE: Individual) ---
    sparsity_plaintext = calculate_sparsity(plaintext)
    sparsity_secure = calculate_sparsity(secure)
    

    if original.ndim == 3:
        if original.shape[0] == 3:
            original_sensitivity = np.mean(original, axis=0)
        else:
            original_sensitivity = np.mean(original, axis=2)
    else:
        original_sensitivity = original


    # --- 3. Robustness (Calculated TWICE: Original vs XAI Output) ---
    # Metric: How much the heatmap differs from the input signal
    sensitivity_plaintext = calculate_sensitivity(original_sensitivity, plaintext)
    sensitivity_secure = calculate_sensitivity(original_sensitivity, secure)
    
    dark_mean = darkness_mean(original)
    non_positive = non_positive_percentage(original)

    return pd.Series({
        # Shared Comparison Metrics
        'l1_norm': l1_norm, 
        'mse': mse, 
        'corr': corr,
        'consistency': consistency,
        'consistency_secure': consistency_secure,
        'consistency_plaintext': consistency_plaintext,

        #Original
        'original_avg_channel': dark_mean,
        'original_non_positives': non_positive,
        
        # Plaintext Specific
        'sparsity_plaintext': sparsity_plaintext, 
        'sensitivity_plaintext': sensitivity_plaintext,
        
        # Secure Specific
        'sparsity_secure': sparsity_secure,       
        'sensitivity_secure': sensitivity_secure,
        
        # Diff
        'sparsity_difference': sparsity_plaintext - sparsity_secure,
        'sensitivity_difference': sensitivity_plaintext - sensitivity_secure,
        'consistency_difference': consistency_plaintext - consistency_secure
    })

def darkness_mean(array):
    try:
        # 1. Ensure array is a numpy array
        array = np.asanyarray(array)
        
        # 2. Check for empty or non-numeric data
        if array.size == 0:
            return np.nan

        # 3. Calculate means using nanmean to ignore existing NaNs
        # Axis (1, 2) for (C, H, W) format
        if array.ndim == 3 and array.shape[0] == 3:
            channel_means = np.nanmean(array, axis=(1, 2))
        # Axis (0, 1) for (H, W, C) format
        elif array.ndim == 3 and array.shape[2] == 3:
            channel_means = np.nanmean(array, axis=(0, 1))
        else:
            channel_means = np.nanmean(array)
            
        # 4. If result is an array (3 channels), return the average of them
        # so it returns a single scalar value for 'original_avg_channel'
        if isinstance(channel_means, np.ndarray):
            return np.nanmean(channel_means)
            
        return channel_means
    except Exception as e:
        # Helpful for debugging: print(f"Error in darkness_mean: {e}")
        return np.nan

def non_positive_percentage(arr: np.ndarray) -> float:
    """
    Calculates the percentage of elements in an array that are <= 0.
    
    Args:
        arr: The input NumPy array (any shape).
        
    Returns:
        float: Percentage of values <= 0 (from 0.0 to 100.0).
    """
    # Create a boolean mask where True indicates value is <= 0
    non_positive_mask = (arr <= 0)
    
    # Calculate percentage: (number of matches / total elements) * 100
    percentage = np.mean(non_positive_mask)
    
    return percentage

def compare_metrics(plaintext_array, secure_array):

    # A. L1 Norm (Manhattan Distance)
    l1_norm = np.sum(np.abs(plaintext_array - secure_array))
    # B. Mean Squared Error (MSE)
    mse = np.mean((plaintext_array - secure_array)**2)
    # C. Pearson Correlation Coefficient
    corr, _ = pearsonr(plaintext_array.flatten(), secure_array.flatten())
    return l1_norm, mse, corr

def calculate_sparsity(heatmap_array: np.ndarray) -> float:
    """
    Calculates the Sparsity of a heatmap array using the formula:
    Sparsity(H(x)) = 1 - 2 * sum_{i=1}^{N} [ H(x)_i / ||H(x)||_1 ] * [ (N - i + 0.5) / N ]
    
    CRITICAL: H(x)_i are the absolute values of the flattened heatmap, 
    sorted in ASCENDING order (i=1 is the smallest value).
    """

    abs_heatmap = np.abs(heatmap_array.flatten())
    N = abs_heatmap.size
    L1_norm = np.sum(abs_heatmap)
    if N == 0 or L1_norm == 0:
        return 1.0 
    H_sorted_abs = np.sort(abs_heatmap)

    i_1_to_N = np.arange(1, N + 1) 
    index_term = (N - i_1_to_N + 0.5) / N
    
    normalized_weights = H_sorted_abs / L1_norm
    weighted_sum = np.sum(normalized_weights * index_term)
    
    sparsity = 1.0 - 2.0 * weighted_sum

    return sparsity

def calculate_sensitivity(original_heatmap: np.ndarray, 
    perturbed_heatmap: np.ndarray
) -> float:
    """
    Calculates the normalized Sensitivity-n (Sens_norm) metric, 
    which is a measure of sensitivity.
    
    Sens_norm = ||H(x) - H(x + epsilon)||_2 / ||H(x)||_2
    
    H(x) is the original_heatmap, H(x + epsilon) is the perturbed_heatmap.
    
    Args:
        original_heatmap: The heatmap array for the original input.
        perturbed_heatmap: The heatmap array for the input with noise epsilon.
        
    Returns:
        The Sens_norm metric (sensitivity score). Closer to 0 is more robust.
    """

    H_original = original_heatmap.flatten()
    H_perturbed = perturbed_heatmap.flatten()

    if H_original.size != H_perturbed.size:
        raise ValueError("Original and perturbed heatmaps must have the same shape.")

    L2_norm_original = np.linalg.norm(H_original, ord=2)
    
    if L2_norm_original == 0:
        return 0.0
    
    difference_vector = H_original - H_perturbed
    L2_norm_difference = np.linalg.norm(difference_vector, ord=2)
    
    sens_norm = L2_norm_difference / L2_norm_original
    return sens_norm

def calculate_consistency(plaintext_heatmap: np.ndarray, 
    secure_heatmap: np.ndarray
) -> float:
    """
    Calculates the Consistency between two heatmaps using the Structural Similarity Index Measure (SSIM).
    
    SSIM = ( (2*mu1*mu2 + C) * (2*sigma12 + D) ) / ( (mu1^2 + mu2^2 + C) * (sigma1^2 + sigma2^2 + D) )
    
    Args:
        plaintext_heatmap: The heatmap array from the plaintext (H1).
        secure_heatmap: The heatmap array from the secure setting (H2).
        
    Returns:
        The SSIM value (consistency score). Closer to 1 is more consistent.
    """
    def to_grayscale(H):
        # Case: (C, H, W) -> e.g., (3, 224, 224)
        if H.ndim == 3 and H.shape[0] == 3:
            return 0.2989 * H[0] + 0.5870 * H[1] + 0.1140 * H[2]
        
        # Case: (H, W, C) -> e.g., (224, 224, 3)
        if H.ndim == 3 and H.shape[2] == 3:
            return 0.2989 * H[:,:,0] + 0.5870 * H[:,:,1] + 0.1140 * H[:,:,2]
        
        # Already 2D/1-channel
        return H

    H1 = to_grayscale(plaintext_heatmap).flatten()
    H2 = to_grayscale(secure_heatmap).flatten()

    min_val = min(H1.min(), H2.min())
    max_val = max(H1.max(), H2.max())
    L = max_val - min_val
    
    K = 0.01
    K_prime = 0.03

    C = (K * L)**2
    D = (K_prime * L)**2
    
    if L == 0:
        return 1.0 if np.all(H1 == H2) else 0.0
    
    mu1 = np.mean(H1)
    mu2 = np.mean(H2)
    
    sigma1_sq = np.var(H1, ddof=0)
    sigma2_sq = np.var(H2, ddof=0)
    
    sigma12 = np.mean((H1 - mu1) * (H2 - mu2))

    num_term1 = 2 * mu1 * mu2 + C
    num_term2 = 2 * sigma12 + D
    numerator = num_term1 * num_term2

    den_term1 = mu1**2 + mu2**2 + C
    den_term2 = sigma1_sq + sigma2_sq + D
    denominator = den_term1 * den_term2
    
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
        
    ssim_value = numerator / denominator
    return ssim_value

def plot_comparison(save, d_name, m_name, original_array, plaintext_array, secure_array, diff_array, mse, title, output_name):

    TITLE_FONTSIZE = 14 
    
    # Define scaling for heatmaps
    if "normed" in output_name:
        combined_arrays = np.concatenate((plaintext_array.flatten(), secure_array.flatten()))
        h_kwargs = {
            'vmin': np.min(combined_arrays),
            'vmax': np.max(combined_arrays)
        }
        # For normalized difference, 0 to 1 (or 2) is sensible
        diff_kwargs = {'vmin': 0, 'vmax': 5}
    else:
        # Empty dict means matplotlib auto-scales (vmax/vmin not set)
        h_kwargs = {}
        diff_kwargs = {}
        
    fig, axes = plt.subplots(1, 4, figsize=(18, 5)) 

    # 1. Original Input
    axes[0].imshow(original_array)
    axes[0].set_title("Original Input", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=15)
    axes[0].axis('off')

    # 2. Plaintext HeatMap - Using unpacked kwargs
    im1 = axes[1].imshow(plaintext_array, cmap='inferno', origin='lower', **h_kwargs)
    axes[1].set_title("Plaintext HeatMap", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=15)
    axes[1].axis('off')

    # 3. Secure HeatMap - Using unpacked kwargs
    im2 = axes[2].imshow(secure_array, cmap='inferno', origin='lower', **h_kwargs)
    axes[2].set_title("Secure HeatMap", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=15)
    axes[2].axis('off')

    # 4. Absolute Difference - Using unpacked kwargs
    im3 = axes[3].imshow(diff_array, cmap='hot', origin='lower', **diff_kwargs)
    axes[3].set_title("Absolute Difference", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=15)
    axes[3].axis('off')

    # Colorbars
    cbar_ax_left = fig.add_axes([0.29, 0.23, 0.015, 0.53])
    fig.colorbar(im1, cax=cbar_ax_left)

    cbar_ax_mid = fig.add_axes([0.5, 0.23, 0.015, 0.53])
    fig.colorbar(im2, cax=cbar_ax_mid)

    cbar_diff_ax_right = fig.add_axes([0.713, 0.23, 0.015, 0.53])
    cbar_diff_ax_right_col = fig.colorbar(im3, cax=cbar_diff_ax_right)
    cbar_diff_ax_right_col.set_ticks([])

    plt.subplots_adjust(wspace=0.4) 
    fig.suptitle(None)
    plt.savefig(output_name, bbox_inches='tight') 
    plt.close(fig)
