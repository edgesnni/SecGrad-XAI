import os, sys
import crypten
import crypten.communicator as comm
import torch.distributed as dist
from plaintext.models import *
from secure.secure_models import *
from secure.secure_explainables import *
from utils import *
from config import *
from pathlib import Path
#import hashlib

if __name__ == "__main__":
    set_seed(1)
    args = parsing_all()

    try:
        
        crypten.init()
        crypten.config.precision_bits = 32
        comm.get().barrier()
    except:
        raise ValueError("Crypten not initialized")
    
    if args.mode == "plaintext":
        raise ValueError("Cannot do plaintext nor multiple inferences on this machine.")
    else:
        comm.get().barrier()
        args_client_list = comm.get().broadcast_obj(None, src=1)

    assert args_client_list[0] == args.dataset
    assert args_client_list[1] == args.model
    assert args_client_list[2] == args.explain
    args.steps = args_client_list[3]
    args.epsilon = args_client_list[4]
    args.normalized = args_client_list[5]
    if args.mode == "secure":
        comm.get().barrier()
    assert isinstance(args.steps, int)
    assert isinstance(args.epsilon, float)
    assert isinstance(args.normalized, str)
    assert len(args.normalized) in (2, 3)
    
    args = rename_inputs(args)
    comm.get().barrier()
    # --- Prepare Datasets & Load Image(s) ---
    try:
        test_dataset, num_classes, class_names = preprocess_data(args)
    except Exception as e:
        raise ValueError(f"Error getting a sample from the dataset: {e}. \
                        Ensure Tiny-ImageNet files are correctly downloaded and the val_annotations.txt is present.")    
    print("\t✅ Pre-processing ended")
    
    # --- Load Models ---
    try:
        model = load_and_configure_model(args, num_classes, MODEL_CONFIG)
    except:
        raise ValueError("Model not loaded.")
    print("\t✅ Loading Model ended")

    # --- Setup Inference ---
    if args.mode == "plaintext":
        raise ValueError("No plaintext inference on this machine.")
    else:
        encrypted_model, str_model = setup_secure_inference(args, model)
        print(f"\t-> The model is {str_model}.")
    comm.get().barrier()
    
    # --- Load Explainability ---

    config = SECURE_CONFIG
    method_config = config[args.explain.split("_")[0]]
    constructor_args = [encrypted_model]

    # Context-specific arguments
    if "integrated" in args.explain:
        constructor_args.append(args.steps) 
    elif "CAM" in args.explain:
        if "VGG11" in args.model:
            target_layer = "124"
            #target_layer = 'conv5_2'
        elif "VGG19" in args.model:
            target_layer = "228"
            #target_layer = 'conv5_4'
        elif "AlexNet" in args.model:
            target_layer = "95"
            #target_layer = 'conv5'
        else:
            raise ValueError(f"Don't use CAM with {args.model}")
        constructor_args.append(target_layer)
    elif "LRP" in args.explain:
        constructor_args.append(args.steps)
            

    # Instantiate the XAI class
    xai_method = method_config['class'](*constructor_args)
    METHOD_TITLE = method_config['title']
    METHOD_TITLE_SHORT = method_config['short_title']
    print(f"\t✅ XAI Method {METHOD_TITLE} initialized")

    comm.get().barrier()
    
    # --- Run Inference ---
    for i in range(len(test_dataset)):
        print(f"\nXAI sample {i}:")
        sample = test_dataset[i]
        image = sample[0]

        if args.mode == "plaintext":
            raise ValueError("No plaintext inference on this machine.")
        else:
            
            # input_tensor = run_secure_inference(encrypted_model, image, args)
            comm.get().barrier()
            input_tensor = crypten.cryptensor(image.unsqueeze(0), src=1, requires_grad=False)
            # print(f"\n\n✅ Sample {i}: {args.mode.capitalize()} Inference (Client has results)")
            comm.get().barrier()
            label = comm.get().broadcast_obj(None, src=1)

            comm.get().barrier()

            calculate_saliency(
                attribution_function=xai_method.attribute,
                model=encrypted_model,
                input_tensor=input_tensor,
                target_class=label,
                args=args)
            print(f"\t✅ Saliency calculated for index {i}")

            comm.get().barrier()
            