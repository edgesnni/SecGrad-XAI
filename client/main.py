import os, sys
import crypten
import crypten.communicator as comm
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from plaintext.models import *
from plaintext.explainables import *
from secure.secure_models import *
from secure.secure_explainables import *
from utils import *
from config import *

from pathlib import Path
import logging

import time


if __name__ == "__main__":
    set_seed(1)
    args = parsing_all()

    if args.mode == "secure":
        try:
            
            crypten.init()
            crypten.config.precision_bits = 32
            comm.get().barrier()
        except:
            raise ValueError("Crypten not initialized")
        
        args_server = [args.dataset, args.model, args.explain, args.steps, args.epsilon, args.normalized]
        comm.get().barrier()
        comm.get().broadcast_obj(args_server, src=1)
    
    args = rename_inputs(args)

    if args.inference == "multiple":
        suffix = f"_entire_testset{args.is_positive}"
    else:
        suffix = f"_{args.explain}{args.is_positive}" 
        
    log_directory = Path("output") / f"logs{args.is_positive}" / f"logs{suffix}"
    log_directory.mkdir(parents=True, exist_ok=True)
    filename = f"logs_{args.identity}{args.is_positive}_{args.explain}.txt"
    log_file_path = log_directory / filename
    log_file_path.touch(exist_ok=True)
    logging_intro(log_file_path)
    if args.mode == "secure":
        comm.get().barrier()
    map_filename = f"{args.identity}{args.is_positive}_{args.explain}_{args.idx}.png"
    output_path = Path("output") / f"maps{args.is_positive}" / args.explain / map_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "secure":
        comm.get().barrier()
    # --- Prepare Datasets & Load Image(s) ---
    try:
        if args.inference == "single":
            test_dataset, num_classes, class_names = preprocess_data(args)
        elif args.inference == "multiple":
            test_loader, num_classes, class_names = preprocess_data(args)
    except Exception as e:
        raise ValueError(f"Error getting a sample from the dataset: {e}. \
                        Ensure Tiny-ImageNet files are correctly downloaded and the val_annotations.txt is present.")    
    
    print("✅ Pre-processing ended")

    # --- Load Models ---
    try:
        model = load_and_configure_model(args, num_classes, MODEL_CONFIG)
    except:
        raise ValueError("Model not loaded.")
    print("✅ Loading Model ended")


    # --- Setup Inference ---
    if args.mode == "plaintext":
        model, device = setup_plaintext_inference(args, model)
    else:
        encrypted_model, str_model, model_enc_time = setup_secure_inference(args, model)
        print(f"\t-> The model is {str_model}.")
        comm.get().barrier()
        
    # --- Load Explainability ---
    if args.explain and args.inference != "multiple":
        config = PLAINTEXT_CONFIG if args.mode == "plaintext" else SECURE_CONFIG
        model_instance = model if args.mode == "plaintext" else encrypted_model

        method_config = config[args.explain.split("_")[0]]
        constructor_args = [model_instance]

        #print("Available layers:", list(dict(model.named_modules()).keys()))
        
        if "integrated" in args.explain:                
            constructor_args.append(args.steps) 
        elif "CAM" in args.explain:
            if "VGG11" in args.model:
                if args.mode == "secure":
                    target_layer = "124"
                else:
                    target_layer = 'conv5_2'
            elif "VGG19" in args.model:
                if args.mode == "secure":
                    target_layer = "228"
                else:
                    target_layer = 'conv5_4'
            elif "AlexNet" in args.model:
                if args.mode == "secure":
                    target_layer = "95"
                else:
                    target_layer = 'conv5'
            else:
                raise ValueError(f"Don't use CAM with {args.model}")
            constructor_args.append(target_layer)
        elif "LRP" in args.explain:
            constructor_args.append(args.steps)
            

        # Instantiate the XAI class
        xai_method = method_config['class'](*constructor_args)
        METHOD_TITLE = method_config['title']
        METHOD_TITLE_SHORT = method_config['short_title']
        print(f"✅ XAI Method {METHOD_TITLE} initialized")
        
        if args.mode == "secure":
            comm.get().barrier()

        for i in range(len(test_dataset)):
            logging.info(f"\nXAI sample {i}:")
            print(f"\nXAI sample {i}:")
            sample = test_dataset[i]
            image, label = sample[0], sample[1]
            if args.mode == "secure":
                logging.info(f"\tmodel encryption: {(model_enc_time):.4f} s")
                print(f"\tmodel encryption: {(model_enc_time):.4f} s")
            
            if "integrated" in args.explain:
                start = time.perf_counter()
                if args.baseline == "zero":
                    baseline = torch.zeros_like(image.unsqueeze(0))
                elif args.baseline == "gray":
                    baseline = torch.full_like(image.unsqueeze(0), 0.5)
                elif args.baseline == "white":
                    baseline = torch.ones_like(image.unsqueeze(0))
                elif args.baseline == "random":
                    baseline = torch.randn_like(image.unsqueeze(0))
                    baseline = torch.clamp(baseline, 0, 1)
                elif args.baseline == "blur":
                    baseline = TVF.gaussian_blur(image.unsqueeze(0), kernel_size=(11, 11), sigma=(5.0, 5.0))
                else:
                    baseline = None
                end = time.perf_counter()
                logging.info(f"\tplaintext baseline creation: {(end-start):.4f} s")
                print(f"\tplaintext baseline creation: {(end-start):.4f} s")
            else:
                baseline = None
            
            # img_hash = hashlib.md5(image.numpy().tobytes()).hexdigest()
            # print(f"DEBUG: Sample {i} - Hash: {img_hash}")
            
            ###############################
            ######## STOP INFERENCE
            # if args.mode == "plaintext":
            #     input_tensor, output = run_plaintext_inference(model, image, device, args)
            # else:
            #     encrypt_tensor = crypten.cryptensor(image, src=1, requires_grad=False)
            #     input_tensor, output = run_secure_inference(encrypted_model, encrypt_tensor, args)
            # is_correct, predicted_class_idx = accuracy_single_inference(
            #     args, output    , label, class_names)
            # print(f"✅ Sample {i}: {args.mode.capitalize()} Inference ({is_correct})")
            ###############################
            

            if args.mode == "plaintext":
                input_tensor = image.unsqueeze(0).to(device)
            elif args.mode == "secure":
                
                start = time.perf_counter()
                comm.get().barrier()
                input_tensor = crypten.cryptensor(image.unsqueeze(0), src=1, requires_grad=False)
                end = time.perf_counter()
                logging.info(f"\tinput encryption: {(end-start):.4f} s")
                print(f"\tinput encryption: {(end-start):.4f} s")

                start = time.perf_counter()
                comm.get().barrier()
                comm.get().broadcast_obj(label, src=1)
                end = time.perf_counter()
                logging.info(f"\tlabel sent: {(end-start):.4f} s")
                print(f"\tlabel t: {(end-start):.4f} s")

                comm.get().barrier()
                saliency_result, predicted_class_idx = calculate_saliency(
                    attribution_function=xai_method.attribute,
                    model=model_instance,
                    input_tensor=input_tensor,
                    target_class=label,
                    class_names=class_names,
                    args=args, i=i, baseline=baseline)

            if args.mode == "secure":
                comm.get().barrier()
            if args.map == "yes" and i == args.idx:
                plot_saliency_map(
                    final_torch_input=image,
                    saliency_np=saliency_result,
                    target_class=label,
                    predicted_class=predicted_class_idx,
                    file_path=output_path,
                    title=METHOD_TITLE,
                    short_title=METHOD_TITLE_SHORT,
                    args=args
                )
                print(f"\t📸 Target map saved to: {map_filename}")

    # elif args.inference == "multiple":
    #     if args.mode == "plaintext":
    #         model, device = setup_plaintext_inference(args, model)
    #         inference_func = lambda model, inputs, device: run_plaintext_inference(model, inputs, device, args)
    #         all_labels, all_preds = run_multiple_inferences(args, model, test_loader, device, inference_func)
    #     else:
    #         encrypted_model, str_model = setup_secure_inference(args, model)
    #         inference_func = lambda model, inputs, device=None: run_secure_inference(model, inputs, args)
    #         all_labels, all_preds = run_multiple_inferences(args, encrypted_model, test_loader, None, inference_func)
    #     accuracy = calculate_and_log_metrics(args, all_labels, all_preds)
    #     print(f"✅ {args.mode.capitalize()} Batch Inference ended. Accuracy: {accuracy:.4f}")
    # else:   
    #     raise ValueError(f"Unknown inference mode: {args.inference}.")
    

    logging.info("\n=====================================================================")