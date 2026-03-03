import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF

import numpy as np
import os
import time

from typing import Callable, Union

class AttributionBase:
    """
    Base class providing core utilities for gradient-based attribution, 
    especially the robust dFc/dI calculation.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        
    def _compute_input_gradient(self, args, input_tensor: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Computes the gradient of the target class score (Fc) with respect to the input (I).
        Uses torch.autograd.grad for robustness.
        """
        input_to_explain = input_tensor.clone().requires_grad_(True)
        self.model.zero_grad()
        start = time.perf_counter()
        output = self.model(input_to_explain)
        end = time.perf_counter()
        forward_time = end-start
        start = time.perf_counter()
        target_score = output[0, target_class]
        gradient_list = torch.autograd.grad(
            outputs=target_score, 
            inputs=input_to_explain, 
            retain_graph=True, 
            create_graph=False 
        )
        result = gradient_list[0].data.clone()
        end = time.perf_counter()
        back_time = end-start
        return result, output, forward_time, back_time

    def _aggregate_channels(self, attribution_map: torch.Tensor) -> torch.Tensor:
        """Aggregates the (B, C, H, W) attribution map to (H, W) for visualization."""
        # Common practice is to take the max across the channel dimension (dim=1)
        # after taking the absolute value (for saliency/importance magnitude).
        start = time.perf_counter()
        agg = attribution_map.abs().max(dim=1, keepdim=False)[0].squeeze()
        end = time.perf_counter()
        agg_time = end-start
        return agg, agg_time

    def attribute(self, input_tensor: torch.Tensor, target_class: int, baseline) -> torch.Tensor:
        """Abstract method to be implemented by derived classes."""
        raise NotImplementedError("Subclasses must implement the 'attribute' method.")
    
class VanillaGradients(AttributionBase):
    def attribute(self, args, input_tensor: torch.Tensor, target_class: int, baseline) -> torch.Tensor:
        """
        Computes the Vanilla Gradient Saliency Map.

        Returns:
            A (H, W) tensor representing the saliency map.
        """
        xai_result, output, forward_time, back_time = self._compute_input_gradient(args, input_tensor, target_class)
        saliency_map, agg_time = self._aggregate_channels(xai_result)
        times = {}
        times["forward pass"] = forward_time
        times["backpropagation"] = back_time
        times["aggregation"] = agg_time
        return saliency_map, output, times
    
class GradientxInput(AttributionBase):
    def attribute(self, args, input_tensor: torch.Tensor, target_class: int, baseline) -> torch.Tensor:
        """
        Computes the Gradient * Input attribution map.

        Returns:
            A (H, W) tensor representing the attribution map.
        """
        xai_result, output, forward_time, back_time = self._compute_input_gradient(args, input_tensor, target_class)
        start = time.perf_counter()
        g_times_i_map_raw = xai_result * input_tensor
        end = time.perf_counter()
        mult_time = end-start
        saliency_map, agg_time = self._aggregate_channels(g_times_i_map_raw)
        times = {}
        times["forward pass"] = forward_time
        times["backpropagation"] = back_time
        times["input multiplication"] = mult_time
        times["aggregation"] = agg_time
        return saliency_map, output, times
    
class IntegratedGradients(AttributionBase):
    def __init__(self, model: nn.Module, steps: int = 2):
        super().__init__(model)
        self.steps = steps
    def attribute(self, args, input_tensor: torch.Tensor, target_class: int, baseline) -> torch.Tensor:
        """
        Computes the Integrated Gradients (IG) attribution map.

        Args:
            baseline: Reference input (I'). Defaults to a black image (zeros).
            
        Returns:
            A (H, W) tensor representing the IG map.
        """
        if baseline.shape != input_tensor.shape:
            raise ValueError("Baseline tensor must have the same shape as the input tensor.")
        
        times = {}
        accumulated_gradients = torch.zeros_like(input_tensor)
        start = time.perf_counter()
        difference_vector = input_tensor - baseline
        end = time.perf_counter()
        times["delta calculation"] = end-start

        forward_time = 0
        back_time = 0
        interpol_time = 0
        start = time.perf_counter()

        # 1. Riemann Sum Approximation
        for m in range(1, self.steps + 1):
            start_a = time.perf_counter()
            alpha = m / self.steps
            interpolated_input = baseline + alpha * difference_vector
            end_a = time.perf_counter()
            interpol_time += (end_a-start_a)
            times[f"interpolation {m}"] = end_a-start_a
            gradient_at_point, _, forward_i, back_i = self._compute_input_gradient(args, interpolated_input, target_class)
            forward_time += forward_i
            back_time += back_i
            times[f"forward pass {m}"] = forward_i
            times[f"backpropagation {m}"] = back_i
            accumulated_gradients += gradient_at_point
        end = time.perf_counter()
        times["total interpolations"] = interpol_time
        times["total forward passes"] = forward_time
        times["total backpropagations"] = back_time
        times["riemann sum"] = end-start

        start = time.perf_counter()
        avg_gradient = accumulated_gradients / self.steps
        integrated_gradients_raw = difference_vector * avg_gradient
        end = time.perf_counter()
        times["integration"] = end-start
        saliency_map, agg_time = self._aggregate_channels(integrated_gradients_raw)
        times["aggregation"] = agg_time
        return saliency_map, _ ,times
    
class GradCAM(AttributionBase):
    def __init__(self, model: nn.Module, target_layer_name: str):
        super().__init__(model)
        self.target_layer = dict(self.model.named_modules()).get(target_layer_name)
        if self.target_layer is None:
            raise ValueError(f"Target layer '{target_layer_name}' not found.")
        
        self.gradients = None
        self.activations = None

    def _save_gradients(self, grad):
        self.gradients = grad

    def _save_activations(self, module, input, output):
        self.activations = output
        output.register_hook(self._save_gradients)

    def attribute(self, args, input_tensor: torch.Tensor, target_class: int, baseline) -> torch.Tensor:
        self.model.eval()
        handle = self.target_layer.register_forward_hook(self._save_activations)
        
        input_to_explain = input_tensor.clone().requires_grad_(True)
        output = self.model(input_to_explain)
        target_score = output[0, target_class]

        if target_score.item() == 0:
            print(f"[WARNING] Target class {target_class} has a raw score of 0. Backprop may fail.")
        else:
            print("[DEBUG] Target classn not zero")
        self.model.zero_grad()
        target_score.backward(retain_graph=True) # Keep graph for Guided Grad-CAM if needed
        
        if self.gradients is None:
            raise RuntimeError("Gradients were not captured. Check if target_layer is correct.")
            
        grad_magnitude = self.gradients.abs().sum().item()
        if grad_magnitude == 0:
            print("[ERROR] Captured gradients at target layer are exactly ZERO. Investigation required.")
        else:
            print(f"[DEBUG] Signal Check: Gradient sum at target layer = {grad_magnitude:.6e}")
        
        
        handle.remove()

        # --- 1. SHAPE VERIFICATION ---
        if self.gradients.shape != self.activations.shape:
            print(f"[DEBUG] SHAPE MISMATCH! Grads: {self.gradients.shape} | Acts: {self.activations.shape}")
        else:
            print(f"[DEBUG] Shape Match OK: {self.gradients.shape}")

        # --- 2. GRADIENT SPARSITY CHECK ---
        grad_flat = self.gradients.detach().cpu().numpy().flatten()
        zero_threshold = 1e-10
        sparsity = np.mean(np.abs(grad_flat) < zero_threshold) * 100
        print(f"[DEBUG] Gradient Sparsity: {sparsity:.2f}% of gradients are near-zero.")

        # --- 3. CAM CALCULATION ---
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam_raw = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # --- 4. PRE-RELU DIAGNOSTICS ---
        cam_flat = cam_raw.detach().cpu().numpy().flatten()
        neg_pct = np.mean(cam_flat < 0) * 100
        zero_pct = np.mean(np.abs(cam_flat) < zero_threshold) * 100
        print(f"[DEBUG] Pre-ReLU Stats: Negative={neg_pct:.2f}%, Zero={zero_pct:.2f}%")

        # --- 5. NORMALIZATION COMPARISON ---
        cam_relu = F.relu(cam_raw)
        
        # Method A: Min-Max (Current)
        cam_min_max = cam_relu.clone()
        cam_min_max -= cam_min_max.min()
        cam_min_max /= (cam_min_max.max() + 1e-7)
        
        # Method B: Max-Only (Proposed)
        cam_max_only = cam_relu.clone()
        cam_max_only /= (cam_max_only.max() + 1e-7)
        
        norm_diff = torch.abs(cam_min_max - cam_max_only).mean().item()
        print(f"[DEBUG] Normalization Difference (Mean Abs): {norm_diff:.6f}")
        
        # Use Max-Only for final result as it is more robust to "background noise"
        #cammax = cam_max_only

        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam /= (cam.max() + 1e-7)

        for name, cam_tensor in [("Min-Max Norm", cam), ("Max-Only Norm", cammax)]:
            print(f"\n--- Currently processing: {name} ---")

            # 6. Upsample
            final_heatmap = F.interpolate(
                cam_tensor, size=(input_tensor.shape[-2], input_tensor.shape[-1]), 
                mode='bilinear', align_corners=False
            ).squeeze()

            # --- 7. GUIDED GRAD-CAM OPTION ---
            vanilla=False

            if vanilla:
                print("[DEBUG] Applying Guided Grad-CAM (Grad-CAM * Vanilla Gradient)")
                vg = VanillaGradients(self.model)
                # We use the same parameters to get the high-res saliency map
                v_saliency = vg.attribute(args, input_tensor, target_class, baseline)
                
                # Element-wise multiplication to "gate" the noise
                final_heatmap = final_heatmap * v_saliency
            
                display_map = final_heatmap.detach().cpu().numpy()

                plt.figure(figsize=(8, 8))
                plt.imshow(display_map, cmap='jet') # 'jet' or 'viridis' are standard for CAM
                plt.axis('off')
                
                # Construct dynamic filename
                # You can expand this to include args.model or args.dataset if available in the method
                save_name = f"{method_name}_idx{target_class}_vanilla_{str(vanilla).lower()}_{name}.png"
                
                # Ensure the output directory exists
                save_path = project_root / "output" / "debug_maps"
                save_path.mkdir(parents=True, exist_ok=True)
                
                full_path = save_path / save_name
                plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
                plt.close() # Close to free up memory
                
                print(f"✅ Debug Heatmap saved to: {full_path}")

        return final_heatmap.detach()
        
# class GradCAM(AttributionBase):
#     def __init__(self, model: nn.Module, target_layer_name: str):
#         super().__init__(model)
#         # Find the target module
#         # Run this to find your target_layer_name
#         for name, module in model.named_modules():
#             if isinstance(module, torch.nn.Conv2d):
#                 print(f"Layer Name: {name} | Type: {module}")
#         self.target_layer = dict(self.model.named_modules()).get(target_layer_name)
#         if self.target_layer is None:
#             raise ValueError(f"Target layer '{target_layer_name}' not found.")
        
#         self.gradients = None
#         self.activations = None

#     def _save_gradients(self, grad):
#         self.gradients = grad

#     def _save_activations(self, module, input, output):
#         self.activations = output
#         # This is the "Secret Sauce": Register hook directly on the output tensor
#         # This captures the gradient exactly as it passes through this specific tensor
#         output.register_hook(self._save_gradients)

#     def attribute(self, args, input_tensor: torch.Tensor, target_class: int, baseline) -> torch.Tensor:
#         self.model.eval()
        
#         # 1. Register hooks
#         handle = self.target_layer.register_forward_hook(self._save_activations)
        
#         # 2. Forward pass
#         input_to_explain = input_tensor.clone().requires_grad_(True)
#         output = self.model(input_to_explain)
        
#         # 3. Targeted Backward pass
#         # Use logits, not softmax output, to prevent vanishing gradients
#         target_score = output[0, target_class]
#         self.model.zero_grad()
#         target_score.backward(retain_graph=False)
        
#         # 4. Remove hook
#         handle.remove()

#         if self.gradients is None or self.activations is None:
#             raise RuntimeError("Hooks failed to capture gradients/activations. Is target_layer_name correct?")

#         # 5. Calculation
#         # G_k: [Batch, Channels, H, W] -> alpha_k: [Batch, Channels, 1, 1]
#         weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
#         # Linear combination of activations
#         cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
#         # Apply ReLU to keep only features that have a POSITIVE influence on the class
#         cam = F.relu(cam)
        
#         # Normalize between 0 and 1 (prevents all-zero maps if values are tiny)
#         cam -= cam.min()
#         cam /= (cam.max() + 1e-7)

#         print("cam shape: ", cam.shape)
#         # 6. Upsample to original image size
#         final_heatmap = F.interpolate(
#             cam, 
#             size=(input_tensor.shape[-2], input_tensor.shape[-1]), 
#             mode='bilinear', 
#             align_corners=False
#         )
        
#         print(f"[DEBUG] After Interpolation:  {final_heatmap.shape}")
        
#         final_heatmap = final_heatmap.squeeze()
        
#         print(f"[DEBUG] Final Squeezed Shape: {final_heatmap.shape}")


#         return final_heatmap.detach()

class LayerwiseRelevancePropagation(AttributionBase):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.activations = {}
        self.handles = []
        
    def _save_activations(self, name: str) -> Callable:
        def hook(module, input, output):
            # We save the input and output
            self.activations[name + '_input'] = input[0].detach().squeeze(0)
            self.activations[name + '_output'] = output.detach().squeeze(0)
        return hook

    def _lrp_epsilon_rule(self, args, name, module: nn.Module, relevance_input: torch.Tensor, activation_input: torch.Tensor) -> torch.Tensor:
        """
        Modified Epsilon Rule with Sign-Aware Denominator to prevent exploding relevance.
        """
        val_relevance = relevance_input.unsqueeze(0) if relevance_input.dim() < 4 else relevance_input
        val_activation = activation_input.unsqueeze(0).detach().requires_grad_(True)
        
        eps = getattr(args, 'epsilon', 1e-9)

        # 1. Local Forward Pass (Ignoring Bias as is standard for LRP-epsilon)
        if isinstance(module, nn.Linear):
            z = F.linear(val_activation, module.weight, bias=None)
        elif isinstance(module, nn.Conv2d):
            z = F.conv2d(val_activation, module.weight, bias=None, 
                         stride=module.stride, padding=module.padding)
        else:
            return relevance_input

        # 2. SIGN-AWARE EPSILON
        # Prevents denominator from approaching zero when z is negative
        z = z + (eps * (z >= 0).float()) - (eps * (z < 0).float())

        # 3. Backward Projection
        s = (val_relevance / z).detach()
        (z * s).sum().backward()
        
        # 4. Result: input * gradient
        r_j = (val_activation * val_activation.grad).squeeze(0).detach()

        # --- DIAGNOSTIC: THE RATIO COLUMN ---
        in_sum = relevance_input.sum().item()
        out_sum = r_j.sum().item()
        ratio = out_sum / in_sum if abs(in_sum) > 1e-12 else 0
        
        status = "OK"
        if ratio == 0: status = "SIGNAL DIED"
        elif abs(1 - ratio) > 0.1: status = "BIAS/LEAKAGE"

        print(f"[DEBUG] Layer {name:<18} | In: {in_sum:10.3e} | Out: {out_sum:10.3e} | Ratio: {ratio:.4f} | {status}")
        
        return r_j

    def _propagate_relevance(self, args, relevance: torch.Tensor, name: str, module: nn.Module) -> torch.Tensor:
        if name + '_output' not in self.activations:
            return relevance

        output_act = self.activations[name + '_output']
        input_act = self.activations[name + '_input']

        # --- DIAGNOSTIC: FLATTEN TRANSITION (AlexNet/VGG Killer) ---
        if relevance.shape != output_act.shape:
            sparsity = (relevance == 0).float().mean().item() * 100
            print(f"[DEBUG] Shape Mismatch at {name}: {list(relevance.shape)} -> {list(output_act.shape)} (Sparsity: {sparsity:.1f}%)")
            try:
                relevance = relevance.reshape(output_act.shape)
            except RuntimeError:
                print(f"[ERROR] Failed to bridge Flatten transition at {name}. Check for pooling layers.")
                return torch.zeros_like(output_act)

        if isinstance(module, (nn.Conv2d, nn.Linear)):
            return self._lrp_epsilon_rule(args, name, module, relevance, input_act)
        elif isinstance(module, nn.MaxPool2d):
            return self._lrp_maxpool_rule(module, relevance, input_act)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            # Simple uniform redistribution for Global Average Pooling
            return (relevance / (output_act.numel() + 1e-12)).expand_as(input_act)
            
        return relevance

    def _save_final_heatmap(self, heatmap_tensor, args, target_class):
        """Converts tensor to image and saves to disk."""
        h_map = heatmap_tensor.cpu().numpy()
        
        plt.figure(figsize=(6, 6))
        plt.imshow(h_map, cmap='hot') # LRP is traditionally viewed with 'hot' or 'seismic'
        plt.axis('off')
        
        # Build path: output/maps/LRP/model/dataset/
        out_dir = project_root / "output" / "debug_lrp" / args.model / args.dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"LRP_idx{target_class}_eps{args.epsilon}.png"
        save_path = out_dir / filename
        
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"✅ LRP Map saved: {save_path}")
        
    def attribute(self, args, input_tensor: torch.Tensor, target_class: int, baseline) -> torch.Tensor:
        print(f"\n--- LRP Backward Pass: {args.model} on {args.dataset} ---")
        self.activations = {}
        self.handles = []
        
        # Register hooks
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.ReLU, nn.AdaptiveAvgPool2d)):
                self.handles.append(module.register_forward_hook(self._save_activations(name)))

        # 1. Forward Pass
        # --- DIAGNOSTIC: LOGITS VS SOFTMAX ---
        output = self.model(input_tensor)
        
        # 2. Initialize Relevance
        # We use raw logits (output) to prevent vanishing gradients from softmax
        R = torch.zeros_like(output)
        R[0, target_class] = output[0, target_class]
        current_relevance = R.squeeze(0) 
        
        print(f"[DEBUG] Initial Logit Score for class {target_class}: {output[0, target_class].item():.4f}")

        # 3. Backward Loop
        reversed_modules = list(self.model.named_modules())
        reversed_modules.reverse()
        
        for name, module in reversed_modules:
            current_relevance = self._propagate_relevance(args, current_relevance, name, module)
        
        # 4. Final Aggregation
        if current_relevance.dim() == 3: # (C, H, W)
            final_lrp_map = current_relevance.abs().sum(dim=0)
        else: # Vector (MNIST case)
            side = int(current_relevance.numel()**0.5)
            final_lrp_map = current_relevance.view(side, side)
       
        self._save_final_heatmap(final_lrp_map, args, target_class)
        
        for handle in self.handles: handle.remove()
        
        return final_lrp_map.detach()

# class LayerwiseRelevancePropagation:
#     def __init__(self, model: nn.Module):
#         self.model = model
#         self.model.eval()
#         self.relevance_map = None
#         self.activations = {}
#         self.handles = []
        
#     def _save_activations(self, name: str) -> Callable:
#         """Saves activations, ensuring they are always stripped of batch dim for internal storage."""
#         def hook(module, input, output):
#             # We save the input and output (squeezing batch dim)
#             self.activations[name + '_input'] = input[0].detach().squeeze(0)
#             self.activations[name + '_output'] = output.detach().squeeze(0)
#         return hook

#     def _lrp_epsilon_rule(self, args, module: nn.Module, relevance_input: torch.Tensor, activation_input: torch.Tensor) -> torch.Tensor:
#         """
#         Internal Rule: Projection R_j = a_j * [ (R_k / Z_k) @ W^T ]
#         """
#         # 1. Prepare dimensions
#         val_relevance = relevance_input.unsqueeze(0) if relevance_input.dim() < 4 else relevance_input
#         val_activation = activation_input.unsqueeze(0).detach().requires_grad_(True)
        
#         # 2. Local Forward Pass
#         if isinstance(module, nn.Linear):
#             z = F.linear(val_activation, module.weight, bias=None) + args.epsilon
#         elif isinstance(module, nn.Conv2d):
#             z = F.conv2d(val_activation, module.weight, bias=None, 
#                          stride=module.stride, padding=module.padding) + args.epsilon
#         else:
#             return relevance_input

#         # 3. Element-wise division (Conservation factor)
#         s = (val_relevance / z).detach()

#         # 4. Backward projection
#         (z * s).sum().backward()
        
#         # 5. Result: input * gradient
#         r_j = (val_activation * val_activation.grad).squeeze(0).detach()
#         return r_j

#     def _lrp_maxpool_rule(self, module: nn.Module, relevance_input: torch.Tensor, input_act: torch.Tensor) -> torch.Tensor:
#         """Propagates relevance only to the max activation location."""
#         # Ensure 4D
#         relevance_4d = relevance_input.unsqueeze(0)
#         input_act_4d = input_act.unsqueeze(0)
        
#         _, max_indices = F.max_pool2d(
#             input_act_4d, 
#             kernel_size=module.kernel_size, 
#             stride=module.stride, 
#             padding=module.padding, 
#             return_indices=True
#         )
        
#         R_j = F.max_unpool2d(
#             relevance_4d, 
#             max_indices, 
#             kernel_size=module.kernel_size, 
#             stride=module.stride, 
#             padding=module.padding, 
#             output_size=input_act_4d.shape 
#         )
#         return R_j.squeeze(0)

#     def _propagate_relevance(self, args, relevance: torch.Tensor, name: str, module: nn.Module) -> torch.Tensor:
#         """Determines which internal rule to apply based on layer type."""
        
#         # Get expected shape of the layer's output (where relevance is coming from)
#         output_act = self.activations[name + '_output']
#         input_act = self.activations[name + '_input']

#         # Shape Check: Handle the Flattening transition (e.g., from FC1 back to Conv5)
#         if relevance.shape != output_act.shape:
#             # Case 1: Both are 1D but different sizes (Common in custom Linear layers)
#             if relevance.dim() == 1 and output_act.dim() == 1:
#                 new_relevance = torch.zeros_like(output_act)
#                 # Fill what we can (min of the two sizes)
#                 size_to_copy = min(relevance.size(0), output_act.size(0))
#                 new_relevance[:size_to_copy] = relevance[:size_to_copy]
#                 relevance = new_relevance
                
#             # Case 2: Relevance is 1D, but we are entering a Conv layer (3D)
#             elif relevance.dim() == 1 and output_act.dim() == 3:
#                 if relevance.numel() != output_act.numel():
#                     # Truncate or Pad to match output_act.numel()
#                     temp_rel = torch.zeros(output_act.numel(), device=relevance.device)
#                     size = min(relevance.numel(), output_act.numel())
#                     temp_rel[:size] = relevance[:size]
#                     relevance = temp_rel.view(output_act.shape)
#                 else:
#                     relevance = relevance.view(output_act.shape)
            
#             # Case 3: Final fallback to prevent crash
#             else:
#                 try:
#                     relevance = relevance.reshape(output_act.shape)
#                 except RuntimeError:
#                     # If all else fails, interpolate or pad
#                     print(f"Extreme shape mismatch at {name}: {relevance.shape} vs {output_act.shape}")
#                     # Create a zero tensor of the right shape and copy what fits
#                     new_rel = torch.zeros_like(output_act)
#                     # This is a 'best effort' copy
#                     return new_rel
                    
#         if isinstance(module, (nn.Conv2d, nn.Linear)):
#             return self._lrp_epsilon_rule(args, module, relevance, input_act)
            
#         elif isinstance(module, nn.MaxPool2d):
#             return self._lrp_maxpool_rule(module, relevance, input_act)
            
#         elif isinstance(module, (nn.ReLU, nn.BatchNorm2d, nn.Dropout)):
#             return relevance # Passthrough
            
#         return relevance

#     def attribute(self, args, input_tensor: torch.Tensor, target_class: int, baseline) -> torch.Tensor:
#         """Main method to perform LRP decomposition."""
        

#         self.activations = {}
#         self.handles = []
#         for name, module in self.model.named_modules():
#             if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.ReLU, nn.BatchNorm2d)):
#                 self.handles.append(module.register_forward_hook(self._save_activations(name)))

#         # 1. Forward Pass
#         with torch.no_grad():
#             output = self.model(input_tensor)
            
#         # 2. Initialize Relevance at the output layer
#         R = output.detach().clone()
#         R_L = torch.zeros_like(R)
#         R_L[0, target_class] = R[0, target_class]
#         current_relevance = R_L.squeeze(0) 

#         # 3. Backward Loop
#         reversed_modules = [(name, module) for name, module in self.model.named_modules() 
#                             if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.ReLU, nn.BatchNorm2d))]
#         reversed_modules.reverse()
        
#         start_time = time.perf_counter()
#         for name, module in reversed_modules:
#             if name + '_input' not in self.activations:
#                 continue 

#             current_relevance = self._propagate_relevance(args, current_relevance, name, module)
        
#         end_time = time.perf_counter()

#         # 4. Final Relevance Map (R_0)
#         self.relevance_map = current_relevance 
        
#         # 5. Cleanup
#         for handle in self.handles:
#             handle.remove()
#         self.handles = []
#         self.activations = {}

#         # 6. Aggregate channels (C, H, W) -> (H, W)
#         if self.relevance_map.dim() == 1 or args.model == "AliceNet":
#             side = int(self.relevance_map.shape[0]**0.5)
#             final_lrp_map = self.relevance_map.view(side, side)
#         else:
#             final_lrp_map = self.relevance_map.abs().sum(dim=0).squeeze()
        
#         return final_lrp_map


