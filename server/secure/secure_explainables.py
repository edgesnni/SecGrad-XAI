import torch
import crypten
import crypten.nn as cnn
import crypten.mpc as mpc
import crypten.communicator as comm
from crypten.config import cfg

import numpy as np
import os

### GRADIENT BASED

def crypten_interpolate_nearest(tensor, size):
    """
    MPC-safe Bilinear Interpolation (align_corners=False).
    tensor: crypten.CrypTensor [B, C, H, W]
    size: (out_h, out_w) -> e.g., (224, 224)
    """
    B, C, in_h, in_w = tensor.size()
    out_h, out_w = size

    # 1. Pre-calculate coordinate mapping (Plaintext logic)
    # align_corners=False uses this specific scaling formula
    scale_h = in_h / out_h
    scale_w = in_w / out_w

    # Calculate center-aligned indices
    grid_h = (torch.arange(out_h).float() + 0.5) * scale_h - 0.5
    grid_w = (torch.arange(out_w).float() + 0.5) * scale_w - 0.5

    # Clamp to ensure we stay within [0, in_h-1]
    grid_h = grid_h.clamp(0, in_h - 1 - 1e-4)
    grid_w = grid_w.clamp(0, in_w - 1 - 1e-4)

    # 2. Find the 4 neighboring pixel indices
    h0 = grid_h.floor().long()
    h1 = (h0 + 1).clamp(0, in_h - 1)
    w0 = grid_w.floor().long()
    w1 = (w0 + 1).clamp(0, in_w - 1)

    # 3. Calculate weights (interpolants)
    # These are public/plaintext values
    wh = (grid_h - h0.float()).view(1, 1, out_h, 1)
    ww = (grid_w - w0.float()).view(1, 1, 1, out_w)

    # 4. MPC-Safe Bilinear Weighted Sum
    # We use advanced indexing to pick the values from the 13x13 grid
    # and "broadcast" them into the 224x224 space.
    
    # Pick corners
    v00 = tensor[:, :, h0, :][:, :, :, w0] # Top-Left
    v01 = tensor[:, :, h0, :][:, :, :, w1] # Top-Right
    v10 = tensor[:, :, h1, :][:, :, :, w0] # Bottom-Left
    v11 = tensor[:, :, h1, :][:, :, :, w1] # Bottom-Right

    # Bilinear Formula: 
    # f(x,y) = v00(1-wh)(1-ww) + v01(1-wh)ww + v10(wh)(1-ww) + v11(wh)(ww)
    
    # Step-by-step to save MPC multiplications
    top = v00 * (1 - ww) + v01 * ww
    bottom = v10 * (1 - ww) + v11 * ww
    final_output = top * (1 - wh) + bottom * wh

    return final_output


class SecureAttributionBase:
    """
    Base class providing core utilities for gradient-based attribution, 
    especially the robust dFc/dI calculation.
    """
    def __init__(self, model: cnn.Module):
        self.model = model
        self.model.eval()
    def _compute_input_gradient(self, args, input_tensor, target_class: int) -> crypten.CrypTensor:
        """
        Computes the gradient of the target class score (Fc) with respect to the input (I).
        Uses torch.autograd.grad for robustness.
        """
        with crypten.enable_grad():

            input_to_explain = input_tensor.detach()
            input_to_explain.is_leaf = True
            input_to_explain.requires_grad = True
            self.model.zero_grad()
            # --- VERIFICATION BLOCK START ---
            sd = self.model.state_dict()
            print("\n--- Scanning all buffers for vanishing values ---")
            for key, tensor in sd.items():
                p_tensor = tensor.get_plain_text()
                avg_val = p_tensor.abs().mean().item()
                min_val = p_tensor.min().item()
                print(avg_val)
                if 0 < avg_val < 0.001:
                    status = "!! RISKY !!" if avg_val < 0.0001 else "Low"
                    print(f"Key: {key:40} | Avg: {avg_val:.8f} | Min: {min_val:.8f} | [{status}]")
                else:
                    print("not risky")
            # --- VERIFICATION BLOCK END ---
            
            comm.get().barrier()
            output = self.model(input_to_explain)
            grad_output_torch = torch.zeros(output.size())
            grad_output_torch[0, target_class] = 1
            grad_output = crypten.cryptensor(grad_output_torch, src=1, requires_grad=True) 
            
            output.backward(grad_output)
            result = input_to_explain.grad     
        return result, output
    
    def _aggregate_channels(self, attribution_map: crypten.CrypTensor) -> crypten.CrypTensor:
        """Aggregates the (B, C, H, W) attribution map to (H, W) for visualization."""
        if attribution_map.dim() == 4:
            agg = attribution_map.abs().max(dim=1)[0]
        elif attribution_map.dim() == 3:
            agg = attribution_map.abs().max(dim=0)[0]
        else:
            agg = attribution_map.abs()
        agg = agg.squeeze()
        return agg

    def attribute(self, input_tensor: crypten.CrypTensor, target_class: int) -> crypten.CrypTensor:
        """Abstract method to be implemented by derived classes."""
        raise NotImplementedError("Subclasses must implement the 'attribute' method.")
    
class SecureVanillaGradients(SecureAttributionBase): 
    def attribute(self, args, input_tensor, target_class: int) -> crypten.CrypTensor:
        """
        Computes the Vanilla Gradient Saliency Map.

        Returns:
            A (H, W) tensor representing the saliency map.
        """

        xai_result, output = self._compute_input_gradient(args, input_tensor, target_class)
        secure_saliency_map = self._aggregate_channels(xai_result)
        
        return secure_saliency_map, output

class SecureGradientxInput(SecureAttributionBase):
    def attribute(self, args, input_tensor: crypten.CrypTensor, target_class: int) -> crypten.CrypTensor:
        """
        Computes the Secure Gradient * Input attribution map.
        """

        xai_result, output = self._compute_input_gradient(args, input_tensor, target_class)
        g_times_i_map_raw = xai_result * input_tensor
        secure_saliency_map= self._aggregate_channels(g_times_i_map_raw)
        
        return secure_saliency_map, output

class SecureIntegratedGradients(SecureAttributionBase):
    def __init__(self, model, steps: int = 2):
        super().__init__(model)
        self.steps = steps

    def attribute(self, args, input_tensor: crypten.CrypTensor, target_class: int) -> crypten.CrypTensor:
        """
        Computes the Secure Integrated Gradients (IG) attribution map.
        """
        comm.get().barrier()
        plaintext_baseline = comm.get().broadcast_obj(None, src=1)
        baseline = crypten.cryptensor(plaintext_baseline, src=1)
        accumulated_gradient_torch = torch.zeros(input_tensor.size())
        accumulated_gradients = crypten.cryptensor(accumulated_gradient_torch, src=1, requires_grad=True) 

        difference_vector = input_tensor - baseline
        comm.get().barrier()
        for m in range(1, self.steps + 1):
            alpha = m / self.steps
            interpolated_input = baseline + (difference_vector * alpha)
            
            gradient_at_point, _ = self._compute_input_gradient(args, interpolated_input, target_class)
            accumulated_gradients = accumulated_gradients + gradient_at_point
        avg_gradient = accumulated_gradients / self.steps
        
        integrated_gradients_raw = difference_vector * avg_gradient
        
        secure_saliency_map = self._aggregate_channels(integrated_gradients_raw)
        
        return secure_saliency_map, _
    
class SecureGradCAM(SecureAttributionBase):
    def __init__(self, model, target_layer_name: str):
        super().__init__(model)
        self.target_layer_name = target_layer_name

    def attribute(self, args, input_tensor: crypten.CrypTensor, target_class: int) -> crypten.CrypTensor:
        print("\n" + "="*80)
        print(f"{'SECURE GRAD-CAM DIAGNOSTIC SESSION':^80}")
        print("="*80)
        # 0. Reset Parameter Usage
        for p in self.model.parameters():
            p._used = False

        # 1. Extract Layers
        all_layers = []
        def extract_layers(m):
            for name, module in m.named_children():
                if len(list(module.children())) > 0: extract_layers(module)
                else:
                    if isinstance(module, crypten.nn.Module) and not isinstance(module, crypten.nn.Parameter):
                        all_layers.append((name, module))
        extract_layers(self.model)

        # 2. Parameter Registry Dump (FIXED FORMATTING)
        print("\n[STEP 2] Parameter Registry:")
        params_dict = dict(self.model.named_parameters())
        for p_name, p_val in params_dict.items():
            # Convert list to string before applying alignment format
            shape_str = str(list(p_val.size()))
            print(f"  - {p_name:<30} | Shape: {shape_str:<25} | Encrypted: {crypten.is_encrypted_tensor(p_val)}")

        # 3. Execution
        target_idx = next(i for i, (name, _) in enumerate(all_layers) if name == self.target_layer_name)
        
        with crypten.enable_grad():
            x = input_tensor if input_tensor.dim() == 4 else input_tensor.unsqueeze(0)
            
            # --- FORWARD PASSES ---
            parts = [("PART 1", all_layers[:target_idx + 1]), ("PART 2", all_layers[target_idx + 1:])]
            activations = None
            
            for part_name, layers in parts:
                print(f"\n>>> STARTING {part_name}")
                for name, layer in layers:
                    print(f"  [LAYER] {name} ({type(layer).__name__})")
                    x = self._safe_forward(x, layer, name, params_dict)
                    print(f"    -> Output Shape: {x.size()}")
                
                if part_name == "PART 1":
                    activations = x
                    activations.requires_grad = True
                    x_out = activations # Carry over to Part 2

            output = x_out

            # 4. Backward
            print(f"\n[STEP 4] Backward Pass (Target Class: {target_class})")
            grad_output_torch = torch.zeros(output.size())
            grad_output_torch[0, target_class] = 1.0
            grad_output = crypten.cryptensor(grad_output_torch, src=0) 
            output.backward(grad_output)
            
            gradients = activations.grad
            if gradients is None:
                print("  !! CRITICAL: Gradients are None. Chain is broken.")
                return crypten.zeros(input_tensor.size())

        # 5. Math
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True).relu()
        
        # 6. Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        final_cam = (cam - cam_min) / (cam_max - cam_min + 1e-7)
        


        target_size = (input_tensor.size(-2), input_tensor.size(-1))
        
        
        final_heatmap = crypten_interpolate_nearest(final_cam, target_size)

        # --- LOGGING ---
        print("-" * 30)
        print(f"SALIENCY MAP GENERATED")
        print(f"  After Interpolation: {list(final_heatmap.size())}")
        
        final_heatmap = final_heatmap.squeeze()
        print(f"  Final Squeezed Shape: {list(final_heatmap.size())}")
        print("-" * 30 + "\n")

        return final_heatmap

    def _safe_forward(self, x, layer, name, params_dict):
        l_type = str(type(layer)).lower()

        # 1. Activation/Spatial (No weights needed)
        if 'relu' in l_type:
            return crypten.nn.ReLU().encrypted(src=0).forward(x)
        
        if 'maxpool2d' in l_type:
            k = getattr(layer, 'kernel_size', 2)
            s = getattr(layer, 'stride', 2)
            p = getattr(layer, 'padding', 0)
            return crypten.nn.MaxPool2d(k, stride=s, padding=p).encrypted(src=0).forward(x)

        # 2. Weight-Bearing Detection
        is_conv = 'conv' in l_type
        is_gemm = 'gemm' in l_type or 'linear' in l_type
        is_bn = 'batchnormalization' in l_type

        if is_conv or is_gemm or is_bn:
            w, b, rm, rv = None, None, None, None
            
            # --- AGGRESSIVE MATCHING ---
            for p_name, p_val in params_dict.items():
                if getattr(p_val, '_used', False): continue
                
                # Check if the parameter name contains the layer index/name
                # (e.g., if layer name is '0', matches '0.weight')
                name_parts = name.split('.')
                if not any(part in p_name for part in name_parts): continue

                if is_conv and p_val.dim() == 4:
                    w = p_val
                elif is_gemm and p_val.dim() == 2:
                    w = p_val
                elif is_bn and p_val.dim() == 1:
                    w = p_val
                
                if w is not None:
                    w._used = True
                    # Look for bias in the immediate vicinity of this weight
                    base = p_name.rsplit('.', 1)[0]
                    b = params_dict.get(f"{base}.bias")
                    if b is not None: b._used = True
                    
                    if is_bn:
                        rm = params_dict.get(f"{base}.running_mean")
                        rv = params_dict.get(f"{base}.running_var")
                    break

            # 3. Forced Native Construction (Prevents the "Unpack" Error)
            if w is not None:
                if is_conv:
                    # Use weight size to define the layer
                    out_c, in_c, kh, kw = w.size()
                    native = crypten.nn.Conv2d(in_c, out_c, (kh, kw), 
                                              stride=getattr(layer, 'stride', 1), 
                                              padding=getattr(layer, 'padding', 0))
                    native.encrypt()
                    native.weight = w
                    if b is not None: native.bias = b
                    return native.forward(x)

                elif is_gemm:
                    if x.dim() > 2: x = x.reshape(x.size(0), -1)
                    native = crypten.nn.Linear(w.size(1), w.size(0))
                    native.encrypt()
                    native.weight = w
                    native.bias = b if b is not None else crypten.zeros(w.size(0))
                    return native.forward(x)

                elif is_bn:
                    native = crypten.nn.BatchNorm2d(w.size(0))
                    native.encrypt()
                    native.weight = w
                    native.bias = b if b is not None else crypten.zeros(w.size(0))
                    if rm is not None: native.running_mean = rm
                    if rv is not None: native.running_var = rv
                    native.training = False
                    return native.forward(x)

        # 4. Global Fallback
        try:
            # If it's a Flatten/View layer, this usually works
            return layer.forward(x)
        except Exception as e:
            # If we are here, a layer was missed. 
            # Check if x is still a CrypTensor or if it got 'de-encrypted'
            print(f"    [CRITICAL] {name} ({l_type}) failed: {e}")
            return x     
    # def _safe_forward(self, x, layer, name, params_dict):
    #     l_type = str(type(layer)).lower()

    #     if 'relu' in l_type:
    #         return crypten.nn.ReLU().encrypt(src=0).forward(x)
    #     if 'maxpool2d' in l_type:
    #         # Reconstruct MaxPool with native CrypTen to ensure spatial integrity
    #         # We use getattr with defaults to mirror the source layer exactly
    #         kernel  = getattr(layer, 'kernel_size', 3)
    #         stride  = getattr(layer, 'stride', 2)
    #         padding = getattr(layer, 'padding', 0)
    #         return crypten.nn.MaxPool2d(kernel, stride=stride, padding=padding).encrypt(src=0).forward(x)
            
    #     # --- 2. WEIGHT-BEARING LAYER LOGIC ---
    #     is_conv = 'conv' in l_type
    #     is_gemm = 'gemm' in l_type or 'linear' in l_type
    #     is_bn = 'batchnormalization' in l_type


    #     is_conv = 'conv' in l_type
    #     is_gemm = 'gemm' in l_type or 'linear' in l_type
    #     is_bn = 'batchnormalization' in l_type

    #     if is_conv or is_gemm or is_bn:
    #         w, b, rm, rv = None, None, None, None
            
    #         # --- IMPROVED MATCHING ---
    #         for p_name, p_val in params_dict.items():
    #             if getattr(p_val, '_used', False): continue
                
    #             # Filter by dimension
    #             if is_conv and p_val.dim() != 4: continue
    #             if is_gemm and p_val.dim() != 2: continue
    #             if is_bn and p_val.dim() != 1: continue

    #             # Numeric layer name should match numeric param if possible
    #             if name.isdigit() and not p_name[0].isdigit(): continue
    #             # Named layer (fc1) should match named param prefix
    #             if not name.isdigit() and not p_name.startswith(name.split('.')[0]): continue

    #             w = p_val
    #             w._used = True
                
    #             # --- BIAS SEARCH ---
    #             # 1. Try numeric increment (84 -> 85)
    #             if name.isdigit():
    #                 try:
    #                     num = int(''.join(filter(str.isdigit, p_name)))
    #                     b = params_dict.get(f"{num + 1}.data")
    #                 except: pass
    #             # 2. Try named replacement (fc1.weight -> fc1.bias)
    #             else:
    #                 base = p_name.replace('.weight.data', '').replace('.weight', '')
    #                 b = params_dict.get(f"{base}.bias.data") or params_dict.get(f"{base}.bias")
                
    #             if b is not None: 
    #                 b._used = True
    #                 print(f"    [MATCHED] {name} -> W: {p_name}, B: Found")
    #             else:
    #                 print(f"    [MATCHED] {name} -> W: {p_name}, B: None (Zero-padding)")

    #             # BN Mean/Var
    #             if is_bn:
    #                 rm = params_dict.get(f"{base}.running_mean.data") or params_dict.get(f"{base}.running_mean")
    #                 rv = params_dict.get(f"{base}.running_var.data") or params_dict.get(f"{base}.running_var")
    #             break

    #         if w is not None:
    #             if is_conv:
    #                 native = crypten.nn.Conv2d(w.size(1), w.size(0), (w.size(2), w.size(3)), 
    #                                           stride=getattr(layer, 'stride', 1), 
    #                                           padding=getattr(layer, 'padding', 0))
    #                 native.encrypt()
    #                 native.weight = w
    #                 # Fix: If bias is None, we must NOT assign it to keep native default or use zeros
    #                 if b is not None: native.bias = b 
    #                 return native.forward(x)

    #             elif is_gemm:
    #                 if x.dim() > 2: x = x.reshape(x.size(0), -1)
    #                 native = crypten.nn.Linear(w.size(1), w.size(0))
    #                 native.encrypt()
    #                 native.weight = w
    #                 # CRITICAL FIX for 'NoneType' size error:
    #                 if b is not None:
    #                     native.bias = b
    #                 else:
    #                     # Create an encrypted zero bias so .add() doesn't receive None
    #                     native.bias = crypten.zeros(w.size(0)) 
    #                 return native.forward(x)

    #             elif is_bn:
    #                 native = crypten.nn.BatchNorm2d(w.size(0))
    #                 native.encrypt()
    #                 native.weight = w
    #                 if b is not None: native.bias = b
    #                 else: native.bias = crypten.zeros(w.size(0))
    #                 if rm is not None: native.running_mean = rm
    #                 if rv is not None: native.running_var = rv
    #                 native.training = False
    #                 return native.forward(x)

    #     # Fallback for MaxPool, ReLU, etc.
    #     try:
    #         return layer.forward(x)
    #     except Exception as e:
    #         print(f"    [WARN] {name} fallback failed: {e}. Returning x.")
    #         return x
    
class SecureLRP(SecureAttributionBase):
    def __init__(self, model):
        super().__init__(model)
        self.activations = {}

    def _lrp_maxpool_rule(self, layer, relevance_input, activation_input):
        """
        Accurate MaxPool LRP: R_j = a_j * [ (R_k / Z_k) * grad(Z_k) ]
        """
        # Ensure inputs are 4D (1, C, H, W)
        val_relevance = relevance_input if relevance_input.dim() == 4 else relevance_input.unsqueeze(0)
        val_activation = activation_input.detach()
        val_activation.requires_grad = True

        with crypten.enable_grad():
            # 1. Local Forward
            z = layer.forward(val_activation) + 1e-9
            # 2. Secure Division
            s = val_relevance / z
            # 3. Backward to find the 'winner' indices
            z.backward(s)
            # 4. Resulting relevance
            r_j = (val_activation * val_activation.grad)
            
        return r_j
    
    def _lrp_epsilon_rule(self, args, module, relevance_input, activation_input):
        eps = float(args.epsilon)
        
        # Both relevance and activation should now be (1, C, H, W) or (1, D)
        val_relevance = relevance_input if relevance_input.dim() >= 2 else relevance_input.unsqueeze(0)
        val_activation = activation_input.detach()
        val_activation.requires_grad = True

        with crypten.enable_grad():
            # Force the forward call
            z = module.forward(val_activation) + eps
            s = val_relevance / z
            z.backward(s)
            
            # R_j = a_j * grad
            r_j = (val_activation * val_activation.grad)
        
        return r_j

    def attribute(self, args, input_tensor: crypten.CrypTensor, target_class: int) -> crypten.CrypTensor:
        # 1. Extract layers
        all_layers = []
        def extract_layers(m):
            for name, module in m.named_children():
                if len(list(module.children())) > 0:
                    extract_layers(module)
                else:
                    all_layers.append((name, module))
        extract_layers(self.model)

        # 2. Forward Pass: Capture activations
        self.activations = {}
        x = input_tensor
        
        for name, layer in all_layers:
            # Standardize dimensions for MNIST
            if x.dim() == 3: # (C, H, W) -> (1, C, H, W)
                x = x.unsqueeze(0)
            elif x.dim() == 1: # (Features,) -> (1, Features)
                x = x.unsqueeze(0)

            input_act = x 
            
            # Handle Flattening before Linear layers
            if isinstance(layer, crypten.nn.Linear) and x.dim() > 2:
                x = x.reshape(x.size(0), -1)
            
            # --- THE FIX: Use standard call but handle the internal dispatcher ---
            # Using layer(x) is safer than layer.forward(x) in most CrypTen versions
            # because it handles the weight/bias retrieval automatically.
            x = layer(x)
            
            # Store for LRP math
            self.activations[name + '_input'] = input_act.squeeze(0)
            self.activations[name + '_output'] = x.squeeze(0)

        # 3. Initialize Relevance
        # Use the actual logit value for the target class
        output = x
        R_mask = torch.zeros(output.size())
        R_mask[0, target_class] = 1.0
        
        # Start relevance: mask out all other classes
        current_relevance = (output * crypten.cryptensor(R_mask, src=1)).squeeze(0)

        # 4. Backward Loop
        reversed_layers = list(reversed(all_layers))
        for name, layer in reversed_layers:
            if name + '_input' not in self.activations:
                continue
                
            input_act = self.activations[name + '_input']
            output_act = self.activations[name + '_output']

            if current_relevance.shape != output_act.shape:
                current_relevance = current_relevance.reshape(output_act.shape)

            if isinstance(layer, (crypten.nn.Conv2d, crypten.nn.Linear)):
                current_relevance = self._lrp_epsilon_rule(args, layer, current_relevance, input_act)
            elif isinstance(layer, crypten.nn.MaxPool2d):
                current_relevance = self._lrp_maxpool_rule(layer, current_relevance, input_act)
            
        # 5. Final Aggregation
        if current_relevance.dim() == 1 or args.model == "AliceNet":
            # Assuming square input for MNIST (28x28)
            side = int(current_relevance.numel()**0.5)
            return current_relevance.view(side, side)
        else:
            return current_relevance.abs().sum(dim=0).squeeze()
    

