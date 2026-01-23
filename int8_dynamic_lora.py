import torch
import folder_paths
import comfy.utils
import comfy.lora
import logging
from torch import nn

class INT8DynamicLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders"

    def load_lora(self, model, lora_name, strength):
        if strength == 0:
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        model_patcher = model.clone()
        
        # 1. Get Patch Map
        key_map = {}
        if model_patcher.model.model_type.name != "ModelType.CLIP":
            key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, key_map)
        
        patch_dict = comfy.lora.load_lora(lora, key_map, log_missing=True)
        
        # 2. Register Global Hook (if not exists)
        from .int8_quant import DynamicLoRAHook
        DynamicLoRAHook.register(model_patcher.model.diffusion_model)

        # 3. Add to Dynamic LoRA list in transformer_options
        # This ensures ComfyUI's cloning handles everything and it's non-sticky
        if "transformer_options" not in model_patcher.model_options:
            model_patcher.model_options["transformer_options"] = {}
        
        opts = model_patcher.model_options["transformer_options"]
        if "dynamic_loras" not in opts:
            opts["dynamic_loras"] = []
        else:
            # Shallow copy the list to avoid modifying the parent patcher's list
            opts["dynamic_loras"] = opts["dynamic_loras"].copy()
            
        opts["dynamic_loras"].append({
            "name": lora_name,
            "strength": strength,
            "patches": patch_dict
        })

        return (model_patcher,)

class INT8DynamicLoraStack:
    """
    Apply multiple LoRAs in one node for efficiency.
    """
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {"model": ("MODEL",)},
            "optional": {},
        }
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        for i in range(1, 11):
            inputs["optional"][f"lora_{i}"] = (lora_list,)
            inputs["optional"][f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01})
        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_stack"
    CATEGORY = "loaders"

    def apply_stack(self, model, **kwargs):
        loader = INT8DynamicLoraLoader()
        current_model = model
        for i in range(1, 11):
            lora_name = kwargs.get(f"lora_{i}")
            strength = kwargs.get(f"strength_{i}", 0)
            if lora_name and lora_name != "None" and strength != 0:
                # We can optimize this by NOT cloning and re-hooking 10 times,
                # but for simplicity/reliability, we'll use the loader.
                (current_model,) = loader.load_lora(current_model, lora_name, strength)
        return (current_model,)

NODE_CLASS_MAPPINGS = {
    "INT8DynamicLoraLoader": INT8DynamicLoraLoader,
    "INT8DynamicLoraStack": INT8DynamicLoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INT8DynamicLoraLoader": "Load LoRA INT8 (Dynamic)",
    "INT8DynamicLoraStack": "INT8 LoRA Stack (Dynamic)",
}
