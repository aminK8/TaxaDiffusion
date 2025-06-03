import torch.distributed as dist

import torch
import importlib
import numpy as np
import safetensors
import shutil
import os
from peft.utils import get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers
from taxa_diffusion.diff_pipeline.pipeline_stable_diffusion_taxonomy import StableDiffusionTaxonomyPipeline



def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)

    

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def print_model_keys(model, logging):
    model_keys = list(model.state_dict().keys())
    logging.info(f"Keys in model: {model_keys}")
    return model_keys


def load_checkpoint(model, unet, optimizer, scheduler, checkpoint_path,
                    logging, is_main_process=True, pretrained_lora_model='', map_location="cpu"):
    """
    Load the model and unet checkpoint from the specified path.
    
    Args:
        model: The main model to load the state into.
        unet: The unet model to load the state into (optional).
        optimizer: Optimizer to load state into (optional).
        scheduler: Learning rate scheduler to load state into (optional).
        checkpoint_path: Path to the checkpoint file.
        logging: Logging module for logging information.
        is_main_process: Boolean indicating if this is the main process for logging.
        map_location: Map location for loading the checkpoint.
        
    Returns:
        model, unet, optimizer, scheduler, epoch, global_step
    """
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        logging.info(f"Checkpoint not found: {checkpoint_path}")
        return model, unet, optimizer, scheduler, 0, 0

    # Load checkpoint state
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    model_missing_keys = []
    model_unexpected_keys = []

    # Adjust model state dict and load
    if model is not None:
        adjusted_model_state_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
        model_missing_keys, model_unexpected_keys = model.load_state_dict(adjusted_model_state_dict, strict=False)

    unet_missing_keys = []
    unet_unexpected_keys = []
    # Adjust unet state dict and load if unet is provided
    if unet is not None:
        if pretrained_lora_model != '':
            lora_state_dict = safetensors.torch.load_file(pretrained_lora_model)
            lora_state_dict = {key.replace("unet.", ""): value for key, value in lora_state_dict.items()}
            lora_state_dict = {key.replace("lora.down", "lora_A.default"): value for key, value in lora_state_dict.items()}
            lora_state_dict = {key.replace("lora.up", "lora_B.default"): value for key, value in lora_state_dict.items()}

            unet_missing_keys, unet_unexpected_keys = unet.load_state_dict(lora_state_dict, strict=False)
            if is_main_process:
                logging.info(f"LoRA loaded from {pretrained_lora_model} - UNet missing keys: {len(unet_missing_keys)}, unexpected keys: {len(unet_unexpected_keys)}")
                # logging.info(f"LoRA loaded from {pretrained_lora_model} - UNet missing keys: {unet_missing_keys}, \n \n unexpected keys: {unet_unexpected_keys}")
        else:
            adjusted_model_state_dict = {key.replace("module.", ""): value for key, value in checkpoint['unet'].items()}
            unet_missing_keys, unet_unexpected_keys = unet.load_state_dict(adjusted_model_state_dict, strict=False)
            if is_main_process:
                logging.info(f"UNet state dict is loaded in checkpoint.")

    else:
        unet_missing_keys, unet_unexpected_keys = [], []
        if is_main_process and unet is not None:
            logging.info(f"UNet state dict not found in checkpoint, or UNet is not provided for loading.")

    # Log information about missing and unexpected keys
    if is_main_process:
        logging.info(f"Checkpoint loaded from {checkpoint_path} - Model missing keys: {len(model_missing_keys)}, unexpected keys: {len(model_unexpected_keys)}")

    # Load optimizer and scheduler states if available
    if 'optimizer' in checkpoint and optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        if is_main_process:
            logging.info(f"Optimizer state loaded from {checkpoint_path}")
    if 'scheduler' in checkpoint and scheduler:
        scheduler.load_state_dict(checkpoint['scheduler'])
        if is_main_process:
            logging.info(f"Scheduler state loaded from {checkpoint_path}")

    # Verify that the loading process was successful
    if (len(model_unexpected_keys) != 0 or len(unet_unexpected_keys)) != 0 and is_main_process:
        logging.warning(f"Unexpected keys were found in the model or unet state dict.")

    # Log epoch and global step
    if is_main_process:
        logging.info(f"Finally, checkpoint restored from {checkpoint_path}, epoch: {epoch}, global step: {global_step}")

    return model, unet, optimizer, scheduler, epoch + 1, global_step

    
def save_checkpoint(model, unet, optimizer, scheduler, output_dir, epoch,
                    global_step, step, train_dataloader, logging, config, 
                    max_checkpoints=5, force_lora_save=False):
    """
    Save the model and unet checkpoint to the specified output directory.
    
    Args:
        model: The main model to save.
        unet: The unet model to save (optional).
        optimizer: Optimizer used for training.
        scheduler: Learning rate scheduler.
        output_dir: Directory to save the checkpoint.
        epoch: Current epoch number.
        global_step: Global training step.
        step: Step in the current epoch.
        train_dataloader: Training dataloader for step information.
        logging: Logging module for logging information.
        max_checkpoints: Maximum number of checkpoints to retain (default is 5).
    """
    # Create checkpoint directory if it doesn't exist
    save_path = os.path.join(output_dir, "checkpoints")
    os.makedirs(save_path, exist_ok=True)

    # Prepare state dictionary for saving
    state_dict = {
        "epoch": epoch,
        "global_step": global_step,
        # "state_dict": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        "state_dict": model.module.state_dict() if model is not None and hasattr(model, 'module') else (model.state_dict() if model is not None else None),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    # Include unet state dict if provided
    # if unet is not None:
    #     state_dict["unet_state_dict"] = unet.module.state_dict() if hasattr(unet, 'module') else unet.state_dict()

    # Define checkpoint filename based on the training step or epoch
    checkpoint_filename = f"checkpoint-epoch-{epoch+1}.ckpt" if step == len(train_dataloader) - 1 else "checkpoint.ckpt"
    checkpoint_path = os.path.join(save_path, checkpoint_filename)

    # Save checkpoint using torch.save for maximum compatibility
    torch.save(state_dict, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path} (global_step: {global_step})")

    checkpoint_filename = f"lora_checkpoint-epoch-{epoch+1}" if step == len(train_dataloader) - 1 else "lora_checkpoint"
    lora_checkpoint_path = os.path.join(save_path, checkpoint_filename)

    if (config.train.add_lora_unet and config.train.taxonomy_cut_off < 1) or force_lora_save: 
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet.module if hasattr(unet, 'module') else unet))
        StableDiffusionTaxonomyPipeline.save_lora_weights(
            save_directory=lora_checkpoint_path,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )
        logging.info(f"LoRA is saved state to {lora_checkpoint_path}")

    # Manage the number of checkpoints to retain
    checkpoint_files = sorted([f for f in os.listdir(save_path) if f.startswith("checkpoint-epoch-")],
                              key=lambda x: os.path.getmtime(os.path.join(save_path, x)))
    if len(checkpoint_files) > max_checkpoints:
        oldest_checkpoint = checkpoint_files[0]
        os.remove(os.path.join(save_path, oldest_checkpoint))
        logging.info(f"Removed old checkpoint: {oldest_checkpoint}")

    checkpoint_files = sorted([f for f in os.listdir(save_path) if f.startswith("lora_checkpoint-epoch-")],
                              key=lambda x: os.path.getmtime(os.path.join(save_path, x)))
    if len(checkpoint_files) > max_checkpoints:
        oldest_checkpoint = checkpoint_files[0]
        oldest_checkpoint_path = os.path.join(save_path, oldest_checkpoint)
        if os.path.isdir(oldest_checkpoint_path):
            shutil.rmtree(oldest_checkpoint_path)  # Use rmtree for directories
            logging.info(f"Removed old checkpoint directory: {oldest_checkpoint}")
        else:
            os.remove(oldest_checkpoint_path)  # Use remove for files
            logging.info(f"Removed old checkpoint file: {oldest_checkpoint}")

    return lora_checkpoint_path
