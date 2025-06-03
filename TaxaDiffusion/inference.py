import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess
import numpy as np


from pathlib import Path
from tqdm.auto import tqdm
# from einops import rearrange
from omegaconf import OmegaConf
from peft import LoraConfig
# from safetensors import safe_open


import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as T


from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from taxa_diffusion.utils import instantiate_from_config
from taxa_diffusion.utils import load_checkpoint
from taxa_diffusion.diff_pipeline.pipeline_stable_diffusion_taxonomy import StableDiffusionTaxonomyPipeline
from taxa_diffusion.datasets.toxonomy import get_keys_at_level, get_lineage_bottom_to_top, fish_taxonomy, taxonomy, bio_scan_taxonomy



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            rank = int(os.environ['RANK'])
            local_rank = rank % num_gpus
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend=backend, **kwargs)
        else:
            rank = int(os.environ['RANK'])
            dist.init_process_group(backend='gloo', **kwargs)
            return 0

    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank


def main(
    name: str,
    use_wandb: bool,
    launcher: str,
    config: dict
    ):
    
    is_debug = config.train.is_debug
    
    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher, port=29503)
    global_rank     = dist.get_rank()
    is_main_process = global_rank == 0
    device = torch.device('cuda', local_rank)

    seed = config.train.global_seed + global_rank
    set_seed(seed)
    
    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(config.train.output_dir, folder_name)
    if is_debug and os.path.exists(output_dir) and is_main_process:
        os.system(f"rm -rf {output_dir}")
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="conffusion", name=folder_name, config=config)
        
    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
           
    vae                        = AutoencoderKL.from_pretrained(config.train.pretrained_model_path, subfolder="vae")
    tokenizer                  = CLIPTokenizer.from_pretrained(config.train.pretrained_model_path, subfolder="tokenizer")
    text_encoder               = CLIPTextModel.from_pretrained(config.train.pretrained_model_path, subfolder="text_encoder")
    unet                       = UNet2DConditionModel.from_pretrained(config.train.pretrained_model_path, subfolder="unet")
    taxonomy_condition_adapter = instantiate_from_config(config.model)

    unet_lora_config = LoraConfig(
        r=config.train.lora_rank,
        lora_alpha=config.train.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    if config.train.add_lora_unet:
        unet.add_adapter(unet_lora_config)
    
    # Get the validation dataset
    level_name = config.dataset.test.level_name
    sample_number = config.dataset.test.sample_number

    if config.train.level_number == 7:
        unique_names_level = get_keys_at_level(taxonomy=taxonomy, level=level_name)
    else:
        if 'name' in config.dataset.test and config.dataset.test.name == 'bio_scan':
            unique_names_level = get_keys_at_level(taxonomy=bio_scan_taxonomy, level=level_name)
        else:
            unique_names_level = get_keys_at_level(taxonomy=fish_taxonomy, level=level_name)
            
    # Move models to GPU
    vae.to(local_rank)
    unet.to(local_rank)
    text_encoder.to(local_rank)
    taxonomy_condition_adapter.to(local_rank)

    logging.info(f"******* pretrained_model_path_last_part from {config.train.pretrained_model_path_last_part} will load!!!")
    # Load pretrained unet weights
    taxonomy_condition_adapter, _, _, _, _, _ = load_checkpoint(
        model=taxonomy_condition_adapter,
        unet=None,
        optimizer=None,
        scheduler=None,
        checkpoint_path=config.train.pretrained_model_path_last_part,
        logging=logging,
        pretrained_lora_model=config.train.pretrained_lora_model,
        is_main_process=is_main_process)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    taxonomy_condition_adapter.requires_grad_(False)

    # Enable xformers
    if config.train.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    vae.eval()
    unet.eval()
    text_encoder.eval()
    taxonomy_condition_adapter.eval()

    logging.info("Validation is started")
    generator = torch.Generator(device=device)
    resolution = config.dataset.test.resolution
    height = resolution[0] if not isinstance(resolution, int) else resolution
    width  = resolution[1] if not isinstance(resolution, int) else resolution
    
    # Validation pipeline
    validation_pipeline = StableDiffusionTaxonomyPipeline.from_pretrained(
        config.train.pretrained_model_path
    ).to(device)

    validation_pipeline.enable_vae_slicing()
    validation_pipeline.taxonomy_condition_adapter = taxonomy_condition_adapter
    validation_pipeline.load_lora_weights(config.train.pretrained_lora_model)
    validation_pipeline.safety_checker = None
    logging.info(f"unique_names_level is \n{unique_names_level}")
    
    for target_name in unique_names_level:
        if target_name == None or target_name == 'any':
            continue

        logging.info(f"target_name is {target_name} and will generate")

        if config.train.level_number == 7:
            lineage = get_lineage_bottom_to_top(taxonomy, target_name, level_name, logging)
        else:
            if 'name' in config.dataset.test and config.dataset.test.name == 'bio_scan':
                lineage = get_lineage_bottom_to_top(bio_scan_taxonomy, target_name, level_name, logging)
            else:
                lineage = get_lineage_bottom_to_top(fish_taxonomy, target_name, level_name, logging)

        if config.train.level_number == 7:
            conditions_list_name = [
                f"kingdom: {lineage['kingdom'] if 'kingdom' in lineage else 'None'}",
                f"phylum: {lineage['phylum'] if 'phylum' in lineage else 'None'}",
                f"class: {lineage['class'] if 'class' in lineage else 'None'}",
                f"order: {lineage['order'] if 'order' in lineage else 'None'}",
                f"family: {lineage['family'] if 'family' in lineage else 'None'}",
                f"genus: {lineage['genus'] if 'genus' in lineage else 'None'}",
                f"specific_epithet: {lineage['species'] if 'species' in lineage else 'None'}",
            ]
        else:
            conditions_list_name = [
                f"class: {lineage['class'] if 'class' in lineage else 'None'}",
                f"order: {lineage['order'] if 'order' in lineage else 'None'}",
                f"family: {lineage['family'] if 'family' in lineage else 'None'}",
                f"genus: {lineage['genus'] if 'genus' in lineage else 'None'}",
                f"specific_epithet: {lineage['species'] if 'species' in lineage else 'None'}",
            ]
        
        logging.info(f"******* conditions_list_name from {conditions_list_name} !!!")

        dim = text_encoder.config.hidden_size  # Get the dimensionality from the text encoder
        encoder_hidden_states = torch.zeros((1, config.train.level_number, tokenizer.model_max_length, dim)).to(device)

        with torch.no_grad():
            for i in range(config.train.level_number):  # Loop over each level
                prompt_ids = tokenizer(
                    conditions_list_name[i], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(device)

                # Compute the hidden states
                hidden_state = text_encoder(prompt_ids)[0]  # Shape: [batch_size, tokenizer.model_max_length, dim]

                # Store the hidden states in the correct index of encoder_hidden_states
                encoder_hidden_states[:, i, :, :] = hidden_state

        if is_main_process:
            logging.info(f"encoder_hidden shape is {encoder_hidden_states.shape}")
                
        for i in range(sample_number):
            combined_images = []
            generator.manual_seed(config.train.global_seed * i + random.randint(1, 10000))
            sample = validation_pipeline(
                taxonomy_conditions  = encoder_hidden_states,
                prompt               = [""],
                range_guidance       = config.train.range_guidance,
                taxonomy_cut_off     = config.train.taxonomy_cut_off,
                do_new_guidance      = config.train.do_new_guidance,
                generator            = generator,
                height               = height,
                width                = width,
                num_inference_steps  = config.dataset.validation.num_inference_steps,
                guidance_scale       = config.dataset.validation.guidance_scale,
            ).images[0]

            sample = torchvision.transforms.functional.to_tensor(sample)
            combined_images.append(sample.cpu())

            # Stack and save the combined images
            combined_images = torch.stack(combined_images)
            directory = f"{output_dir}/samples/sample_target_name_{target_name.replace('/', ' ')}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            save_path = directory + f"/prompt_{i}.png"
            torchvision.utils.save_image(combined_images, save_path, nrow=len(combined_images))
            logging.info(f"Saved samples to {save_path}")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/multigen20.yaml')
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, config=config)