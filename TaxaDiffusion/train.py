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
import safetensors


from pathlib import Path
from tqdm.auto import tqdm
# from einops import rearrange
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
# from safetensors import safe_open


import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as T


from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import convert_state_dict_to_diffusers

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel


from taxa_diffusion.utils import instantiate_from_config, load_checkpoint, save_checkpoint
from taxa_diffusion.models.lora_adapter import inject_lora_into_attention
from taxa_diffusion.diff_pipeline.pipeline_stable_diffusion_taxonomy import StableDiffusionTaxonomyPipeline



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    local_rank      = init_dist(launcher=launcher)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
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
        run = wandb.init(project="taxonomy", name=folder_name, config=config)
        
    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
        
        
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(config.train.noise_scheduler_kwargs))
    
    vae                        = AutoencoderKL.from_pretrained(config.train.pretrained_model_path, subfolder="vae")
    tokenizer                  = CLIPTokenizer.from_pretrained(config.train.pretrained_model_path, subfolder="tokenizer")
    text_encoder               = CLIPTextModel.from_pretrained(config.train.pretrained_model_path, subfolder="text_encoder")
    unet                       = UNet2DConditionModel.from_pretrained(config.train.pretrained_model_path, subfolder="unet")
    taxonomy_condition_adapter = instantiate_from_config(config.model)
    
    # Get the training dataset
    train_dataset = instantiate_from_config(config.dataset.train)
    valid_dataset = instantiate_from_config(config.dataset.validation)

    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=local_rank,
        shuffle=True,
        seed=config.train.global_seed,
    )
    train_dataloader = DataLoader(train_dataset,
                                  sampler=distributed_sampler,
                                  num_workers=config.train.num_workers,
                                  batch_size=config.train.train_batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=True)
    
    valid_dataloader = DataLoader(valid_dataset,
                                  num_workers=config.train.num_workers,
                                  batch_size=config.train.valid_batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=False)
    
    max_train_steps = config.train.max_train_steps
    max_train_epoch = config.train.max_train_epoch
    start_epoch = 0
    
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    checkpointing_steps = config.train.checkpointing_steps
    checkpointing_epochs = config.train.checkpointing_epochs
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)
    
    # Move models to GPU
    vae.to(local_rank)
    unet.to(local_rank)
    text_encoder.to(local_rank)
    taxonomy_condition_adapter.to(local_rank)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    taxonomy_condition_adapter.requires_grad_(False)

    for param in taxonomy_condition_adapter.encoder_layers[config.train.taxonomy_cut_off].parameters():
        param.requires_grad = True

    trainable_params = list(filter(lambda p: p.requires_grad, taxonomy_condition_adapter.parameters()))

    if is_main_process:
        logging.info(f"trainable params number (No Unet): {len(trainable_params)}")
        logging.info(f"trainable params scale (No Unet): {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    unet_lora_config = LoraConfig(
        r=config.train.lora_rank,
        lora_alpha=config.train.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # Add adapter and make sure the trainable params are in float32.
    if config.train.add_lora_unet:
        unet.add_adapter(unet_lora_config)
        if config.train.taxonomy_cut_off < 1:
            lora_layers = filter(lambda p: p.requires_grad, unet.parameters())  
            trainable_params.extend(lora_layers)
        else: 
            for param in unet.parameters():
                param.requires_grad_(False)

    learning_rate = config.optimize.learning_rate
    if config.train.scale_lr:
        learning_rate = (learning_rate * config.optimize.radient_accumulation_steps * config.train.train_batch_size * num_processes)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(config.optimize.adam_beta1, config.optimize.adam_beta2),
        weight_decay=config.optimize.adam_weight_decay,
        eps=config.optimize.adam_epsilon,
    )
    
    lr_scheduler = get_scheduler(
        config.optimize.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.optimize.lr_warmup_steps * config.optimize.gradient_accumulation_steps,
        num_training_steps=max_train_steps * config.optimize.gradient_accumulation_steps,
    )
    
    if is_main_process:
        logging.info(f"trainable params number: {len(trainable_params)}")
        logging.info(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")
    
    if config.train.pretrained_model_path_last_part != '' and config.train.pretrained_lora_model != '':
        taxonomy_condition_adapter, unet, _, _, _, _ = load_checkpoint(
            model=taxonomy_condition_adapter,
            unet=unet,
            optimizer=None,
            scheduler=None,
            checkpoint_path=config.train.pretrained_model_path_last_part,
            logging=logging,
            pretrained_lora_model=config.train.pretrained_lora_model,
            is_main_process=is_main_process)
        if is_main_process:
            logging.info(f"******* pretrained_model_path_last_part from {config.train.pretrained_model_path_last_part} is loaded not checkpoint!!!")

    elif config.train.checkpoint_path != '' and config.train.pretrained_lora_model != '':
        taxonomy_condition_adapter, unet, optimizer, lr_scheduler, start_epoch, _ = load_checkpoint(
            model=taxonomy_condition_adapter,
            unet=unet,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            checkpoint_path=config.train.checkpoint_path,
            logging=logging,
            pretrained_lora_model=config.train.pretrained_lora_model,
            is_main_process=is_main_process)
        if is_main_process:
            logging.info(f"&&&&&&&& checkpoint_path from {config.train.checkpoint_path} is loaded not pretrained model!!!")

    # Enable xformers
    if config.train.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if config.train.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.optimize.gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = config.train.train_batch_size * num_processes * config.optimize.gradient_accumulation_steps
    
    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {config.train.train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {config.optimize.gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = start_epoch * len(train_dataloader)
    first_epoch = start_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")
    
    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if config.train.mixed_precision_training else None
    
    taxonomy_condition_adapter = DDP(taxonomy_condition_adapter, device_ids=[local_rank], output_device=local_rank)
    if config.train.add_lora_unet and config.train.taxonomy_cut_off < 1:
        unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    for epoch in range(first_epoch, num_train_epochs):
        if config.train.add_lora_unet and config.train.taxonomy_cut_off < 1:
            unet.train()
        else:
            unet.eval()
        vae.eval()
        taxonomy_condition_adapter.train()
        text_encoder.eval()

        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
            
        epoch_loss = 0.0 
        num_batches = len(train_dataloader)
        
        for step, batch in enumerate(train_dataloader):   
            # Data batch sanity check
            if epoch == first_epoch and step == 0 and is_main_process:
                target_imgs, texts = batch['target_image'].cpu(), batch['name']
                for idx, (target_img, text) in enumerate(zip(target_imgs, texts)):
                    target_img = target_img / 2. + 0.5
                    torchvision.utils.save_image(target_img, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{local_rank}-{idx}'}.jpg")
            
            # Convert videos to latent space            
            target_image = batch["target_image"].to(local_rank)
            with torch.no_grad():
                latents = vae.encode(target_image).latent_dist
                latents = latents.sample()
                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            batch_size = config.train.train_batch_size 
            dim = text_encoder.config.hidden_size  # Get the dimensionality from the text encoder
            encoder_hidden_states = torch.zeros((batch_size, config.train.level_number, tokenizer.model_max_length, dim)).to(latents.device)

            # Get the text embedding for conditioning
            with torch.no_grad():
                for i in range(config.train.level_number):  # Loop over each level
                    prompt_ids = tokenizer(
                        batch['conditions_list_name'][i], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(latents.device)

                    # Compute the hidden states
                    hidden_state = text_encoder(prompt_ids)[0]  # Shape: [batch_size, tokenizer.model_max_length, dim]

                    # Store the hidden states in the correct index of encoder_hidden_states
                    encoder_hidden_states[:, i, :, :] = hidden_state
                            
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=config.train.mixed_precision_training):
                taxonomy_embeds = taxonomy_condition_adapter(encoder_hidden_states, cut_off=config.train.taxonomy_cut_off)
                model_pred = unet(sample=noisy_latents, 
                                  timestep=timesteps, 
                                  encoder_hidden_states=taxonomy_embeds).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
            # Backpropagate
            if config.train.mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, config.train.max_grad_norm)
                # torch.nn.utils.clip_grad_norm_(taxonomy_condition_adapter.parameters(), config.train.max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(trainable_params, config.train.max_grad_norm)
                # torch.nn.utils.clip_grad_norm_(taxonomy_condition_adapter.parameters(), config.train.max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad()
            if is_main_process:
                progress_bar.update(1)
            global_step += 1
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
            
            epoch_loss += loss.item()

            # logging.info GPU memory usage
            if step % 1000 == 0 and is_main_process:  # Adjust the frequency as needed
                logging.info(f"Epoch: {epoch}, Step: {step}, Allocated GPU memory: {torch.cuda.memory_allocated(local_rank)/1024**2:.2f} MB, Reserved GPU memory: {torch.cuda.memory_reserved(local_rank)/1024**2:.2f} MB")
            
            lora_checkpoint_path = ""
            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == num_train_epochs * len(train_dataloader) - 1 or global_step % config.train.validation_steps == 0 or global_step in config.train.validation_steps_tuple):
                lora_checkpoint_path = save_checkpoint(
                    model=taxonomy_condition_adapter,
                    unet=unet if config.train.add_lora_unet else None, 
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    output_dir=output_dir,
                    epoch=epoch if not (global_step % config.train.validation_steps == 0 or global_step in config.train.validation_steps_tuple) else -1,
                    global_step=global_step,
                    step=step,
                    train_dataloader=train_dataloader,
                    logging=logging,
                    config=config
                )

            # Periodically validation
            if is_main_process and (global_step % config.train.validation_steps == 0 or global_step in config.train.validation_steps_tuple):
                logging.info("Validation is started")
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(config.train.global_seed)
                resolution = config.dataset.validation.params.img_size
                height = resolution[0] if not isinstance(resolution, int) else resolution
                width  = resolution[1] if not isinstance(resolution, int) else resolution
                
                # Validation pipeline
                validation_pipeline = StableDiffusionTaxonomyPipeline.from_pretrained(
                    config.train.pretrained_model_path,
                ).to(device)
                if config.train.add_lora_unet:
                    if config.train.taxonomy_cut_off == 0:
                        validation_pipeline.load_lora_weights(lora_checkpoint_path)
                    else: 
                        validation_pipeline.load_lora_weights(config.train.pretrained_lora_model)
                validation_pipeline.enable_vae_slicing()
                # validation_pipeline.taxonomy_condition_adapter = taxonomy_condition_adapter
                validation_pipeline.taxonomy_condition_adapter = taxonomy_condition_adapter.module
                validation_pipeline.safety_checker = None

                for step_val, batch_val in enumerate(valid_dataloader):
                    batch_size = config.train.valid_batch_size 
                    dim = text_encoder.config.hidden_size  # Get the dimensionality from the text encoder
                    encoder_hidden_states = torch.zeros((batch_size, config.train.level_number, tokenizer.model_max_length, dim)).to(latents.device)

                    # Get the text embedding for conditioning
                    with torch.no_grad():
                        for i in range(config.train.level_number):  # Loop over each level
                            prompt_ids = tokenizer(
                                batch_val['conditions_list_name'][i], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                            ).input_ids.to(latents.device)

                            # Compute the hidden states
                            hidden_state = text_encoder(prompt_ids)[0]  # Shape: [batch_size, tokenizer.model_max_length, dim]
                            encoder_hidden_states[:, i, :, :] = hidden_state
                        
                    names = batch_val['name']
                    target_images = batch_val['target_image'].cpu()
                    for idx, encoder_hidden in enumerate(encoder_hidden_states):
                        encoder_hidden = encoder_hidden.unsqueeze(0)
                        if is_main_process:
                            logging.info(f"encoder_hidden shape is {encoder_hidden.shape}")
                        name = names[idx]
                        sample = validation_pipeline(
                            taxonomy_conditions  = encoder_hidden,
                            prompt               = [""],
                            taxonomy_cut_off     = config.train.taxonomy_cut_off,
                            do_new_guidance      = config.train.do_new_guidance,
                            generator            = generator,
                            height               = height,
                            width                = width,
                            num_inference_steps  = config.dataset.validation.num_inference_steps,
                            guidance_scale       = config.dataset.validation.guidance_scale,
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        combined_images = [sample.cpu()]
                        combined_images.append(target_images[idx] / 2. + 0.5)
                        combined_images = torch.stack(combined_images)

                        directory = f"{output_dir}/samples/sample-{global_step}"
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        save_path = directory + f"/prompt_{'-'.join(name.replace('/', '').split()[:10]) if not name == '' else f'{local_rank}-{step_val}'}.png"
                        torchvision.utils.save_image(combined_images, save_path, nrow=len(combined_images))
                        logging.info(f"Saved samples to {save_path}")
                
                del validation_pipeline
                torch.cuda.empty_cache()
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if is_main_process:
                progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
        
        if is_main_process:
            epoch_loss /= num_batches
            if (not is_debug) and use_wandb:
                wandb.log({"epoch_loss": epoch_loss}, step=epoch)
            logging.info(f"Epoch {epoch} loss: {epoch_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/multigen20.yaml')
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, config=config)