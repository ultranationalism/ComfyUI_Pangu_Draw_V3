import comfy
import comfy.supported_models
import comfy.supported_models_base
import torch
import os

from .pangu import PanGu_SDXL,PanGu_ModelPatcher
from comfy import model_management,model_detection
from comfy.sd import VAE,load_model_weights,CLIP

def PanGu_load_checkpoint(ckpt_path, output_vae=True, output_clip=True, embedding_directory=None, output_model=True):
    sd = comfy.utils.load_torch_file(ckpt_path)
    sd_keys = sd.keys()
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    clip_target = None

    parameters = comfy.utils.calculate_parameters(sd, "model.diffusion_model.")
    unet_dtype = torch.float16
    load_device = model_management.get_torch_device()


    #model_config = model_detection.model_config_from_unet(sd, "model.diffusion_model.", unet_dtype)
    model_config = PanGu_SDXL()
    model_config.set_inference_dtype

    if model_config is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(ckpt_path))


    if output_model:
        inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        offload_device = model_management.unet_offload_device()
        model = model_config.get_model(sd, "model.diffusion_model.", device=inital_load_device)
        model.load_model_weights(sd, "model.diffusion_model.")

    if output_vae:
        vae_sd = comfy.utils.state_dict_prefix_replace(sd, {"first_stage_model.": ""}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd)

    if output_clip:
        clip_target = model_config.clip_target()
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                clip = CLIP(clip_target, embedding_directory=embedding_directory)
                m, u = clip.load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    print("clip missing:", m)

                if len(u) > 0:
                    print("clip unexpected:", u)
            else:
                print("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")


    left_over = sd.keys()
    if len(left_over) > 0:
        print("left over keys:", left_over)

    if output_model:
        model_patcher = PanGu_ModelPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device(), current_device=inital_load_device)
        if inital_load_device != torch.device("cpu"):
            print("loaded straight to GPU")
            model_management.load_model_gpu(model_patcher)

    return (model_patcher, clip, vae, clipvision)
