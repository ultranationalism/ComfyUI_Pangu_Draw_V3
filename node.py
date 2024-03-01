import sys
import os

file_path = os.path.abspath(__file__) 
dir_path = os.path.dirname(file_path) 
sys.path.append(dir_path) 

import folder_paths
import time
from omegaconf import OmegaConf
from .sgm.helpers import SD_XL_BASE_RATIOS, VERSION2SPECS,load_model_from_config,init_sampling
from .sgm.util import seed_everything
from .comfysp.loader import PanGu_load_checkpoint

BASE_SIZE_LIST = [
    (256, 1024),
    (256, 960),
    (320, 768),
    (384, 640),
    (448, 576),
    (512, 512),
    (576, 448),
    (640, 384),
    (768, 320),
    (960, 256),
    (1024, 256),
]
HIGH_SOLUTION_BASE_SIZE_LIST = [
    (512, 2048),
    (512, 1920),
    (768, 1536),
    (864, 1536),
    (960, 1280),
    (1024, 1024),
    (1280, 960),
    (1536, 768),
    (1920, 512),
    (2048, 512),
]

class PanGuCheckpointLoader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"low": (folder_paths.get_filename_list("checkpoints"),),
				"high": (folder_paths.get_filename_list("checkpoints"),),
			}
		}
	RETURN_TYPES = ("MODEL","MODEL",)
	RETURN_NAMES = ("low","high",)
	FUNCTION = "load_checkpoint"
	CATEGORY = "PanGu_draw_v3"
	TITLE = "PanGu_draw_v3 Checkpoint Loader"

	def load_checkpoint(self, low, high):
		low = folder_paths.get_full_path("checkpoints", low)
		high = folder_paths.get_full_path("checkpoints", high)
		comfy_path = os.path.dirname(folder_paths.__file__)
		config_path = os.path.join(comfy_path, 'custom_nodes/ComfyUI_Pangu_Draw_V3/config/inference/pangu_sd_xl_base.yaml')
		config = OmegaConf.load(config_path)
		version = config.pop("version", "PanGu-SDXL-base-1.0")
		model_low = load_model_from_config(
			ckpt = low,
			model_config = config.model,
		)
		config.model.params.conditioner_config = "__is_unconditional__"
		config.model.params.first_stage_config = "__is_unconditional__"

		model_high = load_model_from_config(
			ckpt = high,
			model_config = config.model,
		)
		model_high.first_stage_model = None
		model_high.conditioner = None
		model_low.first_stage_model.encoder = None

		return (model_low,model_high)

class PanGu_txt2img:
	@classmethod
	def INPUT_TYPES(s):
		return {"required":
					{"low": ("MODEL",),
					"high": ("MODEL",),
					"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
					"steps": ("INT", {"default": 40, "min": 2, "max": 10000}),
					"sd_xl_base_ratios":("STRING",{"default":1.0}),
					#"cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
					#"scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
					"prompt": ("STRING", {
                    "multiline": True, #
                    "default": "1girl"
                }),
					"negative_prompt": ("STRING", {
                    "multiline": True, 
                }),
					"orig_width":("INT",{"default":0}),
					"orig_height":("INT",{"default":0}),
					"target_width":("INT",{"default":0}),
					"target_height":("INT",{"default":0}),
					"crop_coords_top":("INT",{"default":0}),
					"crop_coords_left":("INT",{"default":0}),
					"aesthetic_score":("FLOAT",{"default":0}),
					"negative_aesthetic_score":("FLOAT",{"default":0}),
					"aesthetic_scale":("FLOAT",{"default":4.0}),
					"anime_scale":("FLOAT",{"default":0}),
					"photography_scale":("FLOAT",{"default":0}),
					"num_cols":("INT",{"default":1}),
					"guidance_scale":("FLOAT",{
						"default":6.0,
						"display": "number"}),
					#"latent_image": ("LATENT", ),
					#"denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
						}
				}
	RETURN_TYPES = ("IMAGE",)
	#RETURN_NAMES = ("low","high",)
	FUNCTION = "txt2img"
	CATEGORY = "PanGu_draw_v3"
	TITLE = "PanGu_draw_v3 do txt2img"

	def txt2img(self,low,high,sd_xl_base_ratios,prompt,negative_prompt,seed,orig_width,orig_height,target_width,target_height,crop_coords_top,crop_coords_left,aesthetic_score,negative_aesthetic_score,aesthetic_scale,anime_scale,photography_scale,num_cols,guidance_scale,steps):
		version_dict = VERSION2SPECS.get("PanGu-SDXL-base-1.0")
		seed_everything(seed)
		W, H = SD_XL_BASE_RATIOS[sd_xl_base_ratios]
		C = version_dict["C"]
		F = version_dict["f"]
		is_legacy = version_dict["is_legacy"]

		prompts = []
		negative_prompts = [negative_prompt]

		prompts.append(prompt)
		negative_prompts = negative_prompts * len(prompts)

		size_list = HIGH_SOLUTION_BASE_SIZE_LIST #if args.high_solution else BASE_SIZE_LIST
		assert (W, H) in size_list, f"(W, H)=({W}, {H}) is not in SIZE_LIST:{str(size_list)}"
		target_size_as_ind = size_list.index((W, H))

		value_dict = {
			"prompt": prompts,
			"negative_prompt": negative_prompt,
			"orig_width": orig_width if orig_width else W,
			"orig_height": orig_height if orig_height else H,
			"target_width": target_width if target_width else W,
			"target_height": target_height if target_height else H,
			"crop_coords_top": max(crop_coords_top if crop_coords_top else 0, 0),
			"crop_coords_left": max(crop_coords_left if crop_coords_left else 0, 0),
			"aesthetic_score": aesthetic_score if aesthetic_score else 6.0,
			"negative_aesthetic_score": negative_aesthetic_score if negative_aesthetic_score else 2.5,
			"aesthetic_scale": aesthetic_scale if aesthetic_scale else 0.0,
			"anime_scale": anime_scale if anime_scale else 0.0,
			"photography_scale": photography_scale if photography_scale else 0.0,
			"target_size_as_ind": target_size_as_ind,
		}

		sampler, num_rows, num_cols = init_sampling(
        sampler="PanGuEulerEDMSampler",
        num_cols=num_cols,
        guider="PanGuVanillaCFG",
        guidance_scale=guidance_scale,
        discretization="LegacyDDPMDiscretization",
        steps=steps,
        stage2strength=None,
        enable_pangu=True,
        other_scale=get_other_scale(value_dict),
    )
		num_samples = num_rows * num_cols
		print("Txt2Img Sampling")
		s_time = time.time()
		samples = low.pangu_do_sample(
			high,
			sampler,
			value_dict,
			num_samples,
			H,
			W,
			C,
			F,
			force_uc_zero_embeddings=["txt"] if not is_legacy else [],
			return_latents=True,
			filter=filter,
			amp_level=00,
		)
		print(f"Txt2Img sample step {sampler.num_steps}, time cost: {time.time() - s_time:.2f}s")

		return samples

def get_other_scale(value_dict):
    other_scale = []
    if "aesthetic_scale" in value_dict and value_dict["aesthetic_scale"] > 0:
        other_scale.append(value_dict["aesthetic_scale"])
    if "anime_scale" in value_dict and value_dict["anime_scale"] > 0:
        other_scale.append(value_dict["anime_scale"])
    if "photography_scale" in value_dict and value_dict["photography_scale"] > 0:
        other_scale.append(value_dict["photography_scale"])
    return other_scale


class advancedPanGuCheckpointLoader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": (folder_paths.get_filename_list("checkpoints"),),
			}
		}
	RETURN_TYPES = ("MODEL", "CLIP", "VAE")
	#RETURN_NAMES = ("low","high",)
	FUNCTION = "load_checkpoint"
	CATEGORY = "PanGu_draw_v3"
	TITLE = "PanGu_draw_v3 Checkpoint Loader"

	def load_checkpoint(self, model):
		model = folder_paths.get_full_path("checkpoints", model)
		out =PanGu_load_checkpoint(model,output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
		return out[:3]

class PanGu_Sample:
	@classmethod
	def INPUT_TYPES(s):
		return {"required":
					{"low": ("MODEL",),
					"high": ("MODEL",),
					"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
					"steps": ("INT", {"default": 40, "min": 2, "max": 10000}),
					"sd_xl_base_ratios":("STRING",{"default":1.0}),
					#"cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
					#"scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
					"prompt": ("STRING", {
                    "multiline": True, #
                    "default": "1girl"
                }),
					"negative_prompt": ("STRING", {
                    "multiline": True, 
                }),
					"orig_width":("INT",{"default":0}),
					"orig_height":("INT",{"default":0}),
					"target_width":("INT",{"default":0}),
					"target_height":("INT",{"default":0}),
					"crop_coords_top":("INT",{"default":0}),
					"crop_coords_left":("INT",{"default":0}),
					"aesthetic_score":("FLOAT",{"default":0}),
					"negative_aesthetic_score":("FLOAT",{"default":0}),
					"aesthetic_scale":("FLOAT",{"default":4.0}),
					"anime_scale":("FLOAT",{"default":0}),
					"photography_scale":("FLOAT",{"default":0}),
					"num_cols":("INT",{"default":1}),
					"guidance_scale":("FLOAT",{
						"default":6.0,
						"display": "number"}),
					#"latent_image": ("LATENT", ),
					#"denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
						}
				}





NODE_CLASS_MAPPINGS= {
    "load PanGu Draw V3 model":PanGuCheckpointLoader,
    "PanGu Draw V3 do t2i":PanGu_txt2img,
	"load PanGu Draw V3 model(advaced)":advancedPanGuCheckpointLoader
}


