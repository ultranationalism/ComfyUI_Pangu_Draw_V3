# ComfyUI_Pangu_Draw_V3

[中文](./README_CN.md)|[English](./README.md)



## ComfyUI_Pangu_Draw_V3



This program is used to support the operation of **PanGu Draw V3 SDXL** on **ComfyUI**.



# Warning

1. This model consists of two sub-models, one is the high_timestep model and the other is the low_timestep model. The running program needs to use both models simultaneously.

2. The running program of this model is still under development. It can only implement the text-to-image function for now. Please wait for further development if you need more functions.

3. This model has 5B parameters, and the running program is poorly optimized. Please make sure you have at least 20G of VRAM to run this program.

4. Comfyui and webui do not support this model，So I created a comfyui node to run this model.You cannot run this model before these two UIs support it. 

5. This node uses **sgm**. Please ensure that there are no other sgm in your Python environment, otherwise an **error **will be reported

6. This project is in the development stage, and bugs and errors may occur. Please understand the risks and limitations of use.

7. This node will download ViT-bigG-14-laion2b_s39b_b160k, a clip model close to 11G



## Installation

```
git clone https://github.com/ultranationalism/ComfyUI_Pangu_Draw_V3.git
```

to your ComfyUI `custom_nodes` directory

```
cd ComfyUI_Pangu_Draw_V3
```

```
pip install -r requirement.txt
```

This requirement. txt is not complete enough.You may be missing some packages. If you are informed by the program that certain packages are missing, please submit an issue



## usage



use

> PanGuCheckpointLoader

to load two models and then use

> PanGu_txt2img 

to do t2i



## PanGu Draw 3.0



This folder contains **PanGu Draw 3.0** models implemented with MindSpore.and  I migrated it from MindSpore to Torch and implemented it to run on ComfyUI.

### Original Features

In contrast to version 2.0, Pangu Draw 3.0 has been subject to experimentation and updates across various aspects, including multi-language support, diverse resolutions, improved image quality, and model scaling. This includes:

- [x] The current industry's largest 5-billion-parameter Chinese text-to-image model.

- [x] Supports bilingual input in both Chinese and English.

- [x] Supports output of native 1K resolution images.

- [x] Outputs images in multiple size ratios.

- [x] Quantifiable stylized adjustments: cartoon, aesthetic, photography controller.

- [x] Based on Ascend+MindSpore for large-scale training and inference, using a self-developed MindSpore platform and Ascend 910 hardware.

- [x] Utilizes self-developed RLAIF to enhance image quality and artistic expression.



# About Me

I am a noob programmer, this program is a by-product of my learning stable diffusion, I learn sgm while replicating mindone’s program, so I made a very unremarkable program, but fortunately this program ran successfully, thanks to mindspore-lab’s models and programs. Thanks to kblueleaf for guiding me.