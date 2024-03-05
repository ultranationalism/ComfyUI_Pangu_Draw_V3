import os
import json
import torch
import comfy
import numpy as np

from comfy.model_patcher import ModelPatcher
from comfy.clip_model import CLIPTextModel
from comfy.ops import manual_cast
from ..sgm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.supported_models import SDXL
from comfy.model_base import BaseModel,ModelType,Timestep,CLIPEmbeddingNoiseAugmentation,sdxl_pooled,model_sampling
from comfy.sdxl_clip import SDXLTokenizer,SDXLClipModel
from comfy.sd1_clip import SDTokenizer,ClipTokenWeightEncoder,SDClipModel,unescape_important,escape_important
from comfy.supported_models_base import ClipTarget

from ..sgm.modules.embedders.tokenizer.simple_tokenizer import WordpieceTokenizer
from ..sgm.modules.encoders.modules import FrozenCnCLIPEmbedder
from ..sgm.util import (append_dims, autocast, count_params, default,
                     disabled_train, expand_dims_like, instantiate_from_config)




class PanGu_SDXL_UNET(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EDM, device=None,):
        super().__init__(model_config=model_config,model_type=model_type, device=device)
        self.embedder = Timestep(256)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(**{"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1280})


    def encode_adm(self, **kwargs):
        clip_pooled = sdxl_pooled(kwargs, self.noise_augmentor)
        target_size_as_ind=kwargs.get("target_size_as_ind",5)
        """
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)
        target_width = kwargs.get("target_width", width)
        target_height = kwargs.get("target_height", height)
        """

        out = []
        out.append(self.embedder(torch.Tensor([target_size_as_ind])))
        """
        out.append(self.embedder(torch.Tensor([height])))
        out.append(self.embedder(torch.Tensor([width])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([target_height])))
        out.append(self.embedder(torch.Tensor([target_width])))
        """
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)
    

class PanGu_ModelPatcher(ModelPatcher):
    def __init__(self, model, load_device, offload_device, size=0, current_device=None, weight_inplace_update=False):
        super().__init__(model, load_device, offload_device, size=0, current_device=None, weight_inplace_update=False)


class PanGu_SDXL(SDXL):
    unet_config ={
        'use_checkpoint': False, 
        'out_channels': 4, 
        'use_spatial_transformer': True,
        'legacy': False,
        'num_classes': 'sequential', 
        'adm_in_channels': 1536,
        'dtype': torch.float16,
        'in_channels': 4, 'model_channels': 320,
        'image_size': 32,
        'num_res_blocks': [2, 2, 2], 
        'transformer_depth': [0, 0, 2, 2, 10, 10], 
        'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
        'use_linear_in_transformer': True, 
        'context_dim': 2048, 
        'num_head_channels': 64, 
        'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
        'use_temporal_attention': False, 
        'use_temporal_resblock': False
        }
    """{   
        'adm_in_channels': 1536,
        'num_classes': 'sequential',
        'in_channels': 4,
        'out_channels': 4,
        'model_channels': 320,
        'attention_resolutions': [4, 2],
        'num_res_blocks': 2,
        'channel_mult': [1, 2, 4],
        'num_head_channels': 64,
        'use_linear_in_transformer': True,
        'transformer_depth': [1, 2, 10],
        'context_dim': 2048,
        'spatial_transformer_attn_type': 'softmax-xformers'
        }"""


    """    
    {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "context_dim": 2048,
        "adm_in_channels": 1536,
        "use_temporal_attention": False,
        "in_channels": 4,
        "out_channels": 4,
        "num_res_blocks": 2
    }"""


    def __init__(self,unet_config=unet_config):
        super().__init__(unet_config)
        latent_format = SDXL.latent_format
    
    def clip_target(self):
        return ClipTarget(PanGu_SDXLTokenizer, PanGu_SDXLClipModel)
    
    def get_model(self, state_dict, prefix="", device=None):
        out = PanGu_SDXL_UNET(self, model_type=self.model_type(state_dict, prefix), device=device)
        if self.inpaint_model():
            out.set_inpaint()
        return out



class wordPT(WordpieceTokenizer):
    def __init__(self):
        super().__init__()
        self.max_length=77
    def tokenize(self, texts):
        SOT_TEXT = "[CLS]"
        EOT_TEXT = "[SEP]"
        CONTEXT_LEN = self.max_length
        out=[]
        if isinstance(texts, str):
            texts = [texts]
        #sot_token = self.encoder[SOT_TEXT]
        #eot_token = self.encoder[EOT_TEXT]
        #all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        temps=[self.encode(text) for text in texts]
        for temp in temps:
            out.extend(temp)
        print(out)
        return out
        """
        result = np.zeros((len(all_tokens), CONTEXT_LEN))

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > CONTEXT_LEN:
                tokens = tokens[: CONTEXT_LEN - 1] + [eot_token]
            result[i, : len(tokens)] = tokens

        result = np.array(result, dtype=np.int32)
        result_tensor = torch.from_numpy(result)
        return result_tensor, None  # inplace for length"""
    
    def __call__(self, text):
        return self.tokenize(text)


class Wrodpiece(SDTokenizer):

    def __init__(self,embedding_directory=None):
        super().__init__(embedding_directory)
        tokenizer_path=None
        self.tokenizer=wordPT()
    
    def tokenize_with_weights(self, text:str, return_word_ids=False):
        '''
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed CLIP tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of CLIP
        '''
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0

        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        #tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = unescape_important(weighted_segment).replace("\n", " ").split(' ')
            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                #if we find an embedding, deal with the embedding
                if word.startswith(self.embedding_identifier) and self.embedding_directory is not None:
                    embedding_name = word[len(self.embedding_identifier):].strip('\n')
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        print(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                    #if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                #parse word
                tokens.append([(t, weight) for t in self.tokenizer(word)])

        #reshape token array to CLIP input size
        batched_tokens = []
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0, 0))
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            #determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - 1:
                    remaining_length = self.max_length - len(batch) - 1
                    #break word in two and add end token
                    if is_large:
                        batch.extend([(t,w,i+1) for t,w in t_group[:remaining_length]])
                        batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    #add end token and pad
                    else:
                        batch.append((self.end_token, 1.0, 0))
                        if self.pad_to_max_length:
                            batch.extend([(pad_token, 1.0, 0)] * (remaining_length))
                    #start new batch
                    batch = []
                    if self.start_token is not None:
                        batch.append((self.start_token, 1.0, 0))
                    batched_tokens.append(batch)
                else:
                    batch.extend([(t,w,i+1) for t,w in t_group])
                    t_group = []

        #fill last batch
        batch.append((self.end_token, 1.0, 0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.max_length - len(batch)))
        if self.min_length is not None and len(batch) < self.min_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.min_length - len(batch)))

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w,_ in x] for x in batched_tokens]

        return batched_tokens
    
def token_weights(string:str, current_weight):
    # 调用parse_parentheses函数，将字符串按照括号分割成若干子字符串，存储在a中
    a = parse_parentheses(string)
    # 初始化一个空列表out，用于存储最终的结果
    out = []
    # 遍历a中的每个子字符串x
    for x in a:
        # 初始化一个变量weight，赋值为当前的权重current_weight
        weight = current_weight
        # 如果x的长度大于等于2，并且x的首尾都是英文或中文的括号，增加对中文括号的判断
        if len(x) >= 2 and (x[-1] == ')' or x[-1] == '）') and (x[0] == '(' or x[0] == '（'):
            # 去掉x的首尾括号
            x = x[1:-1]
            # 在x中从右往左找到第一个冒号的位置xx，增加对中文冒号的判断
            xx = x.rfind(":") if ":" in x else x.rfind("：")
            # 将weight乘以1.1
            weight *= 1.1
            # 如果找到了冒号
            if xx > 0:
                # 尝试将冒号后面的部分转换成浮点数，并赋值给weight
                try:
                    weight = float(x[xx+1:])
                    # 将x截取为冒号前面的部分
                    x = x[:xx]
                # 如果转换失败，忽略这个错误
                except:
                    pass
            # 递归调用token_weights函数，将x和weight作为参数，将返回的列表添加到out列表中
            out.extend(token_weights(x, weight))
        # 否则
        else:
            # 将x和weight作为一个元组，添加到out列表中
            out.append((x, current_weight))
    # 返回out列表
    return out

def parse_parentheses(string):
    result = []
    current_item = ""
    nesting_level = 0
    parentheses = ["(", ")", "（", "）"]
    last_left = None
    for char in string:
        try:
            index = parentheses.index(char)
            if index % 2 == 0: # 左括号
                if nesting_level == 0:
                    if current_item:
                        result.append(current_item)
                        current_item = char
                    else:
                        current_item = char
                else:
                    current_item += char
                nesting_level += 1
                last_left = index
            else: # 右括号
                nesting_level -= 1
                if nesting_level == 0:
                    if parentheses.index(char) == last_left + 1: # 匹配的右括号
                        result.append(current_item + char)
                        current_item = ""
                    else: # 不匹配的右括号
                        pass # 或者抛出 SyntaxError 异常
                else:
                    current_item += char
        except ValueError: # 不是括号符号
            current_item += char
    if current_item:
        result.append(current_item)
    return result



class PanGu_SDXLTokenizer(SDXLTokenizer):
    def __init__(self,embedding_directory=None):
        super().__init__(embedding_directory)
        self.clip_l=Wrodpiece(embedding_directory=None)

class PanGu_SDClipModel(SDClipModel):
    def __init__(self,embedding_directory=None,dtype=None,device="cpu",model_class=CLIPTextModel):
        super().__init__(embedding_directory,dtype,device,model_class)
    
class PanGu_SDXLClipModel(SDXLClipModel):
    def __init__(self,device,dtype):
        super().__init__(device,dtype)
        self.clip_l =PanGu_SDClipModel()