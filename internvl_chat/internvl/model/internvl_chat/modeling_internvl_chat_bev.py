# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
import os
import math
from typing import Any, List, Optional, Tuple, Union, Dict
from dataclasses import dataclass
from matplotlib.pyplot import get
import torch.distributed as dist
import torch.utils.checkpoint
import torch.nn.functional as F
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers.trainer_pt_utils import LabelSmoother
from .configuration_internvl_chat_bev import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn

from mmdet3d.core import bbox3d2result
from internvl.model.PETR.projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from mmdet.models.builder import build_detector
from mmcv.runner import load_checkpoint
from mmcv.utils import Config
from mmdet.models.builder import build_loss
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core import (multi_apply, multi_apply, reduce_mean, bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh)
from .loss import *

from internvl.gpu_timer import GLOBAL_TIMER

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

@dataclass
class CausalOutputWithBBOX(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pred_2d_loss: Optional[torch.FloatTensor] = None
    pred_3d_loss: Optional[torch.FloatTensor] = None
    cycle_2d_loss1: Optional[torch.FloatTensor] = None
    cycle_2d_loss2: Optional[torch.FloatTensor] = None
    llm_loss: Optional[torch.FloatTensor] = None
    log_vars: Optional[Dict] = None


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]
        # Enable Flash Attention if supported, otherwise fall back to eager attention.
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.bbox_2d_token_id = None
        self.bbox_2d_input_token_id = None
        self.bbox_3d_token_id = None
        self.bev_query_token_id = None

        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.embed_dims = 256

        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)
        
        if config.use_bbox:
            self.init_bbox_projector(config)

        
        if config.bev_tower is not None:
            self.initialize_bev_modules(config)

        self.embedding_num = 0
        if config.use_embedding:
            self.init_detection_embedding(config.embedding_num, config)
            self.embedding_num = config.embedding_num

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def init_bbox_projector(self,config):
        llm_hidden_size = config.llm_config.hidden_size
        ##*********************** for 2d *********************##
        self.bbox_2d_encoder = nn.Sequential(
            nn.Linear(4, llm_hidden_size // 2),
            nn.GELU(),
            nn.Linear(llm_hidden_size // 2, llm_hidden_size),
        ) 
        self.bbox_2d_decoder = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size // 2),
            nn.GELU(),
            nn.Linear(llm_hidden_size // 2, 4),
        )

    def init_detection_embedding(self, embed_num, config):
        self.embedding_num = embed_num
        llm_hidden_size = config.llm_config.hidden_size
        if not getattr(config, 'init_embedding', False):
            self.det_query = nn.Embedding(embed_num, llm_hidden_size)
        else:
            assert hasattr(config, 'pooling_type'), 'when init embedding, must have pooling type'
            in_len = self.num_image_token
            self.kernel_size = math.ceil(in_len / embed_num)
            self.embedding_num = math.floor((in_len - self.kernel_size) // self.kernel_size) + 1

        # petr
        if getattr(self, 'bev_tower', None) is not None and getattr(self.bev_tower.pts_bbox_head, 'add_2d_head', False) and not getattr(self.bev_tower.pts_bbox_head, 'only_2d', False):
            self.det_2d_query = nn.Embedding(6*embed_num, llm_hidden_size)
        self.det_query_projector = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size // 2),
            nn.GELU(),
            nn.Linear(llm_hidden_size // 2, 256)
        )

    def initialize_bev_modules(self, config):
        if getattr(self, 'bev_tower', None) is None:
            bev_config = config.bev_config
            cfg = Config.fromfile(bev_config)
            if hasattr(cfg, 'plugin'):
                if cfg.plugin:
                    import importlib
                    if hasattr(cfg, 'plugin_dir'):
                        plugin_dir = cfg.plugin_dir.rstrip('/')
                        module_dir = plugin_dir.replace('/', '.')
                        if module_dir.startswith('projects.'):
                            module_path = f'internvl.model.PETR.{module_dir}'
                        else:
                            module_path = module_dir
                        print('plugin_module:', module_path)
                        importlib.import_module(module_path)
                    else:
                        importlib.import_module('internvl.model.PETR.projects.mmdet3d_plugin')
            self.bev_tower = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))        #包含backbone和head
            img_norm_cfg = cfg.img_norm_cfg
            scales = next((pipeline['scales'] for pipeline in cfg.train_pipeline if 'RandomScaleImageMultiViewImage' == pipeline['type']), None)
            self.bev_tower.scales = scales #small时为0.8 tiny时为0.5
            self.bev_tower.size_divisor = 32
            self.bev_tower.mean = img_norm_cfg.mean
            self.bev_tower.std = img_norm_cfg.std
            self.bev_tower.to_rgb = img_norm_cfg.to_rgb
            if config.bev_checkpoints is not None:
                checkpoints = config.bev_checkpoints
                checkpoint = load_checkpoint(self.bev_tower, checkpoints, map_location='cpu')
            else:
                self.bev_tower.init_weights()
                if cfg.load_from is not None:
                    load_checkpoint(self.bev_tower, cfg.load_from, map_location='cpu')
            
            if config.views == 'surrounding':
                self.save_camera_view = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']
            elif config.views == 'all_front':
                self.save_camera_view = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT']
            elif config.views == 'all_back':
                self.save_camera_view = ['CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']
            elif config.views == 'front_left':
                self.save_camera_view = ['CAM_FRONT_LEFT']
            elif config.views == 'front_right':
                self.save_camera_view = ['CAM_FRONT_RIGHT']
            elif config.views == 'back_left':
                self.save_camera_view = ['CAM_BACK_LEFT']
            elif config.views == 'back_right':
                self.save_camera_view = ['CAM_BACK_RIGHT']
            elif config.views == 'front':
                self.save_camera_view = ['CAM_FRONT']
            elif config.views == 'back':
                self.save_camera_view = ['CAM_BACK']
            else:
                raise NotImplementedError(f'Not support views: {config.views}')

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            statistics: Optional[torch.LongTensor] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
            cam_path_list:Optional[List] = None,
            can_bus:Optional[List] = None,
            cams: Optional[dict] = None,
            token: Optional[str] = None,
            bbox2d_inputs: Optional[torch.Tensor] = None,  
            bbox2d_targets: Optional[torch.Tensor] = None,
            bbox3d_inputs: Optional[torch.Tensor] = None, 
            bbox3d_targets: Optional[List[torch.Tensor]] = None, 
            ids3d_targets: Optional[List[torch.Tensor]] = None,
            new_bbox2d_targets:  Optional[List[torch.Tensor]] = None, 
            camera_index: Optional[torch.Tensor] = None,
    ):# -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        det_query = None

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)

        loc_2d_embeds = None

        ###************************ old design, have bug****************************##
        if self.config.use_bbox:
            if bbox2d_inputs is not None and len(bbox2d_inputs) > 0:
                loc_2d_embeds = self.bbox_2d_encoder(bbox2d_inputs)
                selected_2d_bbox = ((input_ids == self.bbox_2d_token_id)&(labels.flatten()<0)).nonzero().flatten()
                try:
                    input_embeds[selected_2d_bbox] = input_embeds[selected_2d_bbox] * 0.0 + loc_2d_embeds.to(input_embeds)
                    ignore_flag = False
                except Exception as e:
                    print(f'warning: {e}, input_embeds[selected_2d_bbox].shape={input_embeds[selected_2d_bbox].shape}, '
                    f'loc_2d_embeds.shape={loc_2d_embeds.shape}')
                    ignore_flag = True
        ############################# petr for 2d or 3d **************************##
        if self.config.use_embedding and getattr(self, 'bev_tower', None) is not None:
            if ((getattr(self.bev_tower.pts_bbox_head, 'only_2d', False) and len(torch.cat(new_bbox2d_targets)) > 0) or \
                (bbox3d_targets is not None and len(torch.cat(bbox3d_targets)) > 0)):
                if getattr(self.config, 'init_embedding', False):
                    query_embedding = vit_embeds[camera_index].permute(0, 2, 1).contiguous()
                    if self.config.pooling_type == 'max':
                        det_query = F.max_pool1d(query_embedding, kernel_size=self.kernel_size, stride=self.kernel_size, padding=0).permute(0, 2, 1)
                    elif self.config.pooling_type == 'avg':
                        det_query = F.avg_pool1d(query_embedding, kernel_size=self.kernel_size, stride=self.kernel_size, padding=0).permute(0, 2, 1)
                else:
                    det_query = self.det_query.weight.unsqueeze(0).expand(B, -1, -1)
            elif getattr(self.bev_tower.pts_bbox_head, 'add_2d_head', False) and len(torch.cat(new_bbox2d_targets)) > 0:
                assert len(camera_index) == B, 'the camera index lenth must be equal with batch size.'
                det_2d_query_weight = self.det_2d_query.weight.unsqueeze(0).repeat(B,1,1).reshape(B*6, self.embedding_num, -1)
                mask = torch.zeros(B*6, self.embedding_num)
                mask[camera_index] = 1
                det_query = det_2d_query_weight[mask.bool()].reshape(B,self.embedding_num,-1)
            if det_query is not None:
                selected_3d_query = (input_ids == self.bev_query_token_id)
                try:
                    input_embeds[selected_3d_query] = input_embeds[selected_3d_query] * 0.0 + det_query.reshape(-1, C)
                    labels = labels.reshape(B*N)
                    labels[selected_3d_query] = IGNORE_TOKEN_ID
                    labels = labels.reshape(B, N)
                    ignore_flag = False
                except Exception as e:
                    print(f'warning: {e}, input_embeds[selected_3d_query].shape={input_embeds[selected_3d_query].shape}, '
                    f'det_query.shape={det_query.shape}')
                    ignore_flag = True
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None and not (self.config.use_bbox or self.config.use_embedding):
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0
        elif (self.config.use_bbox or self.config.use_embedding) and labels is not None:
            # Shift so that tokens < n predict n
            ###*********  mask target bbox 2d id************************
            new_labels = labels.clone()
            new_labels[(input_ids.reshape(B,N)==self.bbox_2d_token_id)&(labels>0)] = -100
            shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()   #raw labels
            shift_labels = new_labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            llm_loss = loss
            pred_2d_locs = None    
            pred_3d_locs = None
            pred_3d_cls = None
            last_hidden_states = outputs.hidden_states[-1]  #最后一层输出[batch\ token\ dim] + cycle_2d_loss1 + cycle_2d_loss2 + bbox_3d_loss
            bbox_2d_loss  = torch.tensor(0.).to(llm_loss.device)

            cycle_2d_loss1 = torch.tensor(0.).to(llm_loss.device)
            cycle_2d_loss2 = torch.tensor(0.).to(llm_loss.device)

            bbox_3d_loss = torch.tensor(0.).to(llm_loss.device)
            log_vars= None

            ##*********************************** petr for 3d ******************************##
            if bbox3d_targets is not None and len(torch.cat(bbox3d_targets)) > 0 and self.config.use_embedding:
                ### for debug select why hang
                if getattr(self, 'bev_tower', None) is not None and cams is not None:
                    img_feature, img_metas= self.encode_petr(cams)
                if bbox3d_targets is not None and len(torch.cat(bbox3d_targets)) > 0:
                    object_query_embeds = []
                    for hidden_state, new_input_id in zip(last_hidden_states, input_ids.reshape(B,N)):
                        object_query_embeds.append(hidden_state[new_input_id == self.bev_query_token_id])
                    object_query_embeds = torch.stack(object_query_embeds)
                    object_query_embeds = self.det_query_projector(object_query_embeds)
                    outs = self.bev_tower.simple_llava_test_pts(img_feature, img_metas, object_query_embeds,rescale=True)
                    losses = self.petr_loss(outs, bbox3d_targets, ids3d_targets, None)
                    bbox_3d_loss, log_vars = parse_losses(losses)
                    bbox_3d_loss = bbox_3d_loss.to('cuda')
            
            if self.config.use_bbox and not self.config.use_embedding:
                loss = bbox_2d_loss + cycle_2d_loss1 + cycle_2d_loss2 + llm_loss
                return CausalOutputWithBBOX(
                    loss = loss,
                    logits = logits,
                    past_key_values = outputs.past_key_values,
                    hidden_states = outputs.hidden_states,
                    attentions = outputs.attentions,
                    pred_2d_loss = bbox_2d_loss,
                    pred_3d_loss = None,
                    cycle_2d_loss1 = cycle_2d_loss1,
                    cycle_2d_loss2 = cycle_2d_loss2,
                    llm_loss = llm_loss,
                    log_vars= None,
                )
            elif self.config.use_embedding and not self.config.use_bbox:
                if getattr(self, 'bev_tower', None) is not None and getattr(self.bev_tower.pts_bbox_head, 'add_2d_head', False):
                    if getattr(self.bev_tower.pts_bbox_head, 'only_2d', False):
                        loss = llm_loss + bbox_2d_loss
                    else:
                        loss = llm_loss + bbox_3d_loss + bbox_2d_loss
                else:
                    loss = llm_loss + bbox_3d_loss
                return CausalOutputWithBBOX(
                    loss = loss,
                    logits = logits,
                    past_key_values = outputs.past_key_values,
                    hidden_states = outputs.hidden_states,
                    attentions = outputs.attentions,
                    pred_2d_loss = bbox_2d_loss,
                    pred_3d_loss = bbox_3d_loss,
                    cycle_2d_loss1 = None,
                    cycle_2d_loss2 = None,
                    llm_loss = llm_loss,
                    log_vars= log_vars,
                )
            else:
                loss =  llm_loss + bbox_2d_loss + cycle_2d_loss1 + cycle_2d_loss2 + bbox_3d_loss
                return CausalOutputWithBBOX(
                    loss = loss,
                    logits = logits,
                    past_key_values = outputs.past_key_values,
                    hidden_states = outputs.hidden_states,
                    attentions = outputs.attentions,
                    pred_2d_loss = bbox_2d_loss,
                    pred_3d_loss = bbox_3d_loss,
                    cycle_2d_loss1 = cycle_2d_loss1,
                    cycle_2d_loss2 = cycle_2d_loss2,
                    llm_loss = llm_loss,
                    log_vars= log_vars,
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def bev_head_forward(self, object_query_embed, bev_features):
        object_query_embed = object_query_embed.to(bev_features)
        bev_tower = self.bev_tower
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=-1)

        bs = bev_features.size(0)
        assert bs == object_query_embed.size(0)
        
        reference_points = bev_tower.pts_bbox_head.transformer.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points
        # bevformer 没有bf16类型
        bev_embed = bev_features.permute(1, 0, 2).to(dtype=torch.float32)
        query = query.permute(1, 0, 2).to(bev_embed)
        query_pos = query_pos.permute(1, 0, 2).to(bev_embed)
        
        bev_decoder = bev_tower.pts_bbox_head.transformer.decoder
        bev_decoder = bev_decoder.to(bev_embed)
        inter_states, inter_references = bev_decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=bev_tower.pts_bbox_head.reg_branches,
            cls_branches=bev_tower.pts_bbox_head.cls_branches,
            spatial_shapes=torch.tensor([[bev_tower.pts_bbox_head.bev_h, bev_tower.pts_bbox_head.bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
        )
        
        inter_states = inter_states.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference_out
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = bev_tower.pts_bbox_head.cls_branches[lvl](inter_states[lvl])
            tmp = bev_tower.pts_bbox_head.reg_branches[lvl](inter_states[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                            self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                            self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                            self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs

    def bevformer_loss_single(self, bbox_preds, gt_bboxes_list):        #MLP + BEVformer 计算loss
        batches = bbox_preds.size(0)    #batch数，默认为1
        bbox_preds_list = [bbox_preds[i] for i in range(batches)]
        assert len(bbox_preds_list) == len(gt_bboxes_list), 'the bbox_preds and gt_bboxes must have have the same number of batches'
        bbox_pred = bbox_preds_list[0]
        gt_bbox = gt_bboxes_list[0]
        assert len(bbox_pred) == len(gt_bbox)
        gt_bbox.to(dtype=torch.float32)
        bbox_pred = bbox_pred.to(gt_bbox)
        bbox_weight = torch.ones_like(gt_bbox)
        pred_loss = self.bev_tower.pts_bbox_head.loss_bbox(
            bbox_pred, gt_bbox, bbox_weight,
            avg_factor=len(bbox_pred)
        )
        pred_loss = torch.nan_to_num(pred_loss)
        return pred_loss, None
    def bevformer_loss(self, outs, gt_bboxes_3d, gt_labels_3d):     #bevformer 计算loss
        gt_bboxes_3d = [gt_bbox.to(dtype=torch.float32)for gt_bbox in gt_bboxes_3d]
        # gt_labels_3d = [gt_label.to(dtype=torch.float32)for gt_label in gt_labels_3d]
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.bev_tower.pts_bbox_head.loss(*loss_inputs)
        return losses

    def petr_loss(self, outs, gt_bboxes_3d, gt_labels_3d, gt_bboxes_2d):     #bevformer 计算loss
        gt_bboxes_3d = [gt_bbox.to(dtype=torch.float32)for gt_bbox in gt_bboxes_3d] if gt_bboxes_3d is not None else None
        gt_bboxes_2d = [gt_bbox.to(dtype=torch.float32)for gt_bbox in gt_bboxes_2d] if gt_bboxes_2d is not None else None
        gt_labels_2d = [gt_bbox.new_full((len(gt_bbox), ),0,dtype=torch.long) for gt_bbox in gt_bboxes_2d] if gt_bboxes_2d is not None else None
        if gt_bboxes_3d is not None:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.bev_tower.pts_bbox_head.loss(*loss_inputs)
        elif gt_bboxes_2d is not None:
            loss_inputs = [gt_bboxes_2d, gt_labels_2d, outs]
            losses = self.bev_tower.pts_bbox_head.loss_2d(*loss_inputs)
        return losses    

    def box_3d_loss(self, pred_3d_box, target_3d_box, isnotnan):      #使用MLP计算loss
        pred_3d_locs, pred_3d_cls = pred_3d_box[0].to(dtype= torch.float32), pred_3d_box[1].to(dtype= torch.float32)
        gt_3d_bbox, gt_3d_cls = target_3d_box[0].to(dtype= torch.float32), target_3d_box[1].to(dtype=torch.long)
        label_weights = gt_3d_bbox.new_ones(len(pred_3d_locs))
        bbox_weight = torch.ones_like(gt_3d_bbox)
        loss_bbox_item = self.loss_bbox(pred_3d_locs[isnotnan, :10], gt_3d_bbox[isnotnan, :10], bbox_weight[isnotnan, :10], avg_factor = len(target_3d_box))
        loss_cls_item = self.loss_cls(
            pred_3d_cls, gt_3d_cls, label_weights, avg_factor=len(gt_3d_cls))
        loss_dict = dict(
            loss_bbox = loss_bbox_item,
            loss_cls = loss_cls_item
        )
        return loss_dict

    def box_2d_loss(self, src_boxes, target_boxes):
        src_boxes = src_boxes.to(dtype= torch.float32)
        target_boxes = target_boxes.to(dtype = torch.float32)
        bbox_weight = torch.ones_like(target_boxes)
        loss_bbox_dict = dict(type = 'L1Loss', loss_weight = 5.0)
        loss_bbox = build_loss(loss_bbox_dict)
        bbox_loss = loss_bbox(src_boxes, target_boxes, bbox_weight, avg_factor = len(target_boxes))
        loss_iou_dict = dict(type='GIoULoss', loss_weight=2.0)
        loss_iou = build_loss(loss_iou_dict)
        img_w, img_h = [1600,1600]
        factor = src_boxes.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               src_boxes.size(0), 1)
        bboxes = bbox_cxcywh_to_xyxy(src_boxes) * factor
        bboxes_gt = bbox_cxcywh_to_xyxy(target_boxes) * factor

        iou_loss = loss_iou(bboxes, bboxes_gt, bbox_weight, avg_factor = len(bboxes_gt))
        return bbox_loss + iou_loss

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds
    
    def encode_bev(self, new_cam_path, cams,can_bus):
        images = new_cam_path
        bev_tower = self.bev_tower
        bev_tower.to(dtype=torch.float32)
        
        bev_tower.img_backbone.eval()
        bev_tower.img_neck.eval()
        bev_tower.pts_bbox_head.transformer.encoder.eval()
        
        bev_features = bev_tower.forward_llava_test(images, cams, can_bus)
        return bev_features

    def encode_petr(self, cams, degradation_type = None):
        bev_tower = self.bev_tower
        bev_tower.to(dtype=torch.float32)
        bev_tower.img_backbone.eval()
        bev_tower.img_neck.eval()
        # bev_tower.pts_bbox_head.eval()
        # 这里进的是petr的的forward_llava_test
        img_features, img_metas = bev_tower.forward_llava_test(cams, views=self.save_camera_view, degradation_type=degradation_type)
        return img_features, img_metas

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    def nuscene_chat(self, tokenizer, pixel_values, question, generation_config,
             IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        from internvl.conversation import get_conv_template

        template = get_conv_template(self.template)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        image_bs = pixel_values.shape[0]
        # print(f'dynamic ViT batch size: {image_bs}')

        # image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * image_bs + IMG_END_TOKEN

        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        # query_to_print = query.replace(image_tokens, '<image>')
        # print(query_to_print, response)
        return response #返回回答

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @torch.no_grad()
    def evaluate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            bbox2d_inputs: Optional[torch.Tensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        assert self.bbox_2d_token_id is not None
        assert self.bbox_2d_input_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                GLOBAL_TIMER.event_start('DriveMonkey/First Generation ViT')
                vit_embeds = self.extract_feature(pixel_values) #提取图像特征
                GLOBAL_TIMER.event_end('DriveMonkey/First Generation ViT')

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            
            if bbox2d_inputs is not None and len(bbox2d_inputs) > 0:
                loc_2d_embeds = self.bbox_2d_encoder(bbox2d_inputs)
                selected_2d_bbox = (input_ids == self.bbox_2d_token_id) ##old design
                # selected_2d_bbox = (input_ids == self.bbox_2d_input_token_id) 
                assert selected_2d_bbox.sum() != 0
                input_embeds[selected_2d_bbox] = loc_2d_embeds.to(input_embeds)
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        GLOBAL_TIMER.event_start('DriveMonkey/First Generation LLM')
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )
        GLOBAL_TIMER.event_end('DriveMonkey/First Generation LLM')
        return outputs
    @torch.no_grad()
    def inference(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cam_path_list:Optional[List] = None,
            can_bus:Optional[List] = None,
            cams: Optional[dict] = None,
            token: Optional[str] = None,
            bbox2d_inputs: Optional[torch.Tensor] = None,  
            bbox2d_targets: Optional[torch.Tensor] = None,
            bbox3d_inputs: Optional[torch.Tensor] = None, 
            bbox3d_targets: Optional[List[torch.Tensor]] = None, 
            ids3d_targets: Optional[List[torch.Tensor]] = None,
            new_bbox2d_targets:  Optional[List[torch.Tensor]] = None, 
            camera_index: Optional[torch.Tensor] = None,
            degradation_type: Optional[str] = None,
    ):
        assert self.img_context_token_id is not None
        assert self.bbox_2d_token_id is not None
        assert self.bbox_3d_token_id is not None
        assert self.bev_query_token_id is not None
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        det_query = None
        
        GLOBAL_TIMER.event_start('DriveMonkey/Second Inference ViT')
        vit_embeds = self.extract_feature(pixel_values)
        GLOBAL_TIMER.event_end('DriveMonkey/Second Inference ViT')
        
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)

        ####************************** use petr for 2d or 3d**********************************##
        if self.config.use_embedding and \
        ((getattr(self, 'bev_tower', None) is not None and getattr(self.bev_tower.pts_bbox_head, 'only_2d', False) and len(torch.cat(new_bbox2d_targets)) > 0) or \
            (bbox3d_targets is not None and len(torch.cat(bbox3d_targets)) > 0)):
            if getattr(self.config, 'init_embedding', False):
                query_embedding = vit_embeds[camera_index].permute(0, 2, 1).contiguous()
                if self.config.pooling_type == 'max':
                    det_query = F.max_pool1d(query_embedding, kernel_size=self.kernel_size, stride=self.kernel_size, padding=0).permute(0, 2, 1)
                elif self.config.pooling_type == 'avg':
                    det_query = F.avg_pool1d(query_embedding, kernel_size=self.kernel_size, stride=self.kernel_size, padding=0).permute(0, 2, 1)
            else:
                det_query = self.det_query.weight.unsqueeze(0).expand(B, -1, -1)
            selected_3d_query = (input_ids == self.bev_query_token_id)
            try:
                input_embeds[selected_3d_query] = input_embeds[selected_3d_query] * 0.0 + det_query.reshape(-1, C).to(input_embeds)
                labels = labels.reshape(B*N)
                labels[selected_3d_query] = IGNORE_TOKEN_ID
                labels = labels.reshape(B, N)
            except Exception as e:
                print(f'warning: {e}, input_embeds[selected_3d_query].shape={input_embeds[selected_3d_query].shape}, '
                f'det_query.shape={det_query.shape}')
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C).to(input_embeds)   #将图像token的换为图像的embedding
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token].to(input_embeds)

        input_embeds = input_embeds.reshape(B, N, C)
        
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = outputs.hidden_states[-1]
        if self.config.use_bbox or self.config.use_embedding:
            loc_positions = (labels == self.bbox_2d_token_id)
            if loc_positions.any():
                selected_2d_hidden_states = hidden_states[loc_positions.to(device = hidden_states.device)]
                pred_2d_bbox = self.bbox_2d_decoder(selected_2d_hidden_states).sigmoid()
                pred_3d_bbox = None
                return pred_2d_bbox, pred_3d_bbox , None, None
            elif getattr(self, 'bev_tower', None) is not None and getattr(self.bev_tower.pts_bbox_head, 'only_2d', False) and new_bbox2d_targets is not None and len(new_bbox2d_targets) > 0:
                img_feature, img_metas= self.encode_petr(cams, degradation_type=degradation_type)
                object_query_embeds = []
                for hidden_state, new_input_id in zip(hidden_states, input_ids.reshape(B, N)):
                    query_mask = (new_input_id == self.bev_query_token_id)
                    object_query_embeds.append(hidden_state[query_mask.to(hidden_state.device)])
                object_query_embeds = torch.stack(object_query_embeds)
                object_query_embeds = self.det_query_projector(object_query_embeds)

                bev_tower = self.bev_tower

                outs = bev_tower.simple_llava_test_pts(img_feature, img_metas, object_query_embeds, rescale=True)
                bbox_2d_pred = outs['all_2dbbox_preds'][-1][0]
                cls_2d_score = outs['all_2dcls_scores'][-1][0]

                gt_2d_bboxes = new_bbox2d_targets[0].to(bbox_2d_pred)
                gt_2d_labels = gt_2d_bboxes.new_full((len(gt_2d_bboxes), ),0,dtype=torch.long)
                gt_2d_labels = gt_2d_labels.to(device= cls_2d_score.device)
                assign_result = bev_tower.pts_bbox_head.assigner_2d.assign(bbox_2d_pred, cls_2d_score, gt_2d_bboxes,
                                                    gt_2d_labels, img_shape=[1600,1600])
                sampling_result = bev_tower.pts_bbox_head.sampler.sample(assign_result, bbox_2d_pred,
                                                    gt_2d_bboxes)
                # pos_inds = sampling_result.pos_inds
                pos_bbox = sampling_result.pos_bboxes
                pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds #pos 对应的GT序号
                pred_2d_bbox = torch.empty_like(pos_bbox)
                for i, idx in enumerate(pos_assigned_gt_inds):
                    pred_2d_bbox[idx] = pos_bbox[i]     #预测和GT对应上
                
                return pred_2d_bbox, None, cls_2d_score, bbox_2d_pred

            elif bbox3d_targets is not None and len(bbox3d_targets)>0 and getattr(self, 'bev_tower', None) is not None:
                
                img_feature, img_metas= self.encode_petr(cams, degradation_type=degradation_type)
                
                object_query_embeds = []
                for hidden_state, new_input_id in zip(hidden_states, input_ids.reshape(B, N)):
                    query_mask = (new_input_id == self.bev_query_token_id)
                    object_query_embeds.append(hidden_state[query_mask.to(hidden_state.device)])
                object_query_embeds = torch.stack(object_query_embeds)
                object_query_embeds = self.det_query_projector(object_query_embeds)

                bev_tower = self.bev_tower
            
            
                GLOBAL_TIMER.event_start('DriveMonkey/Second Inference Spatial Decoder')
                outs = bev_tower.simple_llava_test_pts(img_feature, img_metas, object_query_embeds, rescale=True)
                GLOBAL_TIMER.event_end('DriveMonkey/Second Inference Spatial Decoder')
                
                cls_score = outs['all_cls_scores'][-1][0]
                bbox_pred = outs['all_bbox_preds'][-1][0]
                gt_bboxes = bbox3d_targets[0].to(bbox_pred)
                gt_labels = ids3d_targets[0].to(device= cls_score.device)
                assign_result = bev_tower.pts_bbox_head.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                    gt_labels, None)
                sampling_result = bev_tower.pts_bbox_head.sampler.sample(assign_result, bbox_pred,
                                                    gt_bboxes)
                # pos_inds = sampling_result.pos_inds
                pos_bbox = sampling_result.pos_bboxes
                pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds #pos 对应的GT序号
                pred_3d_bbox = torch.empty_like(pos_bbox)
                for i, idx in enumerate(pos_assigned_gt_inds):
                    pred_3d_bbox[idx] = pos_bbox[i]     #预测和GT对应上
                pred_3d_bbox = denormalize_bbox(pred_3d_bbox, pc_range = self.pc_range)
                pred_2d_bbox = None
                
                return pred_2d_bbox, pred_3d_bbox, cls_score, bbox_pred
            else:
                return None, None, None, None
        else:
            return None, None, None, None
    ##new inference
    @torch.no_grad()
    def inference_for2d3d(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cam_path_list:Optional[List] = None,
        can_bus:Optional[List] = None,
        cams: Optional[dict] = None,
        token: Optional[str] = None,
        bbox2d_inputs: Optional[torch.Tensor] = None,  
        bbox2d_targets: Optional[torch.Tensor] = None,
        bbox3d_inputs: Optional[torch.Tensor] = None, 
        bbox3d_targets: Optional[List[torch.Tensor]] = None, 
        ids3d_targets: Optional[List[torch.Tensor]] = None,
        new_bbox2d_targets:  Optional[List[torch.Tensor]] = None, 
        camera_index: Optional[torch.Tensor] = None,
    ):
        assert self.img_context_token_id is not None
        assert self.bbox_2d_token_id is not None
        assert self.bbox_3d_token_id is not None
        assert self.bev_query_token_id is not None
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        det_query = None
        vit_embeds = self.extract_feature(pixel_values)
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
         ############################# petr for 2d or 3d **************************##
        if self.config.use_embedding and getattr(self, 'bev_tower', None) is not None:
            if (bbox3d_targets is not None and len(torch.cat(bbox3d_targets)) > 0):
                det_query = self.det_query.weight.unsqueeze(0).expand(B, -1, -1)
            elif getattr(self.bev_tower.pts_bbox_head, 'add_2d_head', False) and len(torch.cat(new_bbox2d_targets)) > 0:
                det_query = []
                assert len(camera_index) == B, 'the camera index lenth must be equal with batch size.'
                for i, index in enumerate(camera_index):
                    det_query.append(self.det_2d_query[index - i*6].weight)
                det_query = torch.stack(det_query)
            if det_query is not None:
                selected_3d_query = (input_ids == self.bev_query_token_id)
                try:
                    input_embeds[selected_3d_query] = input_embeds[selected_3d_query] * 0.0 + det_query.reshape(-1, C)
                    labels = labels.reshape(B*N)
                    labels[selected_3d_query] = IGNORE_TOKEN_ID
                    labels = labels.reshape(B, N)
                    ignore_flag = False
                except Exception as e:
                    print(f'warning: {e}, input_embeds[selected_3d_query].shape={input_embeds[selected_3d_query].shape}, '
                    f'det_query.shape={det_query.shape}')
                    ignore_flag = True
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)   #将图像token的换为图像的embedding
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = outputs.hidden_states[-1]
        if getattr(self, 'bev_tower', None) is not None and getattr(self.bev_tower.pts_bbox_head, 'add_2d_head', False) and len(torch.cat(new_bbox2d_targets)) > 0 and self.config.use_embedding:
            # if cams is not None:
            img_feature, img_metas= self.encode_petr(cams)
            bev_tower = self.bev_tower
            object_query_embeds = []
            for hidden_state, new_input_id in zip(hidden_states, input_ids.reshape(B,N)):
                object_query_embeds.append(hidden_state[new_input_id == self.bev_query_token_id])
            object_query_embeds = torch.stack(object_query_embeds)
            object_query_embeds = self.det_query_projector(object_query_embeds)
            outs = bev_tower.simple_llava_test_pts(img_feature, img_metas, object_query_embeds,rescale=True)
            bbox_2d_pred = outs['all_2dbbox_preds'][-1][0]
            cls_2d_score = outs['all_2dcls_scores'][-1][0]

            gt_2d_bboxes = new_bbox2d_targets[0].to(bbox_2d_pred)
            gt_2d_labels = gt_2d_bboxes.new_full((len(gt_2d_bboxes), ),0,dtype=torch.long)
            gt_2d_labels = gt_2d_labels.to(device= cls_2d_score.device)
            assign_result = bev_tower.pts_bbox_head.assigner_2d.assign(bbox_2d_pred, cls_2d_score, gt_2d_bboxes,
                                                gt_2d_labels, img_shape=[1600,1600])
            sampling_result = bev_tower.pts_bbox_head.sampler.sample(assign_result, bbox_2d_pred,
                                                gt_2d_bboxes)
            # pos_inds = sampling_result.pos_inds
            pos_bbox = sampling_result.pos_bboxes
            pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds #pos 对应的GT序号
            pred_2d_bbox = torch.empty_like(pos_bbox)
            for i, idx in enumerate(pos_assigned_gt_inds):
                pred_2d_bbox[idx] = pos_bbox[i]     #预测和GT对应上

            return pred_2d_bbox, None, cls_2d_score, bbox_2d_pred, None, None

        elif bbox3d_targets is not None and len(bbox3d_targets)>0 and self.config.use_embedding:
            img_feature, img_metas= self.encode_petr(cams)
            bev_tower = self.bev_tower
            object_query_embeds = []
            for hidden_state, new_input_id in zip(hidden_states, input_ids.reshape(B,N)):
                object_query_embeds.append(hidden_state[new_input_id == self.bev_query_token_id])
            object_query_embeds = torch.stack(object_query_embeds)
            object_query_embeds = self.det_query_projector(object_query_embeds)
            outs = bev_tower.simple_llava_test_pts(img_feature, img_metas, object_query_embeds,rescale=True)

            cls_score = outs['all_cls_scores'][-1][0]
            bbox_pred = outs['all_bbox_preds'][-1][0]

            gt_bboxes = bbox3d_targets[0].to(bbox_pred)
            gt_labels = ids3d_targets[0].to(device= cls_score.device)
            assign_result = bev_tower.pts_bbox_head.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                gt_labels, None)
            sampling_result = bev_tower.pts_bbox_head.sampler.sample(assign_result, bbox_pred,
                                                gt_bboxes)
            # pos_inds = sampling_result.pos_inds
            pos_bbox = sampling_result.pos_bboxes
            pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds #pos 对应的GT序号
            pred_3d_bbox = torch.empty_like(pos_bbox)
            for i, idx in enumerate(pos_assigned_gt_inds):
                pred_3d_bbox[idx] = pos_bbox[i]     #预测和GT对应上
            pred_3d_bbox = denormalize_bbox(pred_3d_bbox, pc_range = self.pc_range)
            pred_2d_bbox = None
            return pred_2d_bbox, pred_3d_bbox, None, None, cls_score, bbox_pred
        else:
            return None, None, None, None, None, None
    ##new inference
    @torch.no_grad()
    def inference_2dbbox(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cam_path_list:Optional[List] = None,
            can_bus:Optional[List] = None,
            cams: Optional[dict] = None,
            token: Optional[str] = None,
            bbox2d_inputs: Optional[torch.Tensor] = None,  
            bbox2d_targets: Optional[torch.Tensor] = None,
            bbox3d_inputs: Optional[torch.Tensor] = None, 
            bbox3d_targets: Optional[List[torch.Tensor]] = None, 
            ids3d_targets: Optional[List[torch.Tensor]] = None,
            new_bbox2d_targets:  Optional[List[torch.Tensor]] = None, 
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            **generate_kwargs,
    ):
        assert self.img_context_token_id is not None
        assert self.bbox_2d_token_id is not None
        assert self.bbox_2d_input_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values) #提取图像特征

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            
            if bbox2d_inputs is not None and len(bbox2d_inputs) > 0:
                loc_2d_embeds = self.bbox_2d_encoder(bbox2d_inputs)
                # selected_2d_bbox = (input_ids == self.bbox_2d_token_id) ##old design
                selected_2d_bbox = (input_ids == self.bbox_2d_input_token_id) 
                assert selected_2d_bbox.sum() != 0
                input_embeds[selected_2d_bbox] = loc_2d_embeds.to(input_embeds)
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=True,
            return_dict_in_generate=True,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )
        hidden_states = outputs.hidden_states
        generated_output_ids = outputs.sequences
        output_hidden_states = torch.cat([hidden[-1] for hidden in hidden_states[1:]], dim = 1)
        hidden_states_len = output_hidden_states.shape[1]
        if getattr(self, 'bev_tower', None) is not None and getattr(self.bev_tower.pts_bbox_head, 'only_2d', False) and new_bbox2d_targets is not None and len(new_bbox2d_targets) > 0:
            mask = generated_output_ids[...,1:-1] == self.bev_query_token_id
            select_hidden_states = output_hidden_states[mask]
            object_query_embeds = self.det_query_projector(select_hidden_states)
            
            img_feature, img_metas= self.encode_petr(cams)
            bev_tower = self.bev_tower

            outs = bev_tower.simple_llava_test_pts(img_feature, img_metas, object_query_embeds.unsqueeze(0), rescale=True)
            bbox_2d_pred = outs['all_2dbbox_preds'][-1][0]
            cls_2d_score = outs['all_2dcls_scores'][-1][0]

            gt_2d_bboxes = new_bbox2d_targets[0].to(bbox_2d_pred)
            gt_2d_labels = gt_2d_bboxes.new_full((len(gt_2d_bboxes), ),0,dtype=torch.long)
            gt_2d_labels = gt_2d_labels.to(device= cls_2d_score.device)
            assign_result = bev_tower.pts_bbox_head.assigner_2d.assign(bbox_2d_pred, cls_2d_score, gt_2d_bboxes,
                                                gt_2d_labels, img_shape=[1600,1600])
            sampling_result = bev_tower.pts_bbox_head.sampler.sample(assign_result, bbox_2d_pred,
                                                gt_2d_bboxes)
            pos_bbox = sampling_result.pos_bboxes
            pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds #pos 对应的GT序号
            pred_2d_bbox = torch.empty_like(pos_bbox)
            for i, idx in enumerate(pos_assigned_gt_inds):
                pred_2d_bbox[idx] = pos_bbox[i]     #预测和GT对应上

            return generated_output_ids, pred_2d_bbox, None, cls_2d_score, bbox_2d_pred
        elif self.config.use_bbox and bbox2d_targets is not None and len(bbox2d_targets) > 0:
            mask = generated_output_ids[...,1:-1] == self.bbox_2d_token_id
            select_hidden_states = output_hidden_states[mask]
            pred_2d_bbox = self.bbox_2d_decoder(select_hidden_states)
            return generated_output_ids, pred_2d_bbox, None, None, None
        else:
            return generated_output_ids, None, None, None, None

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
