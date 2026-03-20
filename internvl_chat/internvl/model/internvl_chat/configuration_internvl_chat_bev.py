# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from internvl.model.internlm2.configuration_internlm2 import InternLM2Config
from internvl.model.phi3.configuration_phi3 import Phi3Config
from transformers import AutoConfig, LlamaConfig, Qwen2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_intern_vit import InternVisionConfig

logger = logging.get_logger(__name__)


class InternVLChatConfig(PretrainedConfig):
    model_type = 'internvl_chat'
    is_composition = True

    def __init__(
            self,
            vision_config=None,
            llm_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            pad2square=False,
            select_layer=-1,
            force_image_size=None,
            downsample_ratio=0.5,
            template=None,
            dynamic_image_size=False,
            use_thumbnail=False,
            ps_version='v1',
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            use_bbox = False,
            init_embedding = False,
            pooling_type = 'max',
            use_embedding = False,
            embedding_num = 0,
            views = 'surrounding',
            bev_tower = None,
            bev_config = None,
            bev_checkpoints = None,
            **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {'architectures': ['InternVisionModel']}
            logger.info('vision_config is None. Initializing the InternVisionConfig with default values.')

        if llm_config is None:
            # TODO: There might still be a bug in transformers version 4.44 and above.
            llm_config = {'architectures': ['']}
            logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')

        self.vision_config = InternVisionConfig(**vision_config)
        if llm_config['architectures'][0] == 'LlamaForCausalLM':
            self.llm_config = LlamaConfig(**llm_config)
        elif llm_config['architectures'][0] == 'InternLM2ForCausalLM':
            self.llm_config = InternLM2Config(**llm_config)
        elif llm_config['architectures'][0] == 'Phi3ForCausalLM':
            self.llm_config = Phi3Config(**llm_config)
        elif llm_config['architectures'][0] == 'Qwen2ForCausalLM':
            self.llm_config = Qwen2Config(**llm_config)
        else:
            raise ValueError('Unsupported architecture: {}'.format(llm_config['architectures'][0]))
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch

        self.use_bbox = use_bbox
        self.use_embedding = use_embedding
        self.embedding_num = embedding_num
        self.init_embedding = init_embedding
        if self.init_embedding:
            self.pooling_type = pooling_type
        self.views = views
        self.bev_tower = bev_tower
        self.bev_config = bev_config
        self.bev_checkpoints = bev_checkpoints

        self.hidden_size = self.llm_config.hidden_size
        # By default, we use tie_word_embeddings=False for models of all sizes.
        self.tie_word_embeddings = False
        self.llm_config.tie_word_embeddings = self.tie_word_embeddings

        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['ps_version'] = self.ps_version
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch

        output['use_bbox'] = self.use_bbox
        output['use_embedding'] = self.use_embedding
        output['embedding_num'] = self.embedding_num
        output['bev_tower'] = self.bev_tower
        output['bev_config'] = self.bev_config
        output['bev_checkpoints'] = self.bev_checkpoints

        return output
