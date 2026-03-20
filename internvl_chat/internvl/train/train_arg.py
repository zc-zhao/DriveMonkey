from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP. Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is v2.'}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    use_liger: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the liger kernel.'}
    )
    use_bbox: bool = field(
        default = False,
        metadata={'help': 'whether use 2d bbox projector'}
    )
    use_embedding: bool = field(
        default=False,
        metadata={'help': 'whether use query embedding'}
    )
    init_embedding: bool = field(default=False) # 是否使用图像特征初始化query
    pooling_type: Optional[str] = field(default=None) # pooling 的类型 max or avg
    embedding_num: int = field(default=0)     #embedding 的维度
    bev_tower: Optional[str] = field(default=None) # 是否使用bev特征
    bev_config: Optional[str] = field(default=None) # bev的config文件
    bev_checkpoints: Optional[str] = field(default=None) # bev的checkpoints的文件
    use_bev_decoder: bool = field(default=True) # 是否使用bev_decoder
    use_llm_bev: Optional[bool] = field(
        default=False,
        metadata={'help': 'whether use llm bev features'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic high resolution strategy. Default is False.'},
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=32,
        metadata={'help': 'The maximum number of frames for video data. Default is 32.'},
    )
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    use_nuscene: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether or not use our nuscene data. '},
    )
    nuscene_view: Optional[str] = field(
        default='surrounding',
        metadata={'help': 'The nuscene data view'},
    )
    use_myconcat_dataset: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether or not use our nuscene data. '},
    )
    # use_packed_ds: bool = field(
    #     default=False,
    #     metadata={'help': 'Whether to use packed dataset for efficient training. Default is False.'},
    # )
    # num_images_expected: int = field(
    #     default=40,
    #     metadata={'help': 'The maximum number of images per packed sample. Default is 40.'},
    # )
    # max_packed_tokens: int = field(
    #     default=8192,
    #     metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    # )
    # max_buffer_size: int = field(
    #     default=20,
    #     metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    # )
    # log_freq: int = field(
    #     default=1000,
    #     metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    # )
    # strict_mode: bool = field(
    #     default=True,
    #     metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    # )
    # replacement: bool = field(
    #     default=False,
    #     metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    # )
    # allow_overflow: bool = field(
    #     default=False,
    #     metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    # )
    # loss_reduction: str = field(
    #     default='token',
    #     metadata={'help': 'Loss reduction method. Default is token.'},
    # )
    # loss_reduction_all_gather: bool = field(
    #     default=False,
    #     metadata={'help': 'Whether to gather all during loss reduction. Default is False.'},
    # )