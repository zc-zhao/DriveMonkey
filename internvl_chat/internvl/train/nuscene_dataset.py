import logging
import os
import sys
import math
import traceback
import random
import re
import warnings
import time
import numpy as np
from typing import Dict, TypeVar, Iterable
from copy import deepcopy
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset
import json
import pickle
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from .dataset_packed import PackedDataset
from .dataset_utils import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    check_conversations_repetition,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm,
                                    preprocess_internvl2_5, preprocess_mpt,
                                    preprocess_phi3)
from .constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
                                    
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)
# Set constants for image processing and logging
from utils import *

T_co = TypeVar('T_co', covariant=True)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
logger = logging.getLogger(__name__)
logger.setLevel(20)
set_verbosity(20)
enable_default_handler()
enable_explicit_format()

class MyConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
        dataset_ratio (sequence): Ratio of each dataset in the batch
        bs (int): Batch size
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset], bs: int): #-> None:
        super(MyConcatDataset, self).__init__()

        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
            
        self.bs = bs
        self.cur_dataset_index = 0
        self.dataset_length = [len(data) for data in self.datasets]
        self.cumulative_size = self.cumsum(self.datasets)
        self.cnt = 0

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_size", DeprecationWarning, stacklevel=2)
        return self.cumulative_size

    def update_dataset_index(self):
        tempt = self.cur_dataset_index
        tempt += 1
        tempt = tempt % len(self.datasets)
        self.cur_dataset_index = tempt

    def __len__(self):
        return self.cumulative_size[-1]

    def __getitem__(self, idx):
        cur_dataset_len = self.dataset_length[self.cur_dataset_index]
        data_idx = idx % cur_dataset_len
        output_data = self.datasets[self.cur_dataset_index][data_idx]
        self.cnt += 1
        if self.cnt == self.bs:
            self.cnt = 0
            self.update_dataset_index()
        return output_data

class NusceneDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        num_query_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        repeat_time=1,
        normalize_type='imagenet',
        nuscene_view='surrounding',
        num_bev_token = None, # currently not used, only the model input num_bev_token is used
    ):
        super(NusceneDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        self.num_query_token = num_query_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] num_query_token: {num_query_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square

        # TODO: quick resume
        self._state_dict = {}

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl') or meta['annotation'].endswith('pkl') or meta['annotation'].endswith('json')

        if meta['annotation'].endswith('pkl'):
            self.raw_data = pickle.load(open(meta['annotation'], 'rb'))
        elif meta['annotation'].endswith('json'):
            self.raw_data = json.load(open(meta['annotation'], 'r'))
        else:
            NotImplementedError
        
        if repeat_time < 1:
            # If repeat_time is less than 1, select a portion of the data
            self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
        else:
            assert isinstance(repeat_time, int)
            # Repeat the list if repeat_time is greater than 1
            self.raw_data = self.raw_data * repeat_time

        self.views = nuscene_view
        self.camera_view = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']
        
        if self.views == 'all_front':
            self.save_camera_view = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT']
        elif self.views == 'all_back':
            self.save_camera_view = ['CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']
        elif self.views == 'front_left':
            self.save_camera_view = ['CAM_FRONT_LEFT']
        elif self.views == 'front_right':
            self.save_camera_view = ['CAM_FRONT_RIGHT']
        elif self.views == 'back_left':
            self.save_camera_view = ['CAM_BACK_LEFT']
        elif self.views == 'back_right':
            self.save_camera_view = ['CAM_BACK_RIGHT']
        elif self.views == 'front':
            self.save_camera_view = ['CAM_FRONT']
        elif self.views == 'back':
            self.save_camera_view = ['CAM_BACK']
        elif self.views == 'surrounding':
            self.save_camera_view = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']
        elif self.views == 'bev':
            self.save_camera_view = ['BEV VIEW']
            assert num_bev_token is not None, 'num_bev_token should be provided when using bev view'
            self.num_bev_token = num_bev_token
            self.cams_info = pickle.load(open("/mnt/vol1/zhaozc_workspace/project/InternVL25/internvl_chat/nuscene_internvl/nuscenes_cams_info.pkl", 'rb'))
        else:
            raise NotImplementedError(f'Not support views: {self.views}')
        
        self.num_views = len(self.camera_view)
        self.begin_str = self._get_intro_string()
        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                # data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def _get_intro_string(self):
        camera_str = ''
        
        if self.views == 'bev':
            return f"The <image> presents a bird's-eye view (BEV) of the surrounding scene of ego vehicles.\n "
        else:
            for i,camera in enumerate(self.camera_view):
                if i != self.num_views-1:
                    camera_str = camera_str + camera.strip()[4:].lower().replace('_',' ') + ','
                else:
                    camera_str = camera_str + 'and ' + camera.strip()[4:].lower().replace('_',' ')
            # camera_view = [camera.strip()[4:].lower().replace('_',' ') for camera in self.camera_view]

            # if self.num_views == 1:
            #     prompt = f"The <image> presents an overview of the {self.views} scene of ego vehicles from the {camera_str} perspective of the ego vehicle.\n "
            # elif self.num_views == 3:
            #     prompt = f"The " + f"<image> "*self.num_views +\
            #     f"present an overview of the {self.views} scene of ego vehicles, sequentially from the {camera_str} perspectives of the ego vehicle.\n "
            # else:
            #     prompt = f"The " + f"<image> "*self.num_views +\
            #     f"present an overview of the surrounding scene of ego vehicles, sequentially from the {camera_str} perspectives of the ego vehicle.\n "
            
            # return prompt
            
            return f"The " + f"<image> "*self.num_views +\
                f"present an overview of the surrounding scene of ego vehicles, sequentially from the {camera_str} perspectives of the ego vehicle.\n "

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        elif self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def multi_modal_multi_image_get_item(self, data_item):

        if data_item['conversations'][0]['from'] != 'human':
            data_item['conversations'] = data_item['conversations'][1:]
        data_item['conversations'][0]['value'] = self.begin_str + data_item['conversations'][0]['value']
        try:
            question_view = re.findall(r'<CAM_\w+>', data_item['conversations'][0]['value'])
            question_view = list(set(question_view))[0]
            question_view = question_view.replace('<','').replace('>','')
            question_view_index = self.camera_view.index(question_view)
        except:
            question_view_index = 0
        assert '<image>' in data_item['conversations'][0]['value'], 'The first conversation must have image. '

        cam_images = []
        new_cam_path = {}
        num_patches_list = []
        cam_path = data_item['cam_path']
        num_image = len(cam_path)

        for key in self.camera_view:
            if key in cam_path:
                if key in self.save_camera_view:
                    try:
                        new_cam_path[key] = os.path.join(self.root, cam_path[key].split('samples/')[-1])
                        image = Image.open(new_cam_path[key]).convert('RGB')
                    except:
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                        new_cam_path[key] = os.path.join(self.root, cam_path[key].split('samples/')[-1])
                        image = Image.open(new_cam_path[key]).convert('RGB')
                else:
                    new_cam_path[key] = os.path.join(self.root, cam_path[key].split('samples/')[-1])
                    image = Image.open(new_cam_path[key]).convert('RGB')
                    image = Image.new('RGB', (image.width, image.height), (0, 0, 0))
                if self.dynamic_image_size:
                    images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=max(1, self.max_dynamic_patch // num_image),
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                    num_patches_list.append(len(images))
                    cam_images.extend(images)
                else:
                    cam_images.append(image)
                    num_patches_list.append(1)

        # Build transformation function
        transform = self.get_transform()

        bbox2d_inputs = []
        bbox2d_targets = []
        bbox3d_inputs = [] 
        bbox3d_targets = []
        ids3d_targets = []
        source = deepcopy(data_item['conversations'])
        ###********************* old design , have bug*****************************###
        # for sentence in source:
        #     bbox_2d_seq = sentence.get('bboxs_2d_seq', None)
        #     bbox_3d_seq = sentence.get('bboxs_3d_seq', None)
        #     if bbox_2d_seq is not None:
        #         bbox2d_inputs.extend(flatten(bbox_2d_seq))
        #         if sentence['from'] == 'gpt':
        #             bbox2d_targets.extend(flatten(bbox_2d_seq))
        #     if bbox_3d_seq is not None:
        #         bbox3d_inputs.extend(flatten(bbox_3d_seq, is_3d=True))
        #         if sentence['from'] == 'gpt':
        #             bbox3d_targets.extend(flatten(bbox_3d_seq, is_3d=True))
        #             ids3d_targets.extend(sentence.get('ids'))
        ##**************************** new design  xyxy*************************************##
        # for sentence in source:
        #     bbox_2d_seq = sentence.get('bboxs_2d_seq', None)
        #     bbox_3d_seq = sentence.get('bboxs_3d_seq', None)
        #     if bbox_2d_seq is not None:
        #         if sentence['from'] == 'human':
        #             bbox2d_inputs.extend(flatten(bbox_2d_seq))
        #         if sentence['from'] == 'gpt':
        #             bbox2d_targets.extend(flatten(bbox_2d_seq))
        #     if bbox_3d_seq is not None:
        #         bbox3d_inputs.extend(flatten(bbox_3d_seq, is_3d=True))
        #         if sentence['from'] == 'gpt':
        #             bbox3d_targets.extend(flatten(bbox_3d_seq, is_3d=True))
        #             ids3d_targets.extend(sentence.get('ids'))
        ##**************************** new design  cxcywh*************************************##
        for sentence in source:
            bbox_2d_seq = sentence.get('bboxs_2d_seq', None)
            bbox_3d_seq = sentence.get('bboxs_3d_seq', None)
            if bbox_2d_seq is not None:
                if sentence['from'] == 'human':
                    bbox2d_inputs.extend(cxcywh_flatten(bbox_2d_seq))
                if sentence['from'] == 'gpt':
                    bbox2d_targets.extend(cxcywh_flatten(bbox_2d_seq))
            if bbox_3d_seq is not None:
                bbox3d_inputs.extend(cxcywh_flatten(bbox_3d_seq, is_3d=True))
                if sentence['from'] == 'gpt':
                    bbox3d_targets.extend(cxcywh_flatten(bbox_3d_seq, is_3d=True))
                    ids3d_targets.extend(sentence.get('ids'))

        pixel_values = [transform(image) for image in cam_images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        if self.views == 'bev':
            num_image_tokens = [self.num_bev_token]
            num_image = 1
            ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                    self.tokenizer, num_image_tokens, num_query_token = self.num_query_token, group_by_length=self.group_by_length,
                                    ds_name=self.ds_name, num_image=num_image)
        else:
            # Preprocess the conversations and generate the return dictionary
            num_image_tokens = [self.num_image_token * num_tile for num_tile in num_patches_list]
            ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                    self.tokenizer, num_image_tokens, num_query_token = self.num_query_token, group_by_length=self.group_by_length,
                                    ds_name=self.ds_name, num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        assert len(bbox3d_targets) == len(ids3d_targets)
        ret.update(dict(
            bbox2d_inputs = bbox2d_inputs,
            bbox2d_targets = bbox2d_targets,
            bbox3d_inputs = bbox3d_inputs ,
            bbox3d_targets = bbox3d_targets,
            ids3d_targets = ids3d_targets, 
            ))
        ##必须和forward的输入名称相同,不然会被屏蔽
        ###transformers/trainer.py
        ###看transformers库的trainer.py的第735行的_get_collator_with_removed_columns函数
        ret['cam_path_list'] = [new_cam_path]

        if self.views == 'bev':
            cams_info = self.cams_info[data_item['token']]
            ret['cams'] = cams_info
        else:
            ret['cams'] = data_item['cams'] if 'cams' in data_item else None

        ret['token'] = data_item['token']
        ret['camera_index'] = question_view_index

        return ret

    def __getitem__(self, i): #-> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = self.raw_data[i]
                # conversations = data_item['conversations']
                # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
                assert 'cam_path' in data_item and len(data_item['cam_path']) != 0, 'The Nuscene Data must have cam path'
                ret = self.multi_modal_multi_image_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name, flush=True)
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    traceback.print_exc()
                data_item = self.raw_data[i]
                if 'cam_path' in data_item:
                    if type(data_item['cam_path']) == dict:
                        images = [os.path.join(self.root, data_item['cam_path'][key].split('samples/')[-1]) for key in self.camera_view]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['cam_path'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['cam_path'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret



def build_nuscene_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    normalize_type='imagenet',
):
    datasets = []
    lengths = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        in_repeat_time = ds_collections[ds_name]['in_repeat_time']  if 'in_repeat_time' in ds_collections[ds_name] else 1
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        start_time = time.time()
        dataset = NusceneDataset(
            data_args.conv_style, ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            num_query_token = model.embedding_num if hasattr(model, 'embedding_num') else 0,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=in_repeat_time,
            normalize_type=normalize_type,
            nuscene_view=data_args.nuscene_view,
            num_bev_token=model.num_bev_token if hasattr(model, 'num_bev_token') else None,
        )
        end_time = time.time()
        load_time = end_time - start_time
        logger.info(f'load dataset {ds_name} spend {load_time}s.')
        repeat_time = 1 if repeat_time < 1 else repeat_time  # don't repeat if repeat_time is less than 1
        for i in range(repeat_time):
            logger.info(f'Add dataset:{ds_name}_{i} with length: {len(dataset)}')
            datasets.append(dataset)
        # logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        # datasets.append(dataset)
            if data_args.use_data_resampling:
                lengths.append(math.sqrt(len(dataset)))
            else:
                lengths.append(len(dataset))

    if data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    elif data_args.use_myconcat_dataset:
        train_dataset = MyConcatDataset(datasets, bs=16)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset