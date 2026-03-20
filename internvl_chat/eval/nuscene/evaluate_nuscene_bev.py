import argparse
import itertools
import json
import pickle
import os
import random
import time
from functools import partial
from tkinter import BEVEL
from typing_extensions import Self
from black import out
import re
import torch
from copy import deepcopy
# from internvl.model.internvl_chat import InternVLChatModel
from internvl.model.internvl_chat.modeling_internvl_chat_bev import InternVLChatModel
from internvl.train.dataset_utils import (build_transform, dynamic_preprocess,preprocess,
                                    preprocess_internlm,preprocess_mpt,
                                    preprocess_phi3)
from mmdet.core import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from PIL import Image
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import AutoTokenizer
from internvl.train.constants import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
                        SIGLIP_MEAN, SIGLIP_STD,BBOX_2D_TOKEN,BBOX_2D_INPUT_TOKEN,BBOX_3D_TOKEN,BEV_QUERY_TOKEN)
from internvl.conversation import get_conv_template

ds_collections = {
    'nusceneqa_3dval_all': {
        "root": 'nuscenes/samples',
        'train': 'Nuinterct train 3D Vision Grounding pkl path',
        'val': 'Nuinterct tesvalt 3D Vision Grounding pkl path',
        'max_new_tokens': 100,
    },
    'nusceneqa_planval_sampling': {
        "root": 'nuscenes/samples',
        'train': 'planning train.pkl',
        'val': 'planning_test.pkl',
        'max_new_tokens': 10,
    },
    'nusceneqa_val_all_text_2d': {
        "root": 'nuscenes/samples',
        'train': 'Nuinterct train 2D Region Description Prediction and Visual Grounding pkl path',
        'val': 'Nuinterct val 2D Region Description Prediction and Visual Grounding pkl path',
        'max_new_tokens': 500,
    },
    'nusceneqa_3dval_all_test': {
        "root": 'nuscenes/samples',
        'train': '/mnt/vol1/zhaozc_workspace/DriveMonkey/NuInteract/NuInteract/test/3D Visual Grounding.pkl',
        'val': '/mnt/vol1/zhaozc_workspace/DriveMonkey/NuInteract/NuInteract/test/3D Visual Grounding.pkl',
        'max_new_tokens': 100,
    },
    'nusceneqa_val_all_text_2d_test': {
        "root": 'nuscenes/samples',
        'train': '/mnt/vol1/zhaozc_workspace/DriveMonkey/NuInteract/NuInteract/test/2D Region Description Prediction and Visual Grounding.pkl',
        'val': '/mnt/vol1/zhaozc_workspace/DriveMonkey/NuInteract/NuInteract/test/2D Region Description Prediction and Visual Grounding.pkl',
        'max_new_tokens': 500,
    },

}

def box_xyxy_expand2square(box, w = 1600, h = 900):
    if w == h:
        return box , w, w
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box , w, w
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box , h, h

def norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box

    # Calculate the normalized coordinates with min-max clamping
    norm_x1 = max(0.0, min(x1 / w, 1.0))
    norm_y1 = max(0.0, min(y1 / h, 1.0))
    norm_x2 = max(0.0, min(x2 / w, 1.0))
    norm_y2 = max(0.0, min(y2 / h, 1.0))

    # Return the normalized box coordinates
    # normalized_box = (round(norm_x1, 3), round(norm_y1, 3), round(norm_x2, 3), round(norm_y2, 3))
    normalized_box = (norm_x1, norm_y1, norm_x2, norm_y2)
    return normalized_box

def norm_box_xywh(box, *, w, h):
    x1, y1, x2, y2 = box
    cx = (x1+x2) / 2
    cy = (y1+y2) / 2
    bw = (x2-x1)
    bh = (y2-y1)
    norm_cx = max(0.0, min(cx / w, 1.0))
    norm_cy = max(0.0, min(cy / h, 1.0))
    norm_bw = max(0.0, min(bw / w, 1.0))
    norm_bh = max(0.0, min(bh / h, 1.0))
    normalized_box_xywh = (norm_cx, norm_cy, norm_bw, norm_bh)
    return normalized_box_xywh

def flatten(boxes, is_3d = False):
    flatten_list =  [a for a in boxes]
    if not is_3d:
        normalized_boxes = []
        for box in flatten_list:
            box, new_w, new_h = box_xyxy_expand2square(box)
            normalized_boxes.append(norm_box_xyxy(box, w = new_w, h= new_h))
        return normalized_boxes
    else:
        return flatten_list

def cxcywh_flatten(boxes, is_3d = False):
    flatten_list =  [a for a in boxes]
    if not is_3d:
        normalized_boxes = []
        for box in flatten_list:
            box, new_w, new_h = box_xyxy_expand2square(box)
            normalized_boxes.append(norm_box_xywh(box, w = new_w, h= new_h))
        return normalized_boxes
    else:
        return flatten_list

def replace_except_first(input_string):
    # 初始化计数器
    count = 0

    # 定义替换函数
    def replacer(match):
        nonlocal count
        count += 1
        if count == 1:
            return match.group(0)  # 第一个匹配项不替换
        else:
            return match.group(0) + " <bbox_2d>"  # 后续匹配项替换

    # 使用re.sub进行替换
    return re.sub(r'(<CAM_\w+>)', replacer, input_string)

def replace_messy_code(input_string):
    index = []
    pattern = r'<CAM_\w+> <bbox_2d>'
    matches = list(re.finditer(pattern, input_string))
    if matches:
    # 生成一个新的字符串，开始是原始字符串
        result_string = input_string[:matches[0].end()]  # 在第一个匹配前的部分

        # 遍历匹配部分
        for i in range(1, len(matches)):
            # 获取前一个匹配和当前匹配的文本之间的部分
            start_prev = matches[i - 1].end()
            end_curr = matches[i].start()
            
            # 将匹配之间的部分替换为逗号
            result_string += ', '
            
            # 添加当前的匹配部分
            result_string += input_string[matches[i].start():matches[i].end()]
    else:
        result_string = ''
    return result_string

class NusceneDataset(torch.utils.data.Dataset):
    def __init__(self, data, root, data_lenth=None, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6, pad2square= False, 
                 num_image_token = 256,num_query_token = 0, nuscene_view = 'surrounding', seed=42):
        random.seed(seed)
        # test_data = open(data).readlines()
        test_data = pickle.load(open(data, 'rb')) if data.endswith('pkl') else json.load(open(data, 'r'))
        if data_lenth is None:
            data_lenth = 300 if 'train' in data else 500
        random.shuffle(test_data)
        test_data = test_data[:data_lenth] if data_lenth<len(test_data) else test_data

        self.data = test_data
        data_lenth = len(self.data)
        self.root = root
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.num_image_token = num_image_token
        self.num_query_token = num_query_token
        self.conv_template = model.template
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
        else:
            raise NotImplementedError(f'Not support views: {self.views}')
        
        self.num_views = len(self.camera_view)
        self.begin_str = self._get_intro_string()
        self.transform = build_transform(is_train=False, input_size=input_size, pad2square=pad2square)
        print(f'the eval data is {data} and the data lenth is {data_lenth}')

    def _get_intro_string(self):
        camera_str = ''
        for i,camera in enumerate(self.camera_view):
            if i != self.num_views-1:
                camera_str = camera_str + camera.strip()[4:].lower().replace('_',' ') + ','
            else:
                camera_str = camera_str + 'and ' + camera.strip()[4:].lower().replace('_',' ')
        # camera_view = [camera.strip()[4:].lower().replace('_',' ') for camera in self.camera_view]

        return f"The " + f"<image> "*self.num_views +\
        f"present an overview of the surrounding scene of ego vehicles, sequentially from the {camera_str} perspectives of the ego vehicle.\n "
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # data = json.loads(self.data[idx].strip())
        output = {}
        data = self.data[idx]
        conversations = data.pop('conversations')
        cams = data.pop('cams') if 'cams' in data else []
        token = data['token']
        output.update(data)

        cam_images = []
        new_cam_path = {}
        num_patches_list = []
        cam_path = data['cam_path']
        num_image = len(cam_path)
        for key in self.camera_view:
            if key in cam_path:
                if key in self.save_camera_view:
                    new_cam_path[key] = os.path.join(self.root, cam_path[key].split('samples/')[-1])
                    try:
                        image = Image.open(new_cam_path[key]).convert('RGB')
                    except:
                        print(new_cam_path[key])
                        image = Image.open(new_cam_path[key]).convert('RGB')
                else:
                    new_cam_path[key] = os.path.join(self.root, cam_path[key].split('samples/')[-1])
                    image = Image.open(new_cam_path[key]).convert('RGB')
                    image = Image.new('RGB', (image.width, image.height), (0, 0, 0))
                if self.dynamic_image_size:
                    images = dynamic_preprocess(image, max_num= max(1, self.max_num // num_image),
                                        image_size=self.input_size, use_thumbnail=self.use_thumbnail)
                    num_patches_list.append(len(images))
                    cam_images.extend(images)
                else:
                    num_patches_list.append(1)
                    cam_images.append(image)

        pixel_values = [self.transform(image) for image in cam_images]
        pixel_values = torch.stack(pixel_values)

        raw_q = conversations[0]['value']
        a = conversations[1]['value']
        output['question'] = raw_q
        output['gt'] = a
        conversations[0]['value'] = self.begin_str + raw_q
        q = self.begin_str + raw_q

        for i in range(num_image):
            image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * self.num_image_token * num_patches_list[i]}{IMG_END_TOKEN}' #image token
            q = q.replace('<image>', image_tokens, 1)

        template = get_conv_template(self.conv_template)
        template.append_message(template.roles[0], q)
        template.append_message(template.roles[1], None)
        conversation_prompt = template.get_prompt()

        bbox2d_inputs = []
        bbox2d_targets = []
        bbox3d_inputs = [] 
        bbox3d_targets = []
        ids3d_targets = []
        source = deepcopy(conversations)
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

        
        data_dict = {}
        data_dict['sources'] = deepcopy([conversations])
        data_dict['cam_path_list'] = new_cam_path
        data_dict['token'] = token
        assert len(bbox3d_targets) == len(ids3d_targets)
        data_dict.update(dict(
            bbox2d_inputs = bbox2d_inputs,
            bbox2d_targets = bbox2d_targets,
            bbox3d_inputs = bbox3d_inputs ,
            bbox3d_targets = bbox3d_targets,
            ids3d_targets = ids3d_targets,
        ))
        data_dict['pixel_values'] = pixel_values
        data_dict['num_patches_list'] = num_patches_list
        data_dict['cams'] = cams
        data_dict['conversation_prompt'] = conversation_prompt

        data_dict['output'] = output

        return data_dict

def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    num_patches_list = [_['num_patches_list'] for _ in batches]
    cam_path_list = [_['cam_path_list'] for _ in batches]
    token = [_['token'] for _ in batches]
    cams = [_['cams'] for _ in batches]
    conversation_prompt = [_['conversation_prompt'] for _ in batches]
    sources = [_['sources'] for _ in batches]
    output = [_['output'] for _ in batches]
    # input_ids= [_['input_ids'] for _ in batches]
    # labels = [_['labels'] for _ in batches]
    bbox2d_inputs = [_['bbox2d_inputs'] for _ in batches]
    bbox2d_targets = [_['bbox2d_targets'] for _ in batches]
    bbox3d_targets = [_['bbox3d_targets'] for _ in batches]
    ids3d_targets = [_['ids3d_targets'] for _ in  batches]

    return sources,bbox2d_inputs,bbox2d_targets,bbox3d_targets,ids3d_targets, \
        pixel_values,cam_path_list,token,cams,conversation_prompt,num_patches_list, output
    
class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def evaluate_chat_model():
    random.seed(args.seed)
    dataset = NusceneDataset(
        data = ds_collections[args.datasets]['val'] if args.val else ds_collections[args.datasets]['train'] , 
        root = ds_collections[args.datasets]['root'],
        data_lenth=args.data_lenth, 
        input_size=image_size, 
        dynamic_image_size=args.dynamic,
        use_thumbnail=use_thumbnail, 
        max_num=args.max_num, 
        pad2square= pad2square,
        num_image_token = num_image_token,
        num_query_token = num_query_token,
        nuscene_view = args.nuscene_view,
        seed=args.seed
    )
    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )
    
    patterns_3d = [
    r'location is.*',r'locations are.*',r'it is.*',r'position is.*',r'positions are.*',
    r'located at.*',r'locations include.*',r'at the location.*',r'located.*',r'location.*',r'locations at.*',
    r'location at.*',r'positions at.*',r'positions.*',r'locations.*',r'at.*',]

    outputs = []

    if model.template == 'Hermes-2':
        preprocess_function = preprocess_mpt
    elif model.template == 'internlm2-chat':
        preprocess_function = preprocess_internlm
    elif model.template == 'phi3-chat':
        preprocess_function = preprocess_phi3
    else:
        preprocess_function = preprocess
    
    conv_template = get_conv_template(model.template)
    generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[args.datasets]['max_new_tokens'],
                min_new_tokens=1,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
    eos_token_id = tokenizer.convert_tokens_to_ids(conv_template.sep)
    generation_config['eos_token_id'] = eos_token_id
    # for _, (pixel_values, questions, gts, raw_questions, others) in tqdm(enumerate(dataloader)):
    for _, (sources,bbox2d_inputs,bbox2d_targets,bbox3d_targets,ids3d_targets, \
    pixel_values,cam_path_list,token,cams,conversation_prompt,num_patches_list, output) in tqdm(enumerate(dataloader)):
    
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        sources = sources[0]
        bbox2d_inputs = bbox2d_inputs[0]
        bbox2d_targets = bbox2d_targets[0]
        bbox3d_targets = bbox3d_targets[0]
        ids3d_targets = ids3d_targets[0]
        # cam_path_list = cam_path_list[0]
        conversation_prompt = conversation_prompt[0]
        num_patches_list = num_patches_list[0]
        output = output[0]
        try:
            question_view = re.findall(r'<CAM_\w+>', sources[0][0]['value'])
            question_view = list(set(question_view))[0]
            question_view = question_view.replace('<','').replace('>','')
            question_view_index = dataset.camera_view.index(question_view)
        except:
            question_view_index = 0

        if len(bbox2d_targets) > 0 or len(bbox3d_targets) > 0:
            ###********************** new test*****************************##
            if len(bbox3d_targets) > 0:
                model_inputs = tokenizer(conversation_prompt, return_tensors='pt')
                input_ids = model_inputs['input_ids'].cuda()
                attention_mask = model_inputs['attention_mask'].cuda()
                
                generation_output = model.evaluate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox2d_inputs=torch.tensor([list(a) for a in bbox2d_inputs]).to(model.device,dtype= model.dtype),
                    **generation_config
                )
                response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
                response = response.split(conv_template.sep)[0].strip()
                response = response.split('\n')[0].split('away.')[0]
                output['gen'] = response
                for pattern in patterns_3d:
                    if '<embeding>' not in response:
                        response = re.sub(pattern, f"{pattern.split('.')[0]} <embeding>. ", response)
                    else:
                        break
                if '<embeding>' not in response:
                    response = re.sub(r'distance.*', "location is <embeding>. ", response)
                    if '<embeding>' not in response:
                        response = response + ' the location is <embeding>.'
                # response = re.sub(r'the location is.*', 'the location is <embeding>', response)
                output['new_gen'] = response
                sources[0][1]['value'] = response
            elif len(bbox2d_targets) > 0:
                model_inputs = tokenizer(conversation_prompt, return_tensors='pt')
                input_ids = model_inputs['input_ids'].cuda()
                attention_mask = model_inputs['attention_mask'].cuda()
                generation_output = model.evaluate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox2d_inputs=torch.tensor([list(a) for a in bbox2d_inputs]).to(model.device,dtype= model.dtype),
                    **generation_config
                )
                ###******************** raw <bbox_2d> special token ***************************##
                response = tokenizer.batch_decode(generation_output, skip_special_tokens=False)[0]
                response = response.split(conv_template.sep)[0].strip()
                output['gen'] = response
                new_response = response.replace(tokenizer.bos_token, '')
                replace_response = replace_except_first(new_response)
                new_response = replace_messy_code(replace_response)
                output['new_gen'] = new_response if len(new_response)>0 else replace_response
                sources[0][1]['value'] = new_response if len(new_response)>0 else replace_response
                
            
            # output['bbox_2d_gt'] = bbox2d_targets     ###xyxy
            output['bbox_2d_gt'] = bbox_cxcywh_to_xyxy(torch.tensor(bbox2d_targets)).tolist()  if len(bbox2d_targets)>0 else bbox2d_targets   ####xywh to xyxy
            output['bbox_3d_gt'] = bbox3d_targets
            output['bbox_3d_gt'] = [[float(data) for data in bbox3d_target] for bbox3d_target in bbox3d_targets]
            num_image_token_list = [dataset.num_image_token * num for num in num_patches_list]
            ret = preprocess_function(
                dataset.conv_template, sources=sources, tokenizer=tokenizer, num_image_token_list=num_image_token_list,
                num_query_token=dataset.num_query_token,num_image = dataset.num_views
            )
            input_ids=ret['input_ids'].cuda()
            labels=ret['labels'].cuda()
            attention_mask=input_ids.ne(tokenizer.pad_token_id)
            
            pred_2d_bbox, pred_3d_bbox, cls_3d_score, bbox_pred_3d = model.inference(
            # pred_2d_bbox, pred_3d_bbox, cls_2d_score, bbox_pred_2d = model.inference(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                cam_path_list=cam_path_list,
                cams=cams,
                token = token,
                bbox2d_inputs = torch.tensor([list(a) for a in bbox2d_inputs]).to(model.device,dtype= model.dtype),
                bbox3d_targets = [torch.tensor(bbox3d_targets) if bbox3d_targets else torch.zeros(0,9)], 
                ids3d_targets = [torch.tensor(ids3d_targets)],
                new_bbox2d_targets= [torch.tensor(bbox2d_targets) if bbox2d_targets else torch.zeros(0,4)],
                camera_index = torch.tensor([question_view_index])
            )
            ##********************** general *****************************##
            output['pred_2d_bbox'] = bbox_cxcywh_to_xyxy(pred_2d_bbox).tolist() if pred_2d_bbox is not None else []
            # output['pred_2d_bbox'] = pred_2d_bbox.tolist() if pred_2d_bbox is not None else []
            output['pred_3d_bbox'] = pred_3d_bbox.tolist() if pred_3d_bbox is not None else []
            output['cls_3d_score'] = cls_3d_score.tolist() if cls_3d_score is not None else []
            output['bbox_pred_3d'] = bbox_pred_3d.tolist() if bbox_pred_3d is not None else []
        else:
            model_inputs = tokenizer(conversation_prompt, return_tensors='pt')
            input_ids = model_inputs['input_ids'].cuda()
            attention_mask = model_inputs['attention_mask'].cuda()
            
            generation_output = model.evaluate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states = True,
                bbox2d_inputs=None if len(bbox2d_targets) > 0 else torch.tensor([list(a) for a in bbox2d_inputs]).to(model.device,dtype= model.dtype),
                **generation_config
            )
            
            response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
            response = response.split(conv_template.sep)[0].strip()
            output['gen'] = response

        outputs.append(output)


    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    if args.out_dir.endswith('.json'):
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
        with open(args.out_dir,'w') as f:
            json.dump(merged_outputs, f, indent=2)
    elif args.out_dir.endswith('.pkl'):
        torch.distributed.all_gather_object(merged_outputs, pickle.dumps(outputs))

        merged_outputs = [pickle.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
        with open(args.out_dir,'wb') as f:
            pickle.dump(merged_outputs, f)
    else:
        raise ImportError

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str,
                        default='nusceneqa_val')
    parser.add_argument('--data_lenth', type=int, default=None)
    parser.add_argument('--nuscene_view', type=str, default='surrounding')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--val', action='store_true')
    args = parser.parse_args()

    # if not os.path.exists(args.out_dir):
    #     os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')[0]
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    if args.auto:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    kwargs = {'device_map': 'auto'} if args.auto else {}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, **kwargs).eval()
    if not args.load_in_8bit and not args.auto:
        model = model.cuda()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    pad2square = model.config.pad2square
    num_image_token = model.num_image_token
    num_query_token = model.embedding_num
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)  #模型的image token id
    model.bbox_2d_token_id = tokenizer.convert_tokens_to_ids(BBOX_2D_TOKEN) 
    model.bbox_2d_input_token_id = tokenizer.convert_tokens_to_ids(BBOX_2D_INPUT_TOKEN)
    model.bbox_3d_token_id = tokenizer.convert_tokens_to_ids(BBOX_3D_TOKEN) 
    model.bev_query_token_id = tokenizer.convert_tokens_to_ids(BEV_QUERY_TOKEN) 

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    

    evaluate_chat_model()