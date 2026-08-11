

<div align="center">
<h3>Extending Large Vision-Language Model for <br>Diverse Interactive Tasks in Autonomous Driving</h3>

Zongchuang Zhao<sup>1\*</sup>, Haoyu Fu<sup>1\*</sup>, Dingkang Liang<sup>1†</sup>, Xin Zhou<sup>1</sup>, Dingyuan Zhang<sup>1</sup>, Hongwei Xie<sup>2</sup>,  Bing Wang<sup>2</sup>, Xiang Bai<sup>1</sup>

<sup>1</sup>  Huazhong University of Science & Technology, <sup>2</sup>  Xiaomi EV 

(*) Equal contribution. (†) Project leader.

<a href="https://arxiv.org/abs/2505.08725"><img src='https://img.shields.io/badge/arXiv-DriveMonkey-red' alt='Paper PDF'></a>
<a href="https://github.com/zc-zhao/DriveMonkey"><img src='https://img.shields.io/badge/Project_Page-DriveMonkey-green' alt='Project Page'></a>
<a href="https://huggingface.co/zczhao/DriveMonkey/tree/main"><img src='https://img.shields.io/badge/Model_Weight-🤗 Hugging Face-yellow' alt='Model Weight'></a>
</div> 

# DriveMonkey

## News

`[2026/03/20]` DriveMonkey code and dataset are now released!

## NuInteract Dataset
NuInteract is constructed based on [nuScenes](https://github.com/nutonomy/nuscenes-devkit). It encompasses 239K images (six single-view images and one surrounding view image for each frame) with high-quality dense captions and 1.3M data across diverse interactive language-based tasks, resulting in a total of 1.5M image-text pairs.

<div align="center">
<img src="assets/images/data_pipeline.jpg" width="1000">
</div>

### Dense Caption

We collect various objects and their information from different experts, including the Ground Truth of [nuScenes](https://github.com/nutonomy/nuscenes-devkit), [GRiT](https://github.com/JialianW/GRiT) for extracting bounding boxes and corresponding object descriptions, and [SAM](https://github.com/facebookresearch/segment-anything) segmenting and identifying objects followed by describing them using [BLIP](https://github.com/salesforce/BLIP). We then filter them using Intersection over Union (IoU) and Image-Text Matching (ITM) criteria. The filtered objects and the corresponding information are then input into the [Gemini](https://ai.google.dev/gemini-api/) to generate dense captions.

1. Please download the dense caption of NuInteract dataset [cap_public.tar.gz](https://github.com/zc-zhao/DriveMonkey/releases/download/NuInteract_Dataset/cap_public.tar.gz) and decompress it.

```
tar -zxvf cap_public.tar.gz
```
After decompressing it, the file tree is as follows:
```
all_caption_public/
├── 0a0d1f7700da446580874d7d1e9fce51.json
├── 0a1b4e0aa3824b0a96bafae7105c58cc.json
├── ...
├── token_name.json
└── fffce4445c964803a12a2d64023fde40.json
```

2. Use the scripts [load_dense_caption.py](tools/load_dense_caption.py) to load dense caption and convert the dense caption to [InternVL](https://github.com/OpenGVLab/InternVL) data format. Note that set the file path to your own.
```
python tools/load_dense_caption.py
```

### Other Diverse Tasks
We use predefined templates combined with object information to create data for diverse interactive driving tasks, including 2D region description, 2D visual grounding, prediction, planning, and 3D visual grounding.

Please download the dataset [NuInteract.zip](https://github.com/zc-zhao/DriveMonkey/releases/download/NuInteract_Dataset/NuInteract.zip) and decompress it.
```
unzip NuInteract.zip
```
After decompressing it, the file tree is as follows:
```
NuInteract/
├── train/
│ ├── 2D Visual Grounding.pkl
│ ├── Region Description and Prediction.pkl
│ ├── planning.pkl
│ └── 3D Visual Grounding.pkl
├── test/
│ ├── 2D Region Description Prediction and Visual Grounding.pkl
│ ├── planning_test.pkl
│ └── 3D Visual Grounding.pkl
```
All files follow the conversation format of [InternVL](https://github.com/OpenGVLab/InternVL).

## Getting Started
```
git clone https://github.com/zc-zhao/DriveMonkey.git
cd ./DriveMonkey
conda create -n DriveMonkey python=3.9
conda activate DriveMonkey
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install mmdet==2.24.0 mmdet3d==1.0.0rc4 mmsegmentation==0.20.0
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install networkx==3.2.1
pip install flash-attn==2.3.6 --no-build-isolation
```
For evaluation, you should install `language_evaluation` following the [language-evaluation](https://github.com/bckim92/language-evaluation).

## Training
Following the InternVL ([code](https://github.com/OpenGVLab/InternVL)), you should set the data path in the JSON file `internvl_chat/shell/internvl_nuscene_pretrain.json` and `internvl_chat/shell/internvl_nuscene_finetune_mixdataset_vgtextx3_rgtextx1_3dx3_capx1_planx1.json`.
```
{
    "nuscene_cap_194k": {
      "root": "./nuscenes/samples",
      "annotation": "your cap json path",
      "data_augment": false,
      "repeat_time": 1,
      "length": "data_length"
    }
}
```
Note that set the `--model_name_or_path` in the shell file.
### Dense caption pretrain
```
bash ./shell_nuscene/internvl2.0/internvl2_2b_internlm2_1_8b_nuscene/internvl2_2b_internlm2_1_8b_dynamic_res_nuscene_pre_loadofficialweight.sh 1 8 pretrain_output_dir
```
### Multi-Task Finetune
```
bash ./shell_nuscene/internvl2.0/internvl2_1b_qwen2_0_5b_nuscene/internvl2_1b_qwen2_0_5b_dynamic_res_nuscene_finetune_vgtextx3_rgtextx1_3dx3_capx1_planx1_loadofficialweight.sh 1 8 output_dir
```

## Evaluation
Note that set the `val path` in the `internvl_chat/eval/nuscene/evaluate_nuscene_bev.py`.

```
GPUS=8 bash evaluate_nuscene_bev.sh \
your_checkpoint_path \
val_task_type \
--val \
--out-dir output.pkl
```
For 2D task metric, as follows:
```
python ./eval/nuscene/calculate_metric/evaluate_2d_text_usecoco.py
```

For 3D task metric, as follows:
```
python ./eval/nuscene/calculate_metric/evaluate_map.py

python ./eval/nuscene/calculate_metric/evaluate_pr3d.py
```
## Model
Download our model checkpoint from [Hugging Face](https://huggingface.co/zczhao/DriveMonkey/tree/main)


## Acknowledgement

This project is based on InternVL ([code](https://github.com/OpenGVLab/InternVL)), nuScenes ([homepage](https://www.nuscenes.org/), [code](https://github.com/nutonomy/nuscenes-devkit)). Thanks for their wonderful works.

## Citation
If this work is helpful for your research, please consider citing:

```
@article{zhao2025extending,
  title={Extending Large Vision-Language Model for Diverse Interactive Tasks in Autonomous Driving},
  author={Zhao, Zongchuang and Fu, Haoyu and Liang, Dingkang and Zhou, Xin and Zhang, Dingyuan and Xie, Hongwei and Wang, Bing and Bai, Xiang},
  journal={arXiv preprint arXiv:2505.08725},
  year={2025}
}
```
