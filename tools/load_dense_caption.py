import json
import os
import random
import re
import numpy as np
from tqdm import tqdm

cap_path = 'all_caption_public' # set the path to the dense caption list path

SURROUNDING_CAPTION_QUESTIONS = [
    'Describe in detail the road scene around the ego vehicle using the provided images.',
    'Using the provided images, describe thoroughly the road scene surrounding the ego vehicle.',
    'Offer a comprehensive depiction of the road scene encompassing the ego vehicle, utilizing the provided images.',
    'Can you provide a detailed description of the road scene surrounding the ego vehicle based on the provided images?',
    'How would you describe the road scene around the ego vehicle using the images provided?',
    'What does the road scene around the ego vehicle look like based on the images provided?',
    'Could you describe the road scene surrounding the ego vehicle in detail with the help of the provided images?',
    'What details can you offer about the road scene surrounding the ego vehicle utilizing the images provided?',
    'Utilizing the provided images, how would you describe the road scene surrounding the ego vehicle?',
    'What is your detailed description of the road scene around the ego vehicle using the provided images?']

SINGLE_CAPTION_QUESTIONS = [
    "Please elaborate on the scene of the ego car's <direction> based on the provided <cam> image.",
    "Describe in detail the ego car's <direction> scene using the provided <cam> image.",
    "Based on the <cam> image provided, please elaborate on the scene of the ego car's <direction>.",
    "By examining the provided <cam> image, describe the ego car's <direction> scene thoroughly.",
    "The ego car's <direction> scene can be accurately depicted by observing the provided <cam> image.",
    "Utilizing the provided <cam> image, delve into the details of the ego car's <direction> scene.",
    "The scene of the ego car's <direction> can be precisely outlined based on the <cam> image provided.",
    "The details of the ego car's <direction> scene can be inferred from the provided <cam> image.",
    "Can you describe the ego car's <direction> scene based on the provided <cam> image?",
    "Would you be able to detail the ego car's <direction> scene by examining the provided <cam> image?",
    "Could you provide a detailed description of the ego car's <direction> scene from the provided <cam> image?",
    "Can you please elaborate on the scene of the ego car's <direction> based on the provided <cam> image?",
    "Would you be able to accurately depict the ego car's <direction> scene by analyzing the provided <cam> image?",
    "Could you provide a comprehensive description of the ego car's <direction> scene from the provided <cam> image?",
    "Can you infer the details of the ego car's <direction> scene from the provided <cam> image?",
]


def collect_cap(cap_path, save_path):
    '''
    collect all the dense caption json files in the given path
    cap_path: the path to the dense caption json files
    '''
    cap_json_list = os.listdir(cap_path)
    cap_json_list = [os.path.join(cap_path, i) for i in cap_json_list if i.endswith('.json')]
    
    '''
    each json file contains a list of dicts
    each dict contains:
        'token': sample token of the current frame
        'prev': sample token of the previous frame
        'next': sample token of the next frame
        'frame_idx': the index of the current frame in the scene
        'scene_token': the token of the scene
        'cam_path': the camera path of the current frame
        'gpt_prompt': the prompt fed into Gemini of the current frame. It contains six camera views and the front and back views of the current frame
        'gemini_caption': the generated dense caption of Gemini in the current frame, corresponding to GPT prompt
    '''
    cap_data = []
    for cap_json in cap_json_list:
        cap = json.load(open(os.path.join(cap_path, cap_json), 'r'))
        cap_data.extend(cap)

    print(f'Loaded {len(cap_data)} samples from {len(cap_json_list)} json files.')

    with open(save_path, 'w') as f:
        json.dump(cap_data, f, indent=4)
    

camera_view = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT','OVERALL']

def convert_cap_to_internvl_format(cap_paths, save_path):
    '''
    convert the dense caption json files to llava format
    cap_paths: the path to the dense caption json files
    save_path: the path to save the converted json files
    '''
    all_cap_data = json.load(open(cap_paths, 'r'))
    all_cap = []
    for cap_info in tqdm(all_cap_data):
        captions = cap_info.pop('gemini_caption', {})
        cap_info.pop('gpt_prompt', None)
        for view in camera_view:
            q = ''
            cap = ''

            if view == 'OVERALL':
                if captions.get('FRONT') and captions.get('BACK'):
                    q = random.choice(SURROUNDING_CAPTION_QUESTIONS).strip().lower()
                    cap = captions['FRONT'] + captions['BACK']
            else:
                if captions.get(view):
                    q = random.choice(SINGLE_CAPTION_QUESTIONS).strip().lower()
                    q = q.replace('<direction>', view.strip()[4:].lower().replace('_',' '))
                    q = q.replace('<cam>','<'+view+'>')
                    cap = captions[view]

            if q and cap:
                conv = [{'from':'human','value':q},{'from':'gpt','value':cap}]
                new_info = {'conversations': conv}
                new_info = {**cap_info,**new_info}
                all_cap.append(new_info)
            else:
                print(cap_info['token'])
    with open(save_path,'w') as f:
        json.dump(all_cap, f, indent=2)

if __name__ == '__main__':
    collect_cap(cap_path, 'all_caption.json')
    # convert_cap_to_llava_format('all_caption.json', 'llava_caption.json')
    convert_cap_to_internvl_format(cap_path, 'llava_caption.json')