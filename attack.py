import numpy as np
from PIL import Image
import os
import json
import random
import argparse
from const import *
from utils import (get_jailbreak_score, query_claude, query_base, list_dir)
from eval import eval


dataformat2prompt = {
    'images_wr': wr_game_prompt,
    'images_mirror': mirror_game_prompt,
    'images_base64': base64_game_prompt,
    'images_rotate': rotate_game_prompt,
}

idx2defense = ['before', 'middle', 'after']
def random_shuffle_sentence(sentence):
    ssp = sentence.split()
    random.shuffle(ssp)
    return ssp

def query(model_id, image_path, prompt, api_key):
    if "gpt" in model_id or 'qwen' in model_id:
        response = query_base(model_id, image_path, prompt, api_key)
    elif model_id == 'claude':
        response = query_claude(image_path, prompt, api_key)
    else:
        raise ValueError(f"no matched model named {model_id}")
    return response
    
    
def main():
    parser = argparse.ArgumentParser()
    api_key = os.environ["OPENAI_API_KEY"] # modify your key here
    parser.add_argument('--save-dir', type=str, default="./save_dir/")
    parser.add_argument('--dataset', type=str, choices=['safebench', 'mm-safebench', 'hades'], default='safebench')
    parser.add_argument('--target-model', type=str, choices=['gpt-4o-mini', 'gpt-4o', 'qwen-vl-max', 'claude'], default="gpt-4o")
    parser.add_argument('--image-format', type=str, default="images_mirror")

    parser.add_argument('--defense', type=int, default=None)
    parser.add_argument('--data-path', type=str, default='./dataset')
    args = parser.parse_args()
    
    data_path = os.path.join(args.data_path, args.dataset)
        
    sub_class = list_dir(data_path)
    
    for sc in sub_class:   
        if args.image_format == 'images_qr' and args.dataset == 'mm-safebench':
            image_path_format = os.path.join(data_path, sc, args.image_format, "{}.jpg")
        elif args.dataset == 'hades':
            image_path_format = os.path.join(data_path, sc, args.image_format, "{}.jpg")
        else:
            image_path_format = os.path.join(data_path, sc, args.image_format, "{}.png")

        data_info = json.load(open(os.path.join(data_path, sc, "data.json")))
        result_save_path = os.path.join(args.save_dir, args.dataset, args.image_format+ (f'_defense_{idx2defense[args.defense]}'  if args.defense is not None else ''), args.target_model)
        eval_path = result_save_path
        print(result_save_path)
        os.makedirs(result_save_path, exist_ok=True)
        result_save_path = os.path.join(result_save_path, f"{sc}_result.json")
        try:
            result = json.load(open(result_save_path))
        except Exception as e:
            result = []
        
        for idx, item in enumerate(data_info):
            print(sc, idx)
            if idx < len(result):
                continue
                
            score_list = []
            res_list = []
            if "images_wr" in args.image_format:
                question = wr_game_prompt.format(item["replace_map"], random_shuffle_sentence(item["original_prompt"]))
            elif args.image_format in dataformat2prompt:
                question = dataformat2prompt[args.image_format].format(random_shuffle_sentence(item["original_prompt"]))
            elif args.image_format == 'images' and args.dataset=='hades':
                question = item["hades_prompt"]
            elif args.image_format == 'images_qr':
                question = item["qr_prompt"]
            elif args.image_format == 'images_figstep':
                question = fs_prompt
            if args.defense==0:
                question = static_defense_pre + question
            elif args.defense==1:
                question = static_defense_m + question + static_defense_m
            elif args.defense==2:
                question = question + static_defense_sub
            print(question)
            for _ in range(5):
            
                response = query(args.target_model, image_path_format.format(f'{item["id"]}'), question, api_key)
                print(response)
                if args.image_format != 'images' and args.image_format != 'images_figstep' and args.image_format != 'images_qr':
                    answer_index = response.find("Content")
                    if answer_index == -1:
                        score_temp = 0
                    else:
                        score_temp = get_jailbreak_score(item["original_prompt"], response, api_key)        
                else:
                    score_temp = get_jailbreak_score(item["original_prompt"], response, api_key)
    
                score_list.append(score_temp)
                res_list.append(response)
                print(f"----{sc}----{item['id']}/{len(data_info)}-------{score_temp}")
                if score_temp == 5:
                    break
            score_idx = score_list.index(max(score_list))
            
            
            result.append({
                    "ori_prompt": item["original_prompt"],
                    "res": res_list[score_idx],
                    "score": score_list[score_idx],
                    "flag": item["original_prompt"] in res_list[score_idx], # check wether decrypt suscessfully
                    "query_num": len(score_list),
                })
            json.dump(result, open(result_save_path, 'w'), indent=4)
    eval(eval_path)
main()