import argparse
import torch
import os
import pandas as pd

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_path.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_file):
    out = []
    image = load_image(image_file)
    out.append(image)
    return out

#### for kinetics ####
def convert_to_jpg(video_path):
    # return f"{os.path.splitext(os.path.basename(video_path))[0]}.jpg"
    return f"{os.path.splitext(os.path.basename(video_path))[0]}"

def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    if args.dataset == "k100":
        anno_file = '/data/psh68380/repos/Video-CBM_/data/video_annotation/kinetics100/train.csv'
        labels_file = '/data/psh68380/repos/Video-CBM_/data/kinetics100_classes.csv'
    elif args.dataset == "k400":
        anno_file = '/data/psh68380/repos/Video-CBM_/data/video_annotation/kinetics400/train.csv'
        labels_file = '/data/psh68380/repos/Video-CBM_/data/kinetics400_classes.csv'
    elif args.dataset == "ssv2":
        anno_file = '/data/psh68380/repos/Video-CBM_/data/video_annotation/SSV2/train.csv'
        labels_file = '/data/psh68380/repos/Video-CBM_/data/ssv2_classes.csv'
    else: 
        print("It is wrong dataset.")
    
    file_name = os.path.basename(labels_file)
    dataset_name = file_name.replace('_classes.csv', '')

    df_videos = pd.read_csv(anno_file, header=None, names=['video_path', 'label'])
    df_labels = pd.read_csv(labels_file)

    df_videos['name'] = df_videos['label'].map(df_labels.set_index('id')['name'])
    df_videos = df_videos.sort_values(by='label', ascending=True)
    # df_videos = df_videos.groupby('label').apply(lambda x: x.sample(n=400, random_state=42)).reset_index(drop=True) # label 별 1000장 선택
    # df_videos['image_file'] = df_videos['video_path'].apply(convert_to_jpg)
    df_videos['image_name'] = df_videos['video_path'].apply(convert_to_jpg)
    ######################

    image_path = args.image_path
    image_names = [name for name in os.listdir(image_path) if name.endswith('.jpg')]
    count = 0
    for row in df_videos.itertuples(index=False):
        image_name = row.image_name
        matching_images = [img for img in image_names if img.startswith(image_name)]
        
        if matching_images:
            label = row.name
            for image_name in matching_images:
                answer_file = os.path.join(args.answer_folder, f'{dataset_name}_ost_{args.descriptor_type}_concepts_{label}.txt')
                if args.descriptor_type == 'spatio':
                    # qs = f'Here is an image of someone performing {label}; Can you give me a list of spatial descriptors for this image?, {args.N_s} descriptors in total.' # tem = 0.2
                    # qs = f'Please give me four object in the form of a single word or short phrases.' # k400_spatio_descriptors/
                    qs = f'Please give me four objects in this image, as single words.' # k100_temporal_descriptors/ # k400_spatio_center_frame
                elif args.descriptor_type == 'temporal':
                    qs = f'This image represents the action {label}. Please divide the action into 2 distinct action in the form of a single word or short phrases.'

                image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                if IMAGE_PLACEHOLDER in qs:
                    if model.config.mm_use_im_start_end:
                        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                    else:
                        qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
                else:
                    if model.config.mm_use_im_start_end:
                        qs = image_token_se + "\n" + qs
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

                if "llama-2" in model_name.lower():
                    conv_mode = "llava_llama_2"
                elif "mistral" in model_name.lower():
                    conv_mode = "mistral_instruct"
                elif "v1.6-34b" in model_name.lower():
                    conv_mode = "chatml_direct"
                elif "v1" in model_name.lower():
                    conv_mode = "llava_v1"
                elif "mpt" in model_name.lower():
                    conv_mode = "mpt"
                else:
                    conv_mode = "llava_v0"

                if args.conv_mode is not None and conv_mode != args.conv_mode:
                    print(
                        "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                            conv_mode, args.conv_mode, args.conv_mode
                        )
                    )
                else:
                    args.conv_mode = conv_mode

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

            #######################
                
                image_file = os.path.join(image_path, image_name)
                images = load_images(image_file)
                image_sizes = [x.size for x in images]
                try:
                    images_tensor = process_images(
                        images,
                        image_processor,
                        model.config
                    ).to(model.device, dtype=torch.float16)
                except ValueError as e:
                    print(f"Skipping image {image_name} due to error: {e}")
                    continue

                input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .cuda()
                )

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=images_tensor,
                        image_sizes=image_sizes,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                # print(outputs)
                count += 1
                if count % 500 == 0:
                    print(f'Finished answering {count} / {len(image_names)}')

                with open(answer_file, 'a') as f:
                    f.write(f"{outputs}\n")
        else:
            print(f"Image File: {image_name} not found in image_names.")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=['k100', 'k400', 'ssv2'], type=str, default="k100")
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_path", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--query", type=str)
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--answer_folder", type=str, default="/data/psh68380/repos/LLaVA/center-frame_LLava_result")
    parser.add_argument("--N_s", type=int, default=2)
    parser.add_argument("--descriptor_type", choices=['spatio', 'temporal'], type=str, default="spatio")
    
    args = parser.parse_args()

    eval_model(args)