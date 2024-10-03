import os
import argparse
import re
import time
import openai
import openai.error
from dotenv import load_dotenv

def get_answer(args):
    load_dotenv(verbose=True)
    openai.api_key = os.getenv('API_KEY')

    #################
    file_name = os.path.basename(args.label_file)
    dataset_name = file_name.replace('_classes.txt', '')

    labels = []
    with open(args.label_file, 'r') as lines:
        for line in lines:
            labels.append(line.strip())
    
    answer_file = os.path.join(args.answers_folder, f'{dataset_name}_spatio-temporal_{args.descriptor_type}_concepts2.txt')
    ##############
    for i, label in enumerate(labels):
        if args.descriptor_type == 'spatio':
            # qs = f'Please give me a long list of descriptors for action scene: {label}, 4 descriptors in total.'
            # qs = f'Please give me the descriptors of key objects and actors in this action scene: {label}, 4 descriptors in total.'
            # qs = f'For the action {label}, please provide four spatial descriptors that describe the key elements associated with this action in a scene.'
            # qs = f'Spatial Descriptors are intended to capture static visual elements that can be discerned from a single image—such as settings and common objects. Give me the four spatial descriptors for the action: "{label}".'
            # qs = f'List the most important features for recognizing something as a {label}:' # label-free qs1
            # qs = f'List the things most commonly seen around a {label}:' # label-free qs2
            # qs = f'Give superclasses for the word {label}:' # label-free qs3
            # qs = f'Give a concise one-sentence description that explains "{label}" clearly and accurately.' # spatio-temporal 1
            qs = f'Give a concise one-sentence description that explains the human action clearly and accurately. In a sentence without a subject. Here are the action: {label}' # spatio-temporal 2
        elif args.descriptor_type == 'temporal':
            # qs = f'Please give me a long list of decompositions of steps for this action: {label}, 4 steps in total.'
            # qs = f'Please give me the descriptors for duration of each key movement or phase in this action: {label}, 4 descriptors in total.'
            # qs = f'For the action {label}, please provide four temporal descriptors that describe the sequential steps involved in performing this action in a video.'
            qs = f'For the action {label}, please provide four temporal descriptors about decompositions of this action involved in a video.'
        # if i > 105:
        print(qs)
        try:
            chat_completion = openai.ChatCompletion.create(
                messages=[
                    {
                        # "role": "assistant",
                        # "content": "Spatio Descriptors are intended to capture static visual elements that can be discerned from a single image—such as settings and common objects.",
                        "role": "user",
                        # "content": 'Please give me a long list of descriptors for action: {label}, 4 descriptors in total.',
                        # spatio
                        "content": qs,
                        # "content": "Please give me a long list of descriptors for action: air drumming, 4 descriptors in total, what concrete spatial features might a video model detect?"
                        # "content": "Please give me a long list of descriptors for action air drumming, total 4 key spatial descriptors that a video model might detect."
                        # temporal
                        # "content": "Describe the temporal dynamics of the action '[action]' in the video. Focus on the duration, sequence of movements, and changes over time."
                    }
                ],
                model="gpt-4o-mini",
                # model="gpt-3.5-turbo"
            )
            print(f"{i}:{label}")
            # extracted_text = re.findall(r'\*\*(.*?)\*\*', chat_completion["choices"][0]["message"]["content"])
            line = chat_completion["choices"][0]["message"]["content"]
            with open(answer_file, 'a') as file:
                # for line in extracted_text:
                    # file.write(line + '\n')
                file.write(line + '\n')
                    
            time.sleep(2)
        except openai.error.RateLimitError as e:
            print(f"{label} : Rate limit exceeded. Waiting for 60 seconds before retrying...")
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--query", type=str, default="what is this?")
    parser.add_argument("--label_file", type=str, default="/data/psh68380/repos/Video-CBM_/data/kinetics400_classes.txt")
    parser.add_argument("--answers_folder", type=str, default="/data/psh68380/repos/LLaVA/GPT_result")
    parser.add_argument("--N_s", type=int, default=4, choices=[2, 4, 8])
    parser.add_argument("--descriptor_type", choices=['spatio', 'temporal'], type=str, default="spatio")
    args = parser.parse_args()

    get_answer(args)