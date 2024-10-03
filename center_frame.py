import ffmpeg
import os
import pandas as pd
import argparse

def extract_center_frame(video_path, output_folder, frame_num):
    # Get video duration using ffmpeg
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    duration = float(video_info['duration'])
    
    interval  = duration / (frame_num + 1)
    
    if frame_num > 1:
        for i in range(1, frame_num + 1):
            timestamp = interval * i
            output_image_path = f'{output_folder}/{os.path.splitext(os.path.basename(video_path))[0]}_{i}.jpg'
            
            (
                ffmpeg
                .input(video_path, ss=timestamp)
                .filter('scale', -1, -1)  
                .output(output_image_path, vframes=1)  # vframes=1 means only one frame is saved
                .global_args('-loglevel', 'quiet')
                .run()
            )
            if os.path.exists(output_image_path):
                # print(f"Frame {i} saved as {output_image_path}")
                pass
            else:
                print(f"Failed to extract frame {i} as {output_image_path}.")

    elif frame_num == 1:
        output_image_path = f'{output_folder}/{os.path.splitext(os.path.basename(video_path))[0]}.jpg'
        (
            ffmpeg
            .input(video_path, ss=interval)
            .filter('scale', -1, -1)  # Optional scaling if needed
            .output(output_image_path, vframes=1)  # vframes=1 means only one frame is saved
            .global_args('-loglevel', 'quiet')
            .run()
        )
        if os.path.exists(output_image_path):
            # print(f"Frame {i} saved as {output_image_path}")
            pass
        else:
            print(f"Failed to extract frame {i} as {output_image_path}.")

def main(args):
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

    output_folder = args.frame_save_folder

    df_videos = pd.read_csv(anno_file, header=None, names=['video_path', 'label'])
    df_labels = pd.read_csv(labels_file)
    # txt class file
    # f = open(labels_file, 'r')
    # classes = [line.strip() for line in f.readlines()]

    df_videos['name'] = df_videos['label'].map(df_labels.set_index('id')['name']) # txt class file 이면 주석처리
    df_videos = df_videos.sort_values(by='label', ascending=True)

    previous_name = None
    for row in df_videos.itertuples(index=False):
        if args.dataset == 'k100' or args.dataset == 'k400':
            video_path = row.video_path # kinetics
        elif args.dataset == 'ssv2':
            video_path = f'/local_datasets/something-something/something-something-v2-mp4/{row.video_path}' # ssv2
        name = row.name
        # name = classes[label] # txt class file
        extract_center_frame(video_path, output_folder, args.frame_num)

        if previous_name != name:
            print(f'{previous_name} is all saved as {args.frame_num} frame.')
        previous_name = name   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=['k100', 'k400', 'ssv2'], type=str, default="k100")
    parser.add_argument("--frame_num", type=int, default=1, help="Number of frames you want to extract")
    parser.add_argument("--frame_save_folder", type=str, default="", help="Number of frames you want to extract")
    args = parser.parse_args()
    main(args)
