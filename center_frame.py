import ffmpeg
import os
import pandas as pd
import argparse

def extract_center_frame(video_path, output_folder, frame_num):
    # Get video duration using ffmpeg
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    duration = float(video_info['duration'])
    
    # Calculate the middle time point (in seconds)
    interval  = duration / (frame_num + 1)
    
    for i in range(1, frame_num + 1):
        timestamp = interval * i
        output_image_path = f'{output_folder}/{os.path.splitext(os.path.basename(video_path))[0]}_{i}.jpg'
        
        # Use ffmpeg to extract the frame at the specific timestamp
        (
            ffmpeg
            .input(video_path, ss=timestamp)  # ss is the timestamp to seek to
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
    anno_file = '/data/psh68380/repos/Video-CBM_/data/video_annotation/kinetics100/train.csv'
    labels_file = '/data/psh68380/repos/Video-CBM_/data/kinetics100_classes.csv'
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
        video_path = row.video_path
        name = row.name # k400
        # name = classes[label] # txt class file
        extract_center_frame(video_path, output_folder, args.frame_num)

        if previous_name != name:
            print(f'{previous_name} is all saved as {args.frame_num} frame.')
        previous_name = name   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_num", type=int, default=1, help="Number of frames you want to extract")
    parser.add_argument("--frame_save_folder", type=str, default="", help="Number of frames you want to extract")
    args = parser.parse_args()
    main(args)
