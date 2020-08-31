# import subprocess

from extract_features_gpu import run
from pathlib import Path

# def extract_frames(video,output):
#     # command = "ffmpeg -i {video} -ac 1  -f flac -vn {output}".format(video=video, output=output)
#     command = "ffmpeg -i {video} vid1/{output}/img_%05d.jpg".format(video=video, output=output)
#     subprocess.call(command,shell=True)


if __name__ == '__main__':

    for i in range(1, 6):
        # file structure: '{flo_folder}/{video_folder}/flow_x.jpg'
        flo_folder = "../../../dataset/ECM/ecm" + str(i) + "/"
        # features will be saved as '{output_dir}/{video_name}-{mode}.npz'
        output_dir = "../../../data/ECM/ecm" + str(i) + "_vid_feat/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Extract video feature for video folders in the 'input_dir', and save as 'output_dir/{video_name}-{mode}.npz'.
        # Either optical flow data or rgb data that are in folder 'input_dir/video_folder/'.
        run(mode="flow",
            # load_model="models/flow_charades.pt",
            load_model="models/flow_imagenet.pt",
            sample_mode="resize",
            frequency=1,
            input_dir=flo_folder,
            output_dir=output_dir,
            batch_size=16,
            usezip=0)
