import os
from extract_features_gpu import run
from pathlib import Path


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


if __name__ == '__main__':
    # get all the subjects folder in the MD2K dataset folder
    subjects = get_immediate_subdirectories("../../../dataset/Sense2StopSync/flow_pwc")

    for subject in subjects:
        # file structure: '{flo_folder}/{video_folder}/flow_x.jpg'
        flo_folder = "../../../dataset/Sense2StopSync/flow_pwc/" + subject
        # features will be saved as 'output_dir/{video_name}-{mode}.npz'
        output_dir = "../../../data/Sense2StopSync/vid_feat/" + subject
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
