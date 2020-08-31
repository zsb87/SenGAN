import os
import pathlib
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from S2S_load_sensor_data import read_data_datefolder_hourfile
from S2S_settings import settings


FPS = settings["FPS"]
FRAME_INTERVAL = settings["FRAME_INTERVAL"]
sample_counts = settings["sample_counts"]

def load_start_time(start_time_file, vid):
    """
    load start time

    Args:
        start_time_file: str
        vid: str, video

    Returns:
        int, start time

    """
    df_start_time = pd.read_csv(start_time_file).set_index("video_name")
    if vid not in df_start_time.index:
        print("Error: ", vid, " not in ", start_time_file)
        exit()
    start_time = df_start_time.loc[vid]["start_time"]
    return int(start_time)


def reliability_df_to_consecutive_seconds(
    df_sensor_rel, window_size_sec, stride_sec, threshold=sample_counts
):
    """
    Convert from reliability df to consecutive seconds represented with start and end time.

    Args:
        df_sensor_rel: dataframe, sensor reliability
        window_size_sec:, int, window_size
        stride_sec: int, stride
        threshold: float

    Returns:
        win_start_end: a list of all the possible [window_start, window_end] pairs.
    """
    # use the threshold criterion to select 'good' seconds
    rel_seconds = (
        df_sensor_rel[df_sensor_rel["SampleCounts"] > threshold]
        .sort_values(by="Time")["Time"]
        .values
    )
    win_start_end = consecutive_seconds(rel_seconds, window_size_sec, stride_sec)
    return win_start_end


def consecutive_seconds(rel_seconds, window_size_sec, stride_sec=1):
    """
    Return a list of all the possible [window_start, window_end] pairs 
        containing consecutive seconds of length window_size_sec inside.
    Args:
        rel_seconds: a list of qualified seconds
        window_size_sec: int
        stride_sec: int
    Returns:
        win_start_end: a list of all the possible [window_start, window_end] pairs.

    Test:
        >>> rel_seconds = [2,3,4,5,6,7,9,10,11,12,16,17,18]; window_size_sec = 3; stride_sec = 1
        >>> print(consecutive_seconds(rel_seconds, window_size_sec))
        >>> [[2, 4], [3, 5], [4, 6], [5, 7], [9, 11], [10, 12], [16, 18]]
    """
    win_start_end = []
    for i in range(0, len(rel_seconds) - window_size_sec + 1, stride_sec):
        if rel_seconds[i + window_size_sec - 1] - rel_seconds[i] == window_size_sec - 1:
            win_start_end.append([rel_seconds[i], rel_seconds[i + window_size_sec - 1]])
    return win_start_end


def load_vid_feat(vid_file, fps, start_time):
    feat = np.load(vid_file)["feature"][0]
    print("video feature shape:", feat.shape)
    frame_len = 1000.0 / fps  # duration of a frame in ms
    frames = feat.shape[0]  # number of frames
    len_ms = frames * frame_len  # duration of all frames in ms
    timestamps_int = np.arange(
        start_time,
        start_time + len_ms,
        frame_len
    ).astype(int)
    l = min(len(timestamps_int), feat.shape[0])
    timestamps_int = timestamps_int[:l]
    feat = feat[:l, :]
    df_flow = pd.DataFrame(
        data=np.hstack((timestamps_int[:,None], feat)),
        index=[i for i in range(feat.shape[0])],
        columns=["time"]+['f'+str(i) for i in range(feat.shape[1])]
    )
    df_flow["second"] = (df_flow["time"] / 1000).astype(int)
    df_flow = df_flow.reset_index()
    return df_flow, len_ms


def load_sensors_cubic(
    sensor_path, sub, device, sensors, sensor_col_headers, start_time, end_time, fps
):
    """
    load sensor data with cubic spline resampling

    Args:
        sensor_path: str,
        sub: str, subject
        device: str
        sensors: list, sensors
        sensor_col_headers: list of sensor column headers
        start_time: int
        end_time: int
        fps: float

    Returns:
        dataframe, sensor data

    """
    df_list = []
    for s, col in zip(sensors, sensor_col_headers):
        df_sensor = read_data_datefolder_hourfile(
            sensor_path, sub, device, s, start_time, end_time
        )
        df_sensor = df_sensor[["time", col]]
        df_sensor["time"] = pd.to_datetime(df_sensor["time"], unit="ms")
        df_sensor = df_sensor.set_index("time")
        df_resample = df_sensor.resample(FRAME_INTERVAL).mean()  
        # FRAME_INTERVAL as 0.03336707S is the most closest value to 1/29.969664 pandas accepts
        df_resample = df_resample.interpolate(method="spline", order=3) # cubic spline interpolation
        df_list.append(df_resample)
    df_sensors = pd.concat(df_list, axis=1)
    return df_sensors


def merge_sensor_flow(
    df_sensor,
    df_flow,
    vid_name,
    win_start_end,
    start_time,
    end_time,
    window_size_sec,
    window_criterion,
    fps
):
    """
    merge sensor flow

    Args:
        df_sensor: dataframe, sensor data
        df_flow: dataframe, flow data
        vid_name: str, video name
        win_start_end: list
        start_time: int
        end_time: int
        window_size_sec: int
        window_criterion: float
        fps: float

    Returns:
        int, count of windows
        list, a list of all dataframes of videos
        list, a list of all video data information
    """
    df_dataset_vid = []
    info_dataset_vid = []
    cnt_windows = 0
    # add an offset to each window sensor-video pair
    for pair in win_start_end:
        start = pair[0] * 1000
        end = pair[1] * 1000 + 1000
        df_window_sensor = df_sensor[
            (df_sensor["time"] >= pd.to_datetime(start, unit="ms"))
            & (df_sensor["time"] < pd.to_datetime(end, unit="ms"))
        ]
        # match video dataframe
        df_window_flow = df_flow[
            (df_flow["time"] >= pd.to_datetime(start, unit="ms"))
            & (df_flow["time"] < pd.to_datetime(end, unit="ms"))
        ]
        pd.options.mode.chained_assignment = None
        df_window_flow.loc[:, "time"] = df_window_flow.loc[:, "time"]
        df_window = pd.merge_asof(
            df_window_sensor,
            df_window_flow,
            on="time",
            tolerance=pd.Timedelta("29.969664ms"),
            direction="nearest",
        ).set_index("time")
        df_window = df_window.dropna(how="any")
        if len(df_window) > fps * window_size_sec * window_criterion:
            cnt_windows += 1
            df_dataset_vid.append(df_window)
            info_dataset_vid.append(
                [vid_name, start, end]
            )  # relatively video name, sensor starttime, sensor endtime
    return cnt_windows, df_dataset_vid, info_dataset_vid


def segment_video(
    subject,
    video,
    window_size_sec,
    stride_sec,
    window_criterion,
    starttime_file,
    fps,
):
    """
    Segment one smoking video.

    Args:
        subject: str
        video: str
        window_size_sec: int
        stride_sec: int
        window_criterion: float
        starttime_file: str
        fps: float

    Returns:
        list, a list of (video name, count of windows) pairs
        list, a list of all dataframes of videos
        list, a list of all video data information

    """
    # ==========================================================================================
    reliability_resample_path = settings['reliability_resample_path']
    sensor_path = settings['sensor_path']
    vid_feat_path = settings["vid_feat_path"]
    # ==========================================================================================

    vid_qual_win_cnt = []
    df_dataset = []
    info_dataset = []

    device = "CHEST"
    sensor = "ACCELEROMETER"
    sensors = ["ACCELEROMETER_X", "ACCELEROMETER_Y", "ACCELEROMETER_Z"]
    sensor_col_headers = ["accx", "accy", "accz"]
    vid_file = os.path.join(
        vid_feat_path, subject, "{}-flow.npz".format(video)
    )

    # load start end time
    vid_name = subject + " " + video
    start_time = load_start_time(starttime_file, vid_name)

    # load optical flow data and assign unixtime to each frame
    df_flow, len_ms = load_vid_feat(vid_file, fps, start_time)
    end_time = int(start_time) + int(len_ms)

    # load sensor reliability data
    df_sensor_rel = read_data_datefolder_hourfile(
        reliability_resample_path,
        subject,
        device,
        sensor + "_reliability",
        start_time,
        end_time,
    )

    # record consecutive seconds of the length the same as window_size
    win_start_end = reliability_df_to_consecutive_seconds(
        df_sensor_rel, window_size_sec, stride_sec, threshold=7
    )


    ## extract the optical flow frames of the good seconds according to sensor data
    df_flow["time"] = pd.to_datetime(df_flow["time"], unit="ms")
    df_flow = df_flow.set_index(
        "time"
    )

    # extract the raw data 'ACCELEROMETER_X' (,'ACCELEROMETER_Y', 'ACCELEROMETER_Z') of consecutive chunk and resample
    #   according to video frame timestamp.
    # note that we need to resample sensor data according to video time,
    #   so the input of resample function here should be raw data instead of already sampled data to avoid resample twice.

    df_sensors = load_sensors_cubic(
        sensor_path,
        subject,
        device,
        sensors,
        sensor_col_headers,
        start_time,
        end_time,
        fps,
    )

    # concatenate df_sensors and df_flow
    df_resample = pd.merge_asof(
        df_flow,
        df_sensors,
        on="time",
        tolerance=pd.Timedelta("30ms"),
        direction="nearest",
    ).set_index("time")

    df_resample = df_resample.dropna(how="any")
    # print('There are {} points left (freq=30Hz) for sensor-opticalflow combined.\n'.format(len(df_resample)))
    df_sensor = df_resample[["accx", "accy", "accz"]].reset_index()
    df_flow = df_resample.reset_index()
    # print(len(df_sensor))
    # print(len(df_flow))

    ## select anchor windows from sensor, apply shifts in videos
    cnt_windows, df_dataset_vid, info_dataset_vid = merge_sensor_flow(
        df_sensor,
        df_flow,
        vid_name,
        win_start_end,
        start_time,
        end_time,
        window_size_sec,
        window_criterion,
        fps
    )
    print(cnt_windows, "/", len(win_start_end), "windows left for video ", vid_name)
    df_dataset += df_dataset_vid
    info_dataset += info_dataset_vid
    vid_qual_win_cnt.append((vid_name, cnt_windows))

    return vid_qual_win_cnt, df_dataset, info_dataset


def segment_all(
    window_size_sec,
    stride_sec,
    window_criterion,
    usage="train",
    fps=FPS,
    load=False
):
    """
    Segment all videos

    Args:
        window_size_sec: int, window size
        stride_sec: int, stride
        window_criterion: float, window criterion
        usage: "train" or "val" or "test"
        fps: float
        load: boolean, load saved pkl file or generate new data file

    Returns:
         list, a list of all dataframes of videos
         list, a list of all video data information

    """

    assert (usage in ["train", "val", "test"])

    title_suffix = "_win{}_str{}_wincrt{}_{}".format(
        window_size_sec,
        stride_sec,
        window_criterion,
        usage
    )

    TEMP_DIR = settings["TEMP_DIR"]
    data_pkl = os.path.join(TEMP_DIR, "all_video" + title_suffix + "_df_dataset.pkl")
    info_pkl = os.path.join(TEMP_DIR, "all_video" + title_suffix + "_info_dataset.pkl")

    if load and os.path.exists(data_pkl) and os.path.exists(info_pkl):
        with open(data_pkl, "rb") as handle:
            df_dataset_all = pickle.load(handle)
        with open(info_pkl, "rb") as handle:
            info_dataset_all = pickle.load(handle)
    else:
        starttime_file = settings["starttime_" + usage + "_file"]  # select train/val/test dataset videos 
        df_start_time = pd.read_csv(starttime_file).set_index("video_name")
        video_names = df_start_time.index.tolist()

        df_dataset_all = []
        info_dataset_all = []
        vid_qual_win_cnt_all = []

        for video_name in video_names:
            subject = video_name[:3]
            video = video_name[4:]
            vid_qual_win_cnt, df_dataset, info_dataset = segment_video(
                subject=subject,
                video=video,
                window_size_sec=window_size_sec,
                stride_sec=stride_sec,
                window_criterion=window_criterion,
                starttime_file=starttime_file,
                fps=fps,
            )
            vid_qual_win_cnt_all += vid_qual_win_cnt
            df_dataset_all += df_dataset
            info_dataset_all += info_dataset

        print(
            len(vid_qual_win_cnt_all),
            "videos with valid window(s), # of qualified windows: ",
            vid_qual_win_cnt_all
        )
        
        pathlib.Path(TEMP_DIR).mkdir(parents=True, exist_ok=True) 
        pd.DataFrame(vid_qual_win_cnt_all, columns=["vid_name", "window_num"]).to_csv(
            os.path.join(TEMP_DIR, "num_valid_windows" + title_suffix + ".csv"), index=None
        )

        with open(data_pkl, "wb") as handle:
            pickle.dump(df_dataset_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(info_pkl, "wb") as handle:
            pickle.dump(info_dataset_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return df_dataset_all, info_dataset_all


if __name__ == '__main__':
    starttime_file = settings["starttime_file"]
    segment_video(
        "263", 
        "GP010351", 
        10,
        1,
        0.8,
        starttime_file,
        FPS,
    )

