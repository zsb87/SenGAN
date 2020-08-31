import os
import pytz


settings = {}
settings["TIMEZONE"] = pytz.timezone("America/Chicago")
settings["FPS"] = 29.969664
settings["FRAME_INTERVAL"] = "0.03336707S"
settings["starttime_file"] = "../../dataset/Sense2StopSync/start_time.csv"
settings["starttime_train_file"] = "../../dataset/Sense2StopSync/start_time_train.csv"
settings["starttime_test_file"] = "../../dataset/Sense2StopSync/start_time_test.csv"
settings["starttime_val_file"] = "../../dataset/Sense2StopSync/start_time_val.csv"

RAW_DIR = os.path.join(os.path.dirname(__file__), "../../dataset/Sense2StopSync/")
settings["RAW_DIR"] = RAW_DIR
settings["sensor_path"] = os.path.join(RAW_DIR, "SENSOR/")
settings["reliability_resample_path"] = os.path.join(RAW_DIR, "RESAMPLE/")
settings["flow_path"] = os.path.join(RAW_DIR, "flow_pwc/")

TEMP_DIR = os.path.join(os.path.dirname(__file__), "../../data/Sense2StopSync")
settings["TEMP_DIR"] = TEMP_DIR
settings["vid_feat_path"] = os.path.join(TEMP_DIR, "vid_feat")

settings["qualified_window_num"] = 10
settings["window_size_sec"] = 10 
settings["window_criterion"] = 0.8
settings["stride_sec"] = 1
settings["video_max_len"] = (17*60+43)*1000
settings["val_set_ratio"] = 0.2
settings["sample_counts"] = 7
