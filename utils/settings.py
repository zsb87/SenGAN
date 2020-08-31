import os
import pytz

settings = {}
settings["TIMEZONE"] = pytz.timezone("America/Chicago")
settings["ROOT_DIR"] = os.path.join(os.path.dirname(__file__), "../data/")

#====================================================================================================
# DATA SPECIFICATION
#====================================================================================================
settings["SAMPLING_RATE"] = 20

#====================================================================================================
# DATETIME FORMAT
#====================================================================================================
settings["ABSOLUTE_TIME_FORMAT"] = "%Y-%m-%d %H:%M:%S.%f"
settings["RELATIVE_TIME_FORMAT"] = "%H:%M:%S.%f"
