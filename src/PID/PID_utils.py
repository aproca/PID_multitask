import os
import json
from pathlib import Path
import numpy as np
import math

def load_dict(PID_file):
    with open(PID_file, 'r') as fp:
        model_PID = json.load(fp)
    return model_PID

def check_dict_exists(PID_file):
    if Path(PID_file).exists():
        return load_dict(PID_file)
    else:
        return {}

def save_dict(model_PID, PID_file, PID_folder):
    if not os.path.exists(PID_folder):
        os.makedirs(PID_folder)
    with open(PID_file + '.json', 'w') as fp:
        json.dump(model_PID, fp)

def is_real_number(measure):
    if np.isnan(measure) or math.isinf(measure):
        return 0
    elif measure < 0:
        return 0
    else:
        return measure