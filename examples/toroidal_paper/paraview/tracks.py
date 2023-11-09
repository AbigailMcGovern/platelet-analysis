import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import ndimage
from pyevtk.hl import imageToVTK, gridToVTK, pointsToVTK, polyLinesToVTK
from nd2reader import ND2Reader
import json
import pyevtk
from scipy import ndimage
import os
from scipy.spatial.transform import Rotation as Rot
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def vtk_compat(m):
    return np.ascontiguousarray(m.astype('float32'))

def vtk_compat(m):
    return np.ascontiguousarray(m.astype('float32').squeeze())

def make_path(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def get_files(path, pattern):
    p = Path(path).rglob(pattern)
    files = sorted([str(x).replace('\\', '/') for x in p if (x.is_file() & (x.stat().st_size!=0))])
    return files
