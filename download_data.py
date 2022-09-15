#!/usr/bin/env python

import tarfile
import urllib.request
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
PTB_DIR = DATA_DIR / "ptb"
PTB_DIR.mkdir(exist_ok=True)
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)
CIFAR_DIR = DATA_DIR / "cifar-10-batches-py"

PTB_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb."
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# Download Penn Treebank dataset
for f in ["train.txt", "test.txt", "valid.txt"]:
    fpath = PTB_DIR / f
    if not fpath.exists():
        urllib.request.urlretrieve(PTB_URL + f, fpath)

# Download CIFAR-10 dataset
if not CIFAR_DIR.is_dir():
    tar_path = DATA_DIR / "cifar-10-python.tar.gz"
    urllib.request.urlretrieve(CIFAR_URL, tar_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(DATA_DIR)

CAMERAMAN_URL = "https://www.math.hkust.edu.hk/~masyleung/Teaching/CAS/MATLAB/image/images/cameraman.jpg"
urllib.request.urlretrieve(CAMERAMAN_URL, IMAGES_DIR / "cameraman.jpg")
