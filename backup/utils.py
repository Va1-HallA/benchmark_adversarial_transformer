from glob import glob
from pathlib import Path

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

"""some part of this file is not used and need to be cleaned"""

"""GLOBAL PATHS"""
PATH_VALIDATION = Path("data/validation")
PATH_VALIDATION_IMAGES = Path("data/validation/images")
PATH_VALIDATION_LABELS = Path("data/validation/ILSVRC2012_validation_ground_truth.txt")
PATH_SYNSET_WORDS = Path("data/validation/imagenet1000_clsidx_to_labels.txt")
PATH_META = Path("data/meta.mat")
image_paths = sorted(glob(str(PATH_VALIDATION_IMAGES / "*")))  # paths of all images

"""Indices mapping from imagenet to pytorch"""
meta = scipy.io.loadmat(str(PATH_META))
original_idx_to_synset = {}
synset_to_name = {}

for i in range(1000):
    ilsvrc2012_id = int(meta["synsets"][i, 0][0][0][0])
    synset = meta["synsets"][i, 0][1][0]
    name = meta["synsets"][i, 0][2][0]
    original_idx_to_synset[ilsvrc2012_id] = synset
    synset_to_name[synset] = name

synset_to_pytorch_idx = {}
pytorch_idx_to_name = {}
with open(str(PATH_SYNSET_WORDS), "r") as f:
    for idx, line in enumerate(f):
        parts = line.split(" ")
        synset_to_pytorch_idx[parts[0]] = idx
        pytorch_idx_to_name[idx] = " ".join(parts[1:])

convert_original_idx_to_pytorch_idx = lambda idx: synset_to_pytorch_idx[original_idx_to_synset[idx]]


def model_utils(model_name):
    """create model and its utils"""
    model = create_model(model_name, pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return {"model": model, "transform": transform}


def f1_score(y_pred, y_true):
    report = metrics.classification_report(y_true, y_pred, output_dict=True)
    score = report["weighted avg"]["f1-score"]
    return score

