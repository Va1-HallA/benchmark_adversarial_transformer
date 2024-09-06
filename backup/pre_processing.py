import os
import gc
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image

from utils import PATH_VALIDATION, convert_original_idx_to_pytorch_idx, PATH_VALIDATION_LABELS, model_utils

"""Section 0: Selecting GPU"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

"""Section 2: Pre-processing and saving images as .npy files, with each file contains one shard of images"""


def preprocess_x(image_paths_name, model_name, number_of_batches=5):
    """preprocess raw images into numpy arrays in batches"""
    image_paths = sorted(glob(str(image_paths_name / "*")))
    m = model_utils(model_name)
    transform = m["transform"]
    Path(str(Path(Path.cwd()) / "data/validation/{}_initial".format(model_name))).mkdir(parents=True, exist_ok=True)
    save_path = Path(str(Path(Path.cwd()) / "data/validation/{}_initial".format(model_name)))

    batch_size = len(image_paths) // number_of_batches

    for i in range(number_of_batches):
        batch_image_paths = image_paths[i * batch_size:(i + 1) * batch_size] if i < number_of_batches - 1 \
            else image_paths[i * batch_size:]

        x_val = None
        for image_path in batch_image_paths:
            img = Image.open(image_path).convert('RGB')
            x_val_temp = transform(img).unsqueeze(0).numpy()  # store as .npy

            try:
                x_val = np.concatenate([x_val, x_val_temp])
            except ValueError:
                x_val = x_val_temp

        np.save(str(save_path / "x_val_{}.npy").format(i+1), x_val)

        if (i + 1) * 100 / number_of_batches % 5 == 0:
            print("{:.0f}% Completed.".format((i + 1) / number_of_batches * 100))

    # np.save(str(save_path / "x_val_1.npy"), x_val)
    return


def preprocess_y():
    """Section 3: Saving image classes as numpy array to another .npy file"""
    with open(str(PATH_VALIDATION_LABELS), "r") as f:
        y_val = f.read().strip().split("\n")
        y_val = np.array([convert_original_idx_to_pytorch_idx(int(idx)) for idx in y_val])
    np.save(str(PATH_VALIDATION / "y_val.npy"), y_val)



