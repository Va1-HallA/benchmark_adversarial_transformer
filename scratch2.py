from utils import PATH_VALIDATION_LABELS, PATH_VALIDATION,PATH_VALIDATION_IMAGES, convert_original_idx_to_pytorch_idx
from tensorflow.keras.utils import to_categorical
from run_test import run_test_imagenet, build_classifier_imagenet
from pre_processing import preprocess_x
from pathlib import Path
import numpy as np
import torch
import os
import time
import run_test
import pre_processing

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

start_time = time.time()
run_test.generate_attack_examples_mnist("vit", "zoo")
# run_test.run_test_mnist("vit", "cw")
print("--- %s seconds ---" % (time.time() - start_time))
