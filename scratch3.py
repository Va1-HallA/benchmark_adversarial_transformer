import run_test
import os
import time
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

run_test.generate_attack_examples_imagenet("vit_base_patch16_224", "pgd")
# run_test.generate_attack_examples_mnist("beit", "pgd")


