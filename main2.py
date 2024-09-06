import run_test
import pre_processing
import os
import time
from utils import PATH_VALIDATION_IMAGES
from multiprocessing import Process

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pre_processing.preprocess_x(PATH_VALIDATION_IMAGES, "beit_base_patch16_224")
pre_processing.preprocess_x(PATH_VALIDATION_IMAGES, "vit_base_patch16_224")
pre_processing.preprocess_y()

run_test.build_classifier_imagenet("beit_base_patch16_224")
run_test.build_classifier_imagenet("vit_base_patch16_224")

run_test.generate_attack_examples_imagenet("beit_base_patch16_224", "zoo")
run_test.generate_attack_examples_imagenet("vit_base_patch16_224", "zoo")
run_test.run_test_imagenet("beit_base_patch16_224", "zoo")
run_test.run_test_imagenet("vit_base_patch16_224", "zoo")