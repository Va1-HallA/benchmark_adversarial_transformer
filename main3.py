import run_test
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# run_test.build_classifier_imagenet("beit_base_patch16_224")
# run_test.build_classifier_imagenet("vit_base_patch16_224")
run_test.run_test_imagenet("beit_base_patch16_224")
run_test.run_test_imagenet("vit_base_patch16_224")