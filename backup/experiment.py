import run_test
import pre_processing
import os
import torch
from torch.multiprocessing import Process
from utils import PATH_VALIDATION_IMAGES


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def preprocessing():
    p_pre_beit = Process(target=pre_processing.preprocess_x, args=(PATH_VALIDATION_IMAGES, "beit_base_patch16_224",))
    p_pre_vit = Process(target=pre_processing.preprocess_x, args=(PATH_VALIDATION_IMAGES, "vit_base_patch16_224",))
    p_pre_y = Process(target=pre_processing.preprocess_y)

    p_pre_beit.start()
    p_pre_beit.join()
    p_pre_vit.start()
    p_pre_vit.join()
    p_pre_y.start()
    p_pre_y.join()
    return


def build_classifiers():
    p_build_beit_mnist = Process(target=run_test.build_train_classifier_mnist, args=("beit",))
    p_build_vit_mnist = Process(target=run_test.build_train_classifier_mnist, args=("vit",))
    p_build_beit_imagenet = Process(target=run_test.build_classifier_imagenet, args=("beit_base_patch16_224",))
    p_build_vit_imagenet = Process(target=run_test.build_classifier_imagenet, args=("vit_base_patch16_224",))

    p_build_beit_mnist.start()
    p_build_beit_mnist.join()
    p_build_vit_mnist.start()
    p_build_vit_mnist.join()
    p_build_beit_imagenet.start()
    p_build_beit_imagenet.join()
    p_build_vit_imagenet.start()
    p_build_vit_imagenet.join()
    return


def generate_attacks():
    p_attack_pgd_beit_mnist = Process(target=run_test.generate_attack_examples_mnist, args=("beit", "pgd",))
    p_attack_cw_beit_mnist = Process(target=run_test.generate_attack_examples_mnist, args=("beit", "cw",))
    p_attack_zoo_beit_mnist = Process(target=run_test.generate_attack_examples_mnist, args=("beit", "zoo",))

    p_attack_pgd_vit_mnist = Process(target=run_test.generate_attack_examples_mnist, args=("vit", "pgd",))
    p_attack_cw_vit_mnist = Process(target=run_test.generate_attack_examples_mnist, args=("vit", "cw",))
    p_attack_zoo_vit_mnist = Process(target=run_test.generate_attack_examples_mnist, args=("vit", "zoo",))

    p_attack_pgd_beit_imagenet = Process(target=run_test.generate_attack_examples_imagenet,
                                         args=("beit_base_patch16_224", "pgd"))
    p_attack_cw_beit_imagenet = Process(target=run_test.generate_attack_examples_imagenet,
                                        args=("beit_base_patch16_224", "cw"))
    p_attack_zoo_beit_imagenet = Process(target=run_test.generate_attack_examples_imagenet,
                                         args=("beit_base_patch16_224", "zoo"))

    p_attack_pgd_vit_imagenet = Process(target=run_test.generate_attack_examples_imagenet,
                                        args=("vit_base_patch16_224", "pgd"))
    p_attack_cw_vit_imagenet = Process(target=run_test.generate_attack_examples_imagenet,
                                       args=("vit_base_patch16_224", "cw"))
    p_attack_zoo_vit_imagenet = Process(target=run_test.generate_attack_examples_imagenet,
                                        args=("vit_base_patch16_224", "zoo"))

    p_attack_pgd_beit_mnist.start()
    p_attack_pgd_beit_mnist.join()
    p_attack_cw_beit_mnist.start()
    p_attack_cw_beit_mnist.join()
    p_attack_zoo_beit_mnist.start()
    p_attack_zoo_beit_mnist.join()

    p_attack_pgd_vit_mnist.start()
    p_attack_pgd_vit_mnist.join()
    p_attack_cw_vit_mnist.start()
    p_attack_cw_vit_mnist.join()
    p_attack_zoo_vit_mnist.start()
    p_attack_zoo_vit_mnist.join()

    p_attack_pgd_beit_imagenet.start()
    p_attack_pgd_beit_imagenet.join()
    p_attack_cw_beit_imagenet.start()
    p_attack_cw_beit_imagenet.join()
    p_attack_zoo_beit_imagenet.start()
    p_attack_zoo_beit_imagenet.join()

    p_attack_pgd_vit_imagenet.start()
    p_attack_pgd_vit_imagenet.join()
    p_attack_cw_vit_imagenet.start()
    p_attack_cw_vit_imagenet.join()
    p_attack_zoo_vit_imagenet.start()
    p_attack_zoo_vit_imagenet.join()
    return


def run_tests():
    p_test_init_beit_mnist = Process(target=run_test.run_test_mnist, args=("beit",))
    p_test_init_vit_mnist = Process(target=run_test.run_test_mnist, args=("vit",))
    p_test_init_beit_imagenet = Process(target=run_test.run_test_imagenet, args=("beit",))
    p_test_init_vit_imagenet = Process(target=run_test.run_test_imagenet, args=("vit",))

    p_test_pgd_beit_mnist = Process(target=run_test.run_test_mnist, args=("beit", "pgd",))
    p_test_cw_beit_mnist = Process(target=run_test.run_test_mnist, args=("beit", "cw",))
    p_test_zoo_beit_mnist = Process(target=run_test.run_test_mnist, args=("beit", "zoo",))
    p_test_pgd_vit_mnist = Process(target=run_test.run_test_mnist, args=("vit", "pgd",))
    p_test_cw_vit_mnist = Process(target=run_test.run_test_mnist, args=("vit", "cw",))
    p_test_zoo_vit_mnist = Process(target=run_test.run_test_mnist, args=("vit", "zoo",))

    p_test_pgd_beit_imagenet = Process(target=run_test.run_test_imagenet, args=("beit_base_patch16_224", "pgd",))
    p_test_cw_beit_imagenet = Process(target=run_test.run_test_imagenet, args=("beit_base_patch16_224", "cw",))
    p_test_zoo_beit_imagenet = Process(target=run_test.run_test_imagenet, args=("beit_base_patch16_224", "zoo",))
    p_test_pgd_vit_imagenet = Process(target=run_test.run_test_imagenet, args=("vit_base_patch16_224", "pgd",))
    p_test_cw_vit_imagenet = Process(target=run_test.run_test_imagenet, args=("vit_base_patch16_224", "cw",))
    p_test_zoo_vit_imagenet = Process(target=run_test.run_test_imagenet, args=("vit_base_patch16_224", "zoo",))

    p_test_init_beit_mnist.start()
    p_test_init_beit_mnist.join()
    p_test_init_vit_mnist.start()
    p_test_init_vit_mnist.join()
    p_test_init_beit_imagenet.start()
    p_test_init_beit_imagenet.join()
    p_test_init_vit_imagenet.start()
    p_test_init_vit_imagenet.join()

    p_test_pgd_beit_mnist.start()
    p_test_pgd_beit_mnist.join()
    p_test_cw_beit_mnist.start()
    p_test_cw_beit_mnist.join()
    p_test_zoo_beit_mnist.start()
    p_test_zoo_beit_mnist.join()
    p_test_pgd_vit_mnist.start()
    p_test_pgd_vit_mnist.join()
    p_test_cw_vit_mnist.start()
    p_test_cw_vit_mnist.join()
    p_test_zoo_vit_mnist.start()
    p_test_zoo_vit_mnist.join()

    p_test_pgd_beit_imagenet.start()
    p_test_pgd_beit_imagenet.join()
    p_test_cw_beit_imagenet.start()
    p_test_cw_beit_imagenet.join()
    p_test_zoo_beit_imagenet.start()
    p_test_zoo_beit_imagenet.join()
    p_test_pgd_vit_imagenet.start()
    p_test_pgd_vit_imagenet.join()
    p_test_cw_vit_imagenet.start()
    p_test_cw_vit_imagenet.join()
    p_test_zoo_vit_imagenet.start()
    p_test_zoo_vit_imagenet.join()

    return


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    preprocessing()
    build_classifiers()
    generate_attacks()
    run_tests()
