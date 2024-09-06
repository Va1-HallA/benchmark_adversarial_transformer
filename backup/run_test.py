import gc
import os, re, sys
import pickle
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from art.attacks.evasion import ProjectedGradientDescent, CarliniL2Method, ZooAttack
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from tensorflow.keras.utils import to_categorical
from timm.models.beit import Beit
from timm.models.vision_transformer import VisionTransformer
from os.path import exists

from utils import PATH_VALIDATION, PATH_VALIDATION_IMAGES
from utils import model_utils

# Select GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # although mostly done with cpu, the model is based on gpu

# global variables defining file names

"""Define global variables"""
Classifier_File_Name_MNist_Beit = "Classifier_Beit_MNist.pkl"
Classifier_File_Name_MNist_Vit = "Classifier_Vit_MNist.pkl"
Attack_Data_File_PGD_MNist_Beit = "Adv_Example_PGD_Beit_MNist.pkl"
Attack_Data_File_CW_MNist_Beit = "Adv_Example_CW_Beit_MNist.pkl"
Attack_Data_File_ZOO_MNist_Beit = "Adv_Example_ZOO_Beit_MNist.pkl"
Attack_Data_File_PGD_MNist_Vit = "Adv_Example_PGD_Vit_MNist.pkl"
Attack_Data_File_CW_MNist_Vit = "Adv_Example_CW_Vit_MNist.pkl"
Attack_Data_File_ZOO_MNist_Vit = "Adv_Example_ZOO_Vit_MNist.pkl"
Test_Result_Initial_MNist_Beit = "y_pred_Init_Beit_MNist.npy"
Test_Result_Initial_MNist_Vit = "y_pred_Init_Vit_MNist.npy"
Test_Result_PGD_MNist_Beit = "y_pred_PGD_Beit_MNist.npy"
Test_Result_CW_MNist_Beit = "y_pred_CW_Beit_MNist.npy"
Test_Result_ZOO_MNist_Beit = "y_pred_ZOO_Beit_MNist.npy"
Test_Result_PGD_MNist_Vit = "y_pred_PGD_Vit_MNist.npy"
Test_Result_CW_MNist_Vit = "y_pred_CW_Vit_MNist.npy"
Test_Result_ZOO_MNist_Vit = "y_pred_ZOO_Vit_MNist.npy"

ATTACK_DATA_FILE_MAP_MNist = {
        "beit&pgd": Attack_Data_File_PGD_MNist_Beit,
        "beit&cw": Attack_Data_File_CW_MNist_Beit,
        "beit&zoo": Attack_Data_File_ZOO_MNist_Beit,
        "vit&pgd": Attack_Data_File_PGD_MNist_Vit,
        "vit&cw": Attack_Data_File_CW_MNist_Vit,
        "vit&zoo": Attack_Data_File_ZOO_MNist_Vit,
    }

TEST_RESULT_MAP_MNist = {
    "beit&pgd": Test_Result_PGD_MNist_Beit,
    "beit&cw": Test_Result_CW_MNist_Beit,
    "beit&zoo": Test_Result_ZOO_MNist_Beit,
    "vit&pgd": Test_Result_PGD_MNist_Vit,
    "vit&cw": Test_Result_CW_MNist_Vit,
    "vit&zoo": Test_Result_ZOO_MNist_Vit,
    "beit&initial": Test_Result_Initial_MNist_Beit,
    "vit&initial": Test_Result_Initial_MNist_Vit
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_classifier_imagenet(model_name):
    classifier_file_name = "classifier_{}_IMAGENET.pkl".format(model_name)
    print("building classifier")

    m = model_utils(model_name)
    model = m["model"]
    model.eval()

    # change criterion and optimizer might get better results
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=1000,
    )
    pickle.dump(classifier, open(classifier_file_name, "wb"))
    return


def run_test_imagenet(model_name, attack_name="initial"):
    """using pre-trained models to validate data"""

    # releasing cuda memory
    torch.cuda.empty_cache()

    # loading classifier according to model name
    classifier_file_name = "classifier_{}_IMAGENET.pkl".format(model_name)
    if not exists(str(PATH_VALIDATION / "y_val.npy")):
        print("True output has not been pre-processed. Please run preprocessing.preprocess_y() function first. Error place: run_test_imagenet, parameters: {}, {}".format(model_name, attack_name))
        return
    y_val = np.load(str(PATH_VALIDATION / "y_val.npy"))
    y_val_one_hot = to_categorical(y_val, 1000)

    print("loading classifier")
    if not exists(classifier_file_name):
        print("Classifier not exist. Please build one first. Error place: run_test_imagenet, parameters: {}, {}".format(model_name, attack_name))
        return
    classifier = pickle.load(open(classifier_file_name, "rb"))
    print("done loading")

    # define x values and result file name according to attack name
    if attack_name == "initial":
        if not exists(str(Path(Path.cwd()) / "data/validation/{}_initial".format(model_name))):
            print("Attack examples not exist. Please generate attack examples first. Error place: run_test_imagenet, parameters: {}, {}".format(model_name, attack_name))
            return

        directory = Path(str(Path(Path.cwd()) / "data/validation/{}_initial".format(model_name)))
        x_val_paths = glob(str(directory / "*"))
        x_val_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
        result_file_name = "y_pred_Initial_{}_Imagenet.npy".format(model_name)

    elif attack_name == "pgd":
        if not exists(str(Path(Path.cwd()) / "data/validation/{}_pgd_Imagenet".format(model_name))):
            print("Attack examples not exist. Please generate attack examples first. Error place: run_test_imagenet, parameters: {}, {}".format(model_name, attack_name))
            return

        directory = Path(str(Path(Path.cwd()) / "data/validation/{}_pgd_Imagenet".format(model_name)))
        x_val_paths = sorted(glob(str(directory / "*")))
        x_val_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
        result_file_name = "y_pred_PGD_{}_Imagenet.npy".format(model_name)

    elif attack_name == "cw":
        if not exists(str(Path(Path.cwd()) / "data/validation/{}_cw_Imagenet".format(model_name))):
            print("Attack examples not exist. Please generate attack examples first. Error place: run_test_imagenet, parameters: {}, {}".format(model_name, attack_name))
            return

        directory = Path(str(Path(Path.cwd()) / "data/validation/{}_cw_Imagenet".format(model_name)))
        x_val_paths = sorted(glob(str(directory / "*")))
        x_val_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
        result_file_name = "y_pred_CW_{}_Imagenet.npy".format(model_name)

    elif attack_name == "zoo":
        if not exists(str(Path(Path.cwd()) / "data/validation/{}_zoo_Imagenet".format(model_name))):
            print("Attack examples not exist. Please generate attack examples first. Error place: run_test_imagenet, parameters: {}, {}".format(model_name, attack_name))
            return

        directory = Path(str(Path(Path.cwd()) / "data/validation/{}_zoo_Imagenet".format(model_name)))
        x_val_paths = sorted(glob(str(directory / "*")))
        x_val_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
        result_file_name = "y_pred_ZOO_{}_Imagenet.npy".format(model_name)
    else:
        print("Unsupported attack name. Valid names are: pgd, cw, zoo.")
        return

    y_pred = None

    for i, x_val_path in enumerate(x_val_paths):
        x_val = torch.from_numpy(np.load(x_val_path))  # loaded as numpy array, but need to convert it to torch.tensor
        y_pred_sharded = classifier.predict(x_val)  # the output is numpy array
        try:
            y_pred = np.concatenate([y_pred, y_pred_sharded])
        except ValueError:
            y_pred = y_pred_sharded

        del x_val
        gc.collect()

        # completed_percentage = (i + 1) * 100 / len(x_val_paths)
        # if completed_percentage % 5 == 0:
        #     print("{:5.1f}% completed.".format(completed_percentage))
    np.save(str(result_file_name), y_pred)
    # y_pred = torch.from_numpy(y_pred)  # change the numpy array to tensor.torch, will cause error when calculating accuracy

    accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_val_one_hot[:500], axis=1)) / len(y_val_one_hot[:500])
    print("Accuracy on {} test examples: {}%".format(attack_name, accuracy * 100))
    return


def load_test_imagenet(model_name, attack_name="initial"):
    if attack_name == "initial":
        file_name = "y_pred_Initial_{}_Imagenet.npy".format(model_name)
    elif attack_name == "pgd":
        file_name = "y_pred_PGD_{}_Imagenet.npy".format(model_name)
    elif attack_name == "cw":
        file_name = "y_pred_CW_{}_Imagenet.npy".format(model_name)
    elif attack_name == "zoo":
        file_name = "y_pred_ZOO_{}_Imagenet.npy".format(model_name)
    else:
        print("Unsupported attack name. Valid names are: pgd, cw, zoo.")
        return

    if not exists(str(PATH_VALIDATION / "y_val.npy")):
        print("True output has not been pre-processed. Please run preprocessing.preprocess_y() function first. Error place: load_test_imagenet, parameters: {}, {}".format(model_name, attack_name))
        return
    y_val = np.load(str(PATH_VALIDATION / "y_val.npy"))
    y_val_one_hot = to_categorical(y_val, 1000)

    if not exists(file_name):
        print("No experiment result has been found. Please run the experiment first. Error place: load_test_imagenet, parameters: {}, {}".format(model_name, attack_name))
        return
    y_pred = np.load(file_name)
    # y_pred = torch.from_numpy(y_pred)

    accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_val_one_hot[:500], axis=1)) / len(y_val_one_hot[:500])
    print("Accuracy of {} test: {}%".format(attack_name, accuracy * 100))
    return


def generate_attack_examples_imagenet(model_name, attack_name):
    classifier_file_name = "classifier_{}_IMAGENET.pkl".format(model_name)

    print("loading classifier")
    if not exists(classifier_file_name):
        print("Classifier not exist. Please build one first. Error place: generate_attack_examples_imagenet, parameters: {}, {}".format(model_name, attack_name))
        return
    classifier = pickle.load(open(classifier_file_name, "rb"))
    print("done loading")

    initial_data_directory = Path(str(Path(Path.cwd()) / "data/validation/{}_initial".format(model_name)))

    if not exists(initial_data_directory):
        print("Input data not exist. Please preprocess the images first. Error place: generate_attack_examples_imagenet, parameters: {}, {}".format(model_name, attack_name))
        return

    x_val_paths = glob(str(initial_data_directory / "*"))
    x_val_paths.sort(key=lambda f: int(re.sub('\D', '', f)))

    if attack_name == "pgd":
        attack = ProjectedGradientDescent(estimator=classifier, eps=0.2)
        Path(str(Path(Path.cwd()) / "data/validation/{}_pgd_Imagenet".format(model_name))).mkdir(parents=True, exist_ok=True)
        attack_directory = Path(str(Path(Path.cwd()) / "data/validation/{}_pgd_Imagenet".format(model_name)))
    elif attack_name == "cw":
        attack = CarliniL2Method(classifier=classifier)
        Path(str(Path(Path.cwd()) / "data/validation/{}_cw_Imagenet".format(model_name))).mkdir(parents=True,exist_ok=True)
        attack_directory = Path(str(Path(Path.cwd()) / "data/validation/{}_cw_Imagenet".format(model_name)))
    elif attack_name == "zoo":
        attack = ZooAttack(classifier=classifier, use_resize=False, max_iter=100, abort_early=True)
        Path(str(Path(Path.cwd()) / "data/validation/{}_zoo_Imagenet".format(model_name))).mkdir(parents=True,exist_ok=True)
        attack_directory = Path(str(Path(Path.cwd()) / "data/validation/{}_zoo_Imagenet".format(model_name)))
    else:
        print("Unsupported attack name. Valid names are: pgd, cw, zoo.")
        return

    start_time = time.time()
    for i, x_val_path in enumerate(x_val_paths):
        x = np.load(x_val_path)
        x_test_adv = attack.generate(x=x)
        np.save(str(attack_directory / "x_val_{}.npy".format(i + 1)),
                x_test_adv)  # save as .npy file, but need to change to torch.tensor when feed into model
    generating_time = time.time() - start_time
    with open("generating_time.txt", "a+") as file:
        file.write("{} attack on {} in ImageNet: use {} seconds.\n".format(attack_name, model_name, generating_time))

    # TODO: check this code
    # calculate mean, then save loss values
    if attack_name == "pgd":
        loss_values = attack._attack.loss_values
    else:
        loss_values = attack.loss_values
    loss_values = np.mean(loss_values, axis=0)
    np.save("loss_value_{}_{}_{}".format(attack_name, model_name, "Imagenet"), loss_values)
    return


def build_train_classifier_mnist(model_name):
    print("building classifier")
    if model_name == "beit":
        classifier_file_name = Classifier_File_Name_MNist_Beit
        model = Beit(
            img_size=28,
            patch_size=4,
            num_classes=10,
            in_chans=1,
            embed_dim=256,
            depth=2,
            num_heads=16,
            mlp_ratio=1,
        )
    elif model_name == "vit":
        classifier_file_name = Classifier_File_Name_MNist_Vit
        model = VisionTransformer(
            img_size=28,
            patch_size=4,
            num_classes=10,
            in_chans=1,
            embed_dim=256,
            depth=2,
            num_heads=16,
            mlp_ratio=1
        )
    else:
        print("Unsupported model name. Valid names are: beit, vit.")
        return

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
    # Swap axes to PyTorch's NCHW format

    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )
    print("training classifier")
    classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)
    model.eval()  # not sure if I should write this here, I think if you want to evaluate a model then you should use it
    pickle.dump(classifier, open(classifier_file_name, "wb"))
    print("done")
    return


def generate_attack_examples_mnist(model_name, attack_name):

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
    # Swap axes to PyTorch's NCHW format

    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

    # todo: in real test we use all instead of first 20
    x_test = x_test[:500]

    file_name = model_name+"&"+attack_name

    if model_name == "beit":
        if not exists(Classifier_File_Name_MNist_Beit):
            print("Classifier not found. Please build one first. Error place: generate_attack_examples_mnist, parameters: {}, {}".format(model_name, attack_name))
            return
        classifier = pickle.load(open(Classifier_File_Name_MNist_Beit, "rb"))
    elif model_name == "vit":
        if not exists(Classifier_File_Name_MNist_Vit):
            print("Classifier not found. Please build one first. Error place: generate_attack_examples_mnist, parameters: {}, {}".format(model_name, attack_name))
            return
        classifier = pickle.load(open(Classifier_File_Name_MNist_Vit, "rb"))
    else:
        print("Unsupported model name. Valid names are: beit, vit")
        return

    print("generating the attack")
    if attack_name == "pgd":
        attack = ProjectedGradientDescent(estimator=classifier, eps=0.2)
    elif attack_name == "cw":
        attack = CarliniL2Method(classifier=classifier)
    elif attack_name == "zoo":
        attack = ZooAttack(classifier=classifier, use_resize=False, max_iter=100, abort_early=True)
    else:
        print("Unsupported attack name. Valid names are: pgd, cw, zoo.")
        return
    start_time = time.time()
    x_test_adv = attack.generate(x=x_test, model_name=model_name, dataset_name="MNIST")
    generating_time = time.time() - start_time
    with open("generating_time.txt", "a+") as file:
        file.write("{} attack on {} in MNIST: use {} seconds.\n".format(attack_name, model_name, generating_time))

    # TODO: check this code
    # calculate mean, then save loss values
    if attack_name == "pgd":
        loss_values = attack._attack.loss_values
    else:
        loss_values = attack.loss_values
    loss_values = np.mean(loss_values, axis=0)
    np.save("loss_value_{}_{}_{}".format(attack_name, model_name, "MNIST"), loss_values)

    pickle.dump(x_test_adv, open(ATTACK_DATA_FILE_MAP_MNist[file_name], "wb"))
    print("done generating attack")
    return


def run_test_mnist(model_name, attack_name="initial"):
    file_name = model_name + "&" + attack_name

    if model_name == "beit":
        if not exists(Classifier_File_Name_MNist_Beit):
            print("Classifier not found. Please build one first. Error place: run_test_mnist, parameters: {}, {}".format(model_name, attack_name))
            return

        classifier = pickle.load(open(Classifier_File_Name_MNist_Beit, "rb"))
    elif model_name == "vit":
        if not exists(Classifier_File_Name_MNist_Vit):
            print("Classifier not found. Please build one first. Error place: run_test_mnist, parameters: {}, {}".format(model_name, attack_name))
            return
        classifier = pickle.load(open(Classifier_File_Name_MNist_Vit, "rb"))
    else:
        print("Unsupported model name. Valid names are: beit, vit")
        return
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
    # x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

    # todo: in real test we use all instead of first 20
    x_test = x_test[:500]
    y_test = y_test[:500]

    # check whether the initial test is running
    if attack_name == "initial":
        x_test_adv = x_test
    else:
        x_test_adv = pickle.load(open(ATTACK_DATA_FILE_MAP_MNist[file_name], "rb"))
    y_pred = classifier.predict(x_test_adv)
    # saving result
    np.save(TEST_RESULT_MAP_MNist[file_name], y_pred)
    accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on {} test: {}%".format(attack_name, accuracy * 100))
    return


def load_test_mnist(model_name, attack_name="initial"):
    file_name = model_name + "&" + attack_name
    if not exists(file_name):
        print("Experiment output has not been found. Please run experiment first. Error place: load_test_mnist, parameters: {}, {}".format(model_name, attack_name))
        return
    y_pred = np.load(TEST_RESULT_MAP_MNist[file_name])
    (_, _), (_, y_test), _, _ = load_mnist()
    #todo: in real test use 500 instead of 20
    y_test = y_test[:500]
    accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on {} test: {}%".format(attack_name, accuracy * 100))
    return
