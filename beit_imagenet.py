import sys, os, time, re, gc
from pathlib import Path
from glob import glob

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.applications import vgg16, vgg19, resnet_v2
from PIL import Image
from utils import PATH_VALIDATION, PATH_VALIDATION_LABELS, PATH_META, PATH_SYNSET_WORDS, PATH_VALIDATION_IMAGES, model_utils
from timm import create_model
from art.attacks.evasion import ProjectedGradientDescent, CarliniL2Method
from art.estimators.classification import PyTorchClassifier


import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# Select GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

Build_Classifier = True

Run_Initial_Test = True
Load_Initial_Test = False

Generate_Adversarial_Example_PGD = False
Run_Adversarial_Test_PGD = False
Load_Adversarial_Test_PGD = False

Generate_Adversarial_Example_CW = False
Run_Adversarial_Test_CW = False
Load_Adversarial_Test_CW = False

Generate_Adversarial_Example_ZOO = False
Run_Adversarial_Test_ZOO = False
Load_Adversarial_Test_ZOO = False

ClassifierFilename = "classifier_ART_BEIT_IMAGENET.pkl"
ARTTestSet_PGD = "adv_imagenet_beit_PGD.pkl"
ARTTestSet_CW = "adv_imagenet_beit_CW.pkl"
ARTTestSet_ZOO = "adv_imagenet_beit_ZOO.pkl"

Original_Prediction_Saving_Beit = "y_pred_Original_Beit_Imagenet.npy"

m = model_utils("vit_base_patch16_224")
beit = m["model"]
transform_beit = m["transform"]
beit.eval()

if Build_Classifier:
    print("building classifier")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(beit.parameters(), lr=0.01)
    classifier = PyTorchClassifier(
        model=beit,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=1000,
    )
    pickle.dump(classifier, open(ClassifierFilename, "wb"))
else:
    print("loading classifier")
    classifier = pickle.load(open(ClassifierFilename, "rb"))


# x_val_paths = glob(str(PATH_VALIDATION / "x_val*.npy"))
x_val_paths = glob(str(PATH_VALIDATION_IMAGES / "ILSVRC2012_val*.JPEG"))

# Sort filenames in ascending order
x_val_paths.sort(key=lambda f: int(re.sub('\D', '', f)))

y_val = np.load(str(PATH_VALIDATION / "y_val.npy"))
""" Notice that to_categorical only works using keras version, because the classes have to 
start from 0"""
y_val_one_hot = to_categorical(y_val, 1000)

if Run_Initial_Test:
    y_pred = None
    for i, x_val_path in enumerate(x_val_paths):
        img = Image.open(x_val_path).convert('RGB')
        transform = transform_beit
        x_val = transform(img).unsqueeze(0)
        # x_val = np.transpose(x_val, (0, 3, 1, 2)).astype(np.float32)
        y_pred_sharded = classifier.predict(x_val)
        try:
            y_pred = np.concatenate([y_pred, y_pred_sharded])
        except ValueError:
            y_pred = y_pred_sharded

        del x_val
        gc.collect()

        completed_percentage = (i + 1) * 100 / len(x_val_paths)
        if completed_percentage % 5 == 0:
            print("{:5.1f}% completed.".format(completed_percentage))
    np.save(str(Original_Prediction_Saving_Beit), y_pred)
    y_pred = torch.from_numpy(y_pred)
    print(np.argmax(y_pred[:20],axis=1))
    # accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_val_broadcasted, axis=1)) / len(y_val_broadcasted)
    # print("Accuracy on benign test examples: {}%".format(accuracy * 100))
if Load_Initial_Test:
    y_pred = np.load(Original_Prediction_Saving_Beit)
    accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_val_one_hot, axis=1)) / len(y_val_one_hot)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))


# for now all samples are in one array, need to be sharded
if Run_Adversarial_Test_PGD:
    if Generate_Adversarial_Example_PGD:
        print("generating the attack")
        attack = ProjectedGradientDescent(estimator=classifier, eps=0.2)

        x_adv = None
        for i, x_val_path in enumerate(x_val_paths):
            x_val = np.load(x_val_path).astype('float32')
            x_val = np.transpose(x_val, (0, 3, 1, 2)).astype(np.float32)
            x_adv_sharded = attack.generate(x=x_val)
            try:
                x_adv = np.concatenate([x_adv, x_adv_sharded])
            except ValueError:
                x_adv = x_adv_sharded

            del x_val
            gc.collect()

            completed_percentage = (i + 1) * 100 / len(x_val_paths)
            if completed_percentage % 5 == 0:
                print("{:5.1f}% completed.".format(completed_percentage))

        pickle.dump(x_adv, open(ARTTestSet_PGD, "wb"))
    else:
        x_adv = pickle.load(open(ARTTestSet_PGD, "rb"))








