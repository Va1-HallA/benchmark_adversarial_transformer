from art.attacks.evasion import ProjectedGradientDescent, CarliniL2Method
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from timm.models.beit import Beit

import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle

m = Beit(
    img_size=28,
    patch_size=4,
    num_classes=10,
    in_chans=1,
    embed_dim=256,
    depth=2,
    num_heads=16,
    mlp_ratio=1,
)


Build_Classifier = False
Run_Initial_Test = True
Generate_Adversarial_Example_PGD = False
Run_Adversarial_Test_PGD = False
Generate_Adversarial_Example_CW = False
Run_Adversarial_Test_CW = False
Generate_Adversarial_Example_ZOO = False
Run_Adversarial_Test_ZOO = False
ClassifierFilename = "classifier_ART_BEIT.pkl"
ARTTestSet_PGD = "adv_example_beit_PGD.pkl"
ARTTestSet_CW = "adv_example_beit_CW.pkl"
ARTTestSet_ZOO = "adv_example_beit_ZOO.pkl"

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
# Swap axes to PyTorch's NCHW format

x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
if Build_Classifier:

    print("building classifier")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(m.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=m,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)
    pickle.dump(classifier, open(ClassifierFilename, "wb"))
else:
    print("loading classifier")
    classifier = pickle.load(open(ClassifierFilename, "rb"))

if Run_Initial_Test:
    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# redundant
if Run_Adversarial_Test_PGD:
    if Generate_Adversarial_Example_PGD:
        print("generating the attack")
        attack = ProjectedGradientDescent(estimator=classifier, eps=0.2)
        x_test_adv = attack.generate(x=x_test)
        pickle.dump(x_test_adv, open(ARTTestSet_PGD, "wb"))
    else:
        x_test_adv = pickle.load(open(ARTTestSet_PGD, "rb"))
    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

if Run_Adversarial_Test_CW:
    if Generate_Adversarial_Example_CW:
        print("generating the attack")
        attack = CarliniL2Method(classifier=classifier, batch_size=64, confidence=0.1)
        x_test_adv = attack.generate(x=x_test)
        pickle.dump(x_test_adv, open(ARTTestSet_CW, "wb"))
    else:
        x_test_adv = pickle.load(open(ARTTestSet_CW, "rb"))
    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

if Run_Adversarial_Test_ZOO:
    if Generate_Adversarial_Example_ZOO:
        print("generating the attack")
        attack = ProjectedGradientDescent(estimator=classifier, eps=0.2)
        x_test_adv = attack.generate(x=x_test)
        pickle.dump(x_test_adv, open(ARTTestSet_ZOO, "wb"))
    else:
        x_test_adv = pickle.load(open(ARTTestSet_ZOO, "rb"))
    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))