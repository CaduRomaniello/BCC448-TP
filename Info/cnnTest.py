import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TF debugg log

# Auxiliar imports
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

# Model auxiliar imports
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import keras_tuner

# Model imports
from tensorflow.keras.models import load_model  # Load saved model
from tensorflow.keras.optimizers import Adam  # Optimizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D


def getSetDir(set="train"):
    curr_path = os.path.dirname(__file__)
    dirs = [f"{curr_path}/../dataset/archive/dataset/{set}/"]
    return dirs


# Load dataset training data
def loadData(set="train", verbose=False):
    data = []
    labels = []

    for dir in getSetDir(set):
        for subdir in os.listdir(dir):
            for img in os.listdir(dir + subdir)[0:100]:
                arrayLikeImage = load_img(dir + subdir + "/" + img, target_size=(300, 300))
                arrayLikeImage = img_to_array(arrayLikeImage)
                data.append(arrayLikeImage)
                labels.append(subdir)
            if verbose:
                print(f"Done with [{subdir}] images.")

    return data, labels


trainData, trainLabels = loadData("train")
testData, testLabels = loadData("test")

# One-Hot Encode dataset
lb = LabelBinarizer()
trainLabels = lb.fit_transform(trainLabels)
trainLabels = to_categorical(trainLabels)  # Turn each label into a binary array

testLabels = lb.fit_transform(testLabels)
testLabels = to_categorical(testLabels)  # Turn each label into a binary array

trainData = np.array(trainData, dtype="float32")
trainLabels = np.array(trainLabels)
testData = np.array(testData, dtype="float32")
testLabels = np.array(testLabels)


# Create Model
kernelWindow = (6, 6)
headModel = Conv2D(64, kernel_size=kernelWindow[0], activation="relu", input_shape=(300, 300, 3))
headModel = MaxPooling2D(pool_size=kernelWindow)(headModel)
headModel = Conv2D(64, kernel_size=kernelWindow[0], activation="relu")(headModel)
headModel = MaxPooling2D(pool_size=kernelWindow)(headModel)
headModel = Flatten(name="Flatten Layer")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.6)(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.6)(headModel)
headModel = Dense(26, activation="softmax")(headModel)

model = Model(inputs=Input(shape=(300, 300, 3)), outputs=headModel)

# Compile the model
STEP_LEN = 1e-5
EPOCHS = 5
BATCH_SIZE = 10
opt = Adam(learning_rate=STEP_LEN, decay=STEP_LEN / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train
T = model.fit(
    x=trainData,
    y=trainLabels,
    batch_size=BATCH_SIZE,
    steps_per_epoch=len(trainData) // BATCH_SIZE,
    validation_data=(testData, testLabels),
    validation_steps=len(testData) // BATCH_SIZE,
    epochs=EPOCHS)
