import numpy as np
import random
from tensorflow.keras.models import load_model
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from load.load_data import load_opendata

# Intialize the tensorflow-gpu <-> physical matching
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# OOP pseudo labeler
openloader = load_opendata()
X_open_test, Y_open_files = openloader.load_data(classifier_label=None, dsize=(64,64), comp_ratio=30)

model_load = load_model("./ckpt/"+"model_oop_cnn")
X_open_test = np.array(X_open_test)/255.0

X_open_pred = model_load.predict(X_open_test)

how_many = 500 ########## set this how many do you work
idxs = np.arange(0, len(Y_open_files)).tolist()
picking = random.sample(idxs, how_many)
openloader.pseudo_label_marking(imgs= np.array(X_open_test)[picking], file_names= np.array(Y_open_files)[picking], pseudo_labels=np.array(X_open_pred)[picking], classifier_label="OOP")