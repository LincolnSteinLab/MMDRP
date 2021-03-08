# This script 
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train_path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Combination_Prediction/Data/LINCS/GSE70138/DMSO_celllines_subtypes_TRAIN.txt"
test_path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Combination_Prediction/Data/LINCS/GSE70138/DMSO_celllines_subtypes_TEST.txt"
def read_data(path, chunk_size=1000):
    cur_data = pd.read_csv(path, sep='\t', header=0)
    cur_data = cur_data.drop(labels='y', axis=1)
    cur_labels = pd.read_csv(path, sep='\t', header=0, usecols=['y'])
    cur_labels = cur_labels.iloc[:, 0].values
    # cur_split = train_test_split(np.arange(len(cur_labels)), stratify=cur_labels, test_size=0.2)

    le = LabelEncoder()
    le.fit(cur_labels)
    len(list(le.classes_))
    cur_labels = le.transform(cur_labels)
    return cur_data, cur_labels
    # Get chunk, encode labels,
    # cur_chunk = cur_file.get_chunk()


def create_mlp():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=12328, activation="relu", input_shape=(12328, )))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=128, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=64, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=16, activation="softmax"))

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=[keras.metrics.sparse_categorical_crossentropy, keras.metrics.sparse_categorical_accuracy])
    return model


train_data, train_labels = read_data(train_path)
test_data, test_labels = read_data(test_path)
model = create_mlp()
def train_model(model, train_data, train_labels):
    model.fit(train_data, train_labels,
              batch_size=128, epochs=5,
              verbose=1, use_multiprocessing=True)
    model.evaluate(test_data, test_labels)

