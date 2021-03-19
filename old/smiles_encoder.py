from tensorflow import keras
import pandas as pd
import numpy as np

from tensorflow.python.keras.layers import Input, GRU, Dense, Conv1D, BatchNormalization, Flatten, RepeatVector, concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
# from tensorflow.python.keras.preprocessing.sequence import



import pydot
import graphviz

# from sklearn.preprocessing import LabelEncoder


batch_size = 64  # Batch size for training.
epochs = 300  # Number of epochs to train for.
latent_dim = 488  # Latent dimensionality of the encoding space.
num_samples = 1796  # Number of samples to train on.
max_len = 260

num_genes = 12328


# TODO USE ZINCS data to augment the autoencoder...
input_path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Combination_Prediction/Data/LINCS_Training/A375_cellline_ctrl_input_data.txt"
label_path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Combination_Prediction/Data/LINCS_Training/A375_cellline_trt_label_data.txt"
smiles_path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Combination_Prediction/Data/LINCS_Training/A375_SMILES_data.txt"
def read_smiles(smiles_path, smiles_len):
    # cur_meta = pd.read_csv(path, sep='\t', header=0)
    # cur_meta = cur_meta.loc[cur_meta['pert_type'] == 'trt_cp']
    # cur_smiles = cur_meta['canonical_smiles'].values
    cur_smiles = pd.read_csv(smiles_path)['cur_smiles'].values
    # cur_smiles = cur_smiles.astype(dtype=str)

    input_characters = set()
    for smiles in cur_smiles:
        for char in smiles:
            input_characters.add(char)

    cur_smiles = [word for word in cur_smiles if len(word) <= smiles_len]
    # len(cur_smiles)
    input_characters = sorted(list(input_characters))
    num_encoder_tokens = len(input_characters)
    all_lens = [len(x) for x in cur_smiles]
    max_encoder_seq_length = max(all_lens)


    print("Only SMILES of maximum length", str(smiles_len), "were retained!")
    print('Number of samples:', len(cur_smiles))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Average sequence length for inputs:', np.mean(all_lens))
    print('Average sequence length for inputs:', np.median(all_lens))
    print('Max sequence length for inputs:', max_encoder_seq_length)

    # Create a list of valid sequences as RNN input
    smile_lists = [' '.join(list(x)) for x in cur_smiles]
    cur_tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_encoder_tokens,
                                                       filters="", lower=False, split=' ')
    cur_tokenizer.fit_on_texts(smile_lists)
    smile_seqs = cur_tokenizer.texts_to_sequences(smile_lists)
    smile_seqs = keras.preprocessing.sequence.pad_sequences(smile_seqs,
                                                            maxlen=max_encoder_seq_length,
                                                            padding="pre")
    # 1 at the end indicates the number of features/channels
    # Reshape input into [samples, timesteps, features]
    train_data = smile_seqs.reshape((smile_seqs.shape[0], smile_seqs.shape[1], 1))
    # encoder_input_data = np.zeros(
    #     (len(cur_smiles), max_encoder_seq_length, num_encoder_tokens),
    #     dtype='float32')
    return train_data


def read_lincs(input_path, label_path):
    cur_input = pd.read_csv(input_path, sep=",", header=0)
    cur_labels = pd.read_csv(label_path, sep=",", header=0)

    return cur_input, cur_labels

def root_mean_squared_error(y_true, y_pred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true)))

smile_data = read_smiles(smiles_path, 260)

lincs_input, lincs_labels = read_lincs(input_path, label_path)
# lincs_input, lincs_labels = cur_input, cur_labels
# input_shape = (max_encoder_seq_length, 1)
# def rnn_autoencoder(input_shape):
    A1 = Input(shape=(max_len, 1), name="Chem_input")
    A2 = Conv1D(filters=9, kernel_size=9, activation="tanh", input_shape=(max_len, 1)) (A1)
    A3 = BatchNormalization() (A2)
    A4 = Conv1D(filters=9, kernel_size=9, activation="tanh") (A3)
    A5 = BatchNormalization() (A4)
    A6 = Conv1D(filters=11, kernel_size=10, activation="tanh") (A5)
    A7 = BatchNormalization() (A6)
    A8 = Flatten() (A7)
    A9 = Dense(196, name="Chem_midlayer") (A8)
    A10 = RepeatVector(max_len) (A9)
    A11 = GRU(units=latent_dim, activation="tanh", return_sequences=True) (A10)
    A12 = BatchNormalization() (A11)
    A13 = GRU(units=latent_dim, activation="tanh", return_sequences=True) (A12)
    A14 = GRU(units=latent_dim, activation="softmax", return_sequences=True, name="Chem_output") (A13)

    B1 = Input(shape=(num_genes, ), name="LINCS_input")
    B2 = Dense(units=num_genes, activation="relu", input_shape=(num_genes, )) (B1)
    B3 = BatchNormalization() (B2)
    B4 = Dense(units=512, activation="relu") (B3)
    B5 = BatchNormalization() (B4)
    B6 = Dense(units=256, activation="relu") (B5)
    B7 = BatchNormalization() (B6)
    B8 = Dense(units=128, activation="linear", name="B8") (B7)

    # Must use lowercase c!
    merged = concatenate([A9, B8])

    AB_1 = Dense(units=128, activation="relu") (merged)
    AB_2 = Dense(units=num_genes, activation="linear", name="LINCS_Chem_output") (AB_1)

    model = Model(inputs=[A1, B1], outputs=[A14, AB_2])

    model.compile(optimizer=Adam(lr=0.001),
                  loss={"LINCS_Chem_output": root_mean_squared_error,
                        "Chem_output": "sparse_categorical_crossentropy"})
    model.summary()
    # keras.losses.r
    model.fit(x={"Chem_input": smile_data, "LINCS_input": lincs_input},
              y={"Chem_output": smile_data, "LINCS_Chem_output": lincs_labels},
              epochs=1, verbose=1, batch_size=batch_size, validation_split=0.1)

    # plot_model(model, show_shapes=True, to_file='lstm_autoencoder.png')
    # keras.utils.plot_model(model, show_shapes=True, to_file="lstm_autoencoder.png")
