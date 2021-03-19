import pandas as pd
import numpy as np
import time

from tensorflow.python.keras.layers import Input, GRU, Dense, Conv1D,\
    BatchNormalization, Flatten, RepeatVector, concatenate, MaxPooling1D,\
    UpSampling1D, GaussianNoise, Embedding, Conv2DTranspose, Lambda, Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.models import load_model

path = "Data/RNAi/Train_Data/"


# Load FCAE model
gene_fcae_model = load_model(path+'gene_FCAE.h5')
gene_fcae_model.summary()
# Extract encoder
fcae_encoder = Model(gene_fcae_model.input, gene_fcae_model.layers[28].output)
# fcae_encoder = K.function([gene_fcae_model.layers[0].input, K.learning_phase()],
#                           [gene_fcae_model.layers[3].output])
# Get output in test mode
fcae_encoder
for i, l in enumerate(fcae_encoder.layers):
    l.trainable = False
    l._name = 'FCAE_Encoder_'+str(i)
fcae_encoder.summary()


# Generate a feature map of all available transcripts using the encoder
def transcript_generator(transcript_path, chunk_size):
    while True:  # keras requires all generators to be infinite
        base_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0],
                     'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
        for cur_transcripts in pd.read_csv(path+transcript_path, engine='c', sep=',', chunksize=chunk_size):
            cur_alt_lists = cur_transcripts['transcript'].tolist()
            list_alt = [list(x) for x in cur_alt_lists]
            # Use a dictionary to perform multiple replacements at once
            # ~4x faster than using list comprehension in succession
            encoded = [[base_dict.get(x, x) for x in y] for y in list_alt]
            one_hot_alt = [np.array(x).reshape((1, len(x), 4)) for x in encoded]

            all_lengths = np.unique([len(x) for x in encoded])
            # Group them by the same length
            for length in all_lengths:
                cur_sub = np.array([x for x in one_hot_alt if x.shape[1] == length])
                cur_sub = cur_sub.reshape(cur_sub.shape[0], cur_sub.shape[2], 4)
                assert not np.isnan(np.sum(cur_sub))
                # for i in range(len(one_hot_alt)):
                # Detect odd length input

                if (cur_sub.shape[1] & 1) == 1:
                    # Duplicate last base or just add a zero vector
                    # x_train[i][len(x_train[i])-1]
                    cur_sub = np.concatenate((cur_sub,
                                                 np.tile(np.array([0,0,0,0]),
                                                         (cur_sub.shape[0], 1)).reshape((cur_sub.shape[0], 1, 4))), axis=1)
                    # Since we're doing 3 transposed convolutions, we're dividing by 48, so must add 2 more
                if cur_sub.shape[1] % 4 == 2:
                    cur_sub = np.concatenate((cur_sub,
                                              np.tile(np.array([0,0,0,0]),
                                                      (cur_sub.shape[0]*2, 1)).reshape((cur_sub.shape[0], 2, 4))), axis=1)
                if cur_sub.shape[1] % 8 == 4:
                    cur_sub = np.concatenate((cur_sub,
                                              np.tile(np.array([0,0,0,0]),
                                                      (cur_sub.shape[0]*4, 1)).reshape((cur_sub.shape[0], 4, 4))), axis=1)

                cur_size = np.prod(cur_sub.shape)
                if cur_size > 1e6:
                    num_parts = np.int(np.ceil(cur_size/1e6))
                    for i in range(0, cur_sub.shape[0], num_parts):
                        # This automatically ensures only valid indices are kept,
                        # so num_parts can be larger than what remains
                        cur_train = cur_sub[i:i+num_parts, :]
                        yield cur_train, cur_train
                else:
                    yield (cur_sub, cur_sub)


all_predictions = fcae_encoder.predict_generator(generator=transcript_generator(transcript_path=path+'all_transcript_seqs.txt',
                                                                                chunk_size=1000), steps=,
                                                 use_multiprocessing=True, verbose=True)


base_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0],
             'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
all_preds = []
for cur_transcripts in pd.read_csv(path+'all_transcript_seqs.txt', engine='c', sep=',', chunksize=1000):
    cur_alt_lists = cur_transcripts['transcript'].tolist()
    list_alt = [list(x) for x in cur_alt_lists]
    # Use a dictionary to perform multiple replacements at once
    # ~4x faster than using list comprehension in succession
    encoded = [[base_dict.get(x, x) for x in y] for y in list_alt]
    one_hot_alt = [np.array(x).reshape((1, len(x), 4)) for x in encoded]

    all_lengths = np.unique([len(x) for x in encoded])
    # Group them by the same length
    for length in all_lengths:
        cur_sub = np.array([x for x in one_hot_alt if x.shape[1] == length])
        cur_sub = cur_sub.reshape(cur_sub.shape[0], cur_sub.shape[2], 4)
        assert not np.isnan(np.sum(cur_sub))
        # for i in range(len(one_hot_alt)):
        # Detect odd length input

        if (cur_sub.shape[1] & 1) == 1:
            # Duplicate last base or just add a zero vector
            # x_train[i][len(x_train[i])-1]
            cur_sub = np.concatenate((cur_sub,
                                         np.tile(np.array([0,0,0,0]),
                                                 (cur_sub.shape[0], 1)).reshape((cur_sub.shape[0], 1, 4))), axis=1)
            # Since we're doing 3 transposed convolutions, we're dividing by 48, so must add 2 more
        if cur_sub.shape[1] % 4 == 2:
            cur_sub = np.concatenate((cur_sub,
                                      np.tile(np.array([0,0,0,0]),
                                              (cur_sub.shape[0]*2, 1)).reshape((cur_sub.shape[0], 2, 4))), axis=1)
        if cur_sub.shape[1] % 8 == 4:
            cur_sub = np.concatenate((cur_sub,
                                      np.tile(np.array([0,0,0,0]),
                                              (cur_sub.shape[0]*4, 1)).reshape((cur_sub.shape[0], 4, 4))), axis=1)

        cur_size = np.prod(cur_sub.shape)
        if cur_size > 1e6:
            num_parts = np.int(np.ceil(cur_size/1e6))
            for i in range(0, cur_sub.shape[0], num_parts):
                # This automatically ensures only valid indices are kept,
                # so num_parts can be larger than what remains
                cur_train = cur_sub[i:i+num_parts, :]
                cur_preds = fcae_encoder.predict_on_batch(cur_train)
        else:
            cur_preds = fcae_encoder.predict_on_batch(cur_sub)

        all_preds.append(cur_preds)

all_preds[0][0].shape
