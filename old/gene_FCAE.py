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

path = "Data/RNAi/Train_Data/"


# cur_transcripts = pd.read_csv(path+'transcripts_alt_and_ref.txt', sep=',', nrows=1000)
#
# cur_alt_lists = cur_transcripts['alt_cdna'].tolist()
#
# list_alt = [list(x) for x in cur_alt_lists]
# unlist_alt = [x for y in list_alt for x in y]
# t1 = [[[1,0,0,0] if x=='A' else x for x in y] for y in list_alt]
# t2 = [[[0,1,0,0] if x=='T' else x for x in y] for y in t1]
# t3 = [[[0,0,1,0] if x=='C' else x for x in y] for y in t2]
# t4 = [[[0,0,0,1] if x=='G' else x for x in y] for y in t3]
# len(t4)
# one_hot_alt = np.array(t4)
# one_hot_alt.shape
# len(one_hot_alt[0])
# len(one_hot_alt[100])
#
# one_hot_alt = [np.array(x).reshape((len(x), 4)) for x in t4]
# one_hot_alt[0].shape
# one_hot_alt[100].shape
#
# batch_size = 64
# num_channels = 3
# num_classes = 10

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same', activation='relu'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1),
                        strides=(strides, 1), padding=padding, activation=activation)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

def create_model():
    print('Defining model...')
    cdna_input = Input(shape=(None, 4))
    # X = Reshape((None, 4))(cdna_input)
    # uses theano ordering. Note that we leave the image size as None to allow multiple image sizes
    tower1_1 = Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation='relu')(cdna_input)
    tower1_1 = BatchNormalization()(tower1_1)

    tower1_2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(cdna_input)
    tower1_2 = BatchNormalization()(tower1_2)

    tower1_3 = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(cdna_input)
    tower1_3 = BatchNormalization()(tower1_3)

    concat1 = concatenate([tower1_1, tower1_2, tower1_3], axis=2)
    concat1 = MaxPooling1D(pool_size=2, data_format='channels_last')(concat1)

    tower2_1 = Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation='relu')(concat1)
    tower2_1 = Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation='relu')(tower2_1)
    tower2_1 = BatchNormalization()(tower2_1)

    tower2_2 = Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation='relu')(concat1)
    tower2_2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(tower2_2)
    tower2_2 = BatchNormalization()(tower2_2)

    tower2_3 = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(concat1)
    tower2_3 = BatchNormalization()(tower2_3)

    concat2 = concatenate([tower2_1, tower2_2, tower2_3], axis=2)
    concat2 = MaxPooling1D(pool_size=2, data_format='channels_last')(concat2)

    tower3_1 = Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation='relu')(concat2)
    tower3_1 = Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation='relu')(tower3_1)
    tower3_1 = BatchNormalization()(tower3_1)

    tower3_2 = Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation='relu')(concat2)
    tower3_2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(tower3_2)
    tower3_2 = BatchNormalization()(tower3_2)

    tower3_3 = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(concat2)
    tower3_3 = BatchNormalization()(tower3_3)

    concat3 = concatenate([tower3_1, tower3_2, tower3_3], axis=2)
    latent_map = MaxPooling1D(pool_size=2, data_format='channels_last')(concat3)

    # X = MaxPooling1D(pool_size=2)(X)
    # X = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(X)
    # X = BatchNormalization()(X)
    # X = MaxPooling1D(pool_size=2)(X)

    # X = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(X)
    X = Conv1DTranspose(latent_map, filters=32, kernel_size=3, strides=2, padding='same', activation='relu')
    X = BatchNormalization()(X)
    X = Conv1DTranspose(X, filters=32, kernel_size=3, strides=2, padding='same', activation='relu')
    X = BatchNormalization()(X)
    X = Conv1DTranspose(X, filters=32, kernel_size=3, strides=2, padding='same', activation='relu')
    X = BatchNormalization()(X)

    cdna_output = Conv1D(filters=4, kernel_size=1, strides=1, activation='sigmoid')(X)

    # X = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(X)
    # X = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(X)
    # X = MaxPooling1D(pool_size=2)(X)

    cur_model = Model(cdna_input, cdna_output)
    cur_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # cur_model.build(input_shape=(1103, 4))
    cur_model.summary()
    return cur_model
# plot_model(cur_model, show_shapes=True, show_layer_names=True, to_file='gene_FCAE.png')

# x_train = np.array(one_hot_alt)
# x_train[i].shape
# for i in range(100):
#     # Detect odd length input
#     if (len(x_train[i]) & 1) == 1:
#         # Duplicate last base or just add a zero vector
#         # x_train[i][len(x_train[i])-1]
#         x_train[i] = np.vstack((x_train[i], np.array([0,0,0,0])))
#         # Since we're doing 2 transposed convolutions, we're dividing by 4, so must add 2 more
#     if (len(x_train[i]) % 4 == 2):
#         x_train[i] = np.vstack((x_train[i],
#                                 np.tile(np.array([0,0,0,0]),(2,1))))
#     # print(x_train[i].shape)
#     cur_model.train_on_batch(x=x_train[i].reshape(1, x_train[i].shape[0], 4),
#                              y=x_train[i].reshape(1, x_train[i].shape[0], 4))
#
#     i = 0
#     cur_model.predict(x_train[i].reshape(1, x_train[i].shape[0], 4))
#
# cur_transcripts = pd.read_csv(path+'all_transcript_seqs.txt', nrows=1000)
def my_generator(transcript_path, chunk_size):
    while True:  # keras requires all generators to be infinite
        base_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0],
                     'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
        for cur_transcripts in pd.read_csv(path+transcript_path, engine='c', sep=',', chunksize=chunk_size):
            cur_alt_lists = cur_transcripts['transcript'].tolist()
            list_alt = [list(x) for x in cur_alt_lists]
            # Use a dictionary to perform multiple replacements at once
            # ~4x faster than using list comprehension in succession
            encoded = [[base_dict.get(x, x) for x in y] for y in list_alt]
            # t1 = [[[1,0,0,0] if x=='A' else x for x in y] for y in list_alt]
            # t2 = [[[0,1,0,0] if x=='T' else x for x in y] for y in t1]
            # t3 = [[[0,0,1,0] if x=='C' else x for x in y] for y in t2]
            # t4 = [[[0,0,0,1] if x=='G' else x for x in y] for y in t3]
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


# temp = my_generator(transcript_path='all_transcript_seqs.txt', chunk_size=1000)
# x, y = next(temp)
# cur_model.fit(x, y)


if __name__ == '__main__':
    cur_model = create_model()
    cur_model.fit_generator(generator=my_generator(transcript_path='all_transcript_seqs.txt', chunk_size=1000),
                            steps_per_epoch=22500, epochs=1, use_multiprocessing=True, verbose=True, max_queue_size=100)

    cur_model.save(path+'gene_FCAE.h5')
