import pandas as pd
import numpy as np
import time
import sys
import gc
import logging
import threading
import pickle
import tensorflow
import os

# Enable mixed precision training
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# tf.version
# tensorflow.__version__
from tensorflow.keras.callbacks import ModelCheckpoint
# if '1.13' in tf.VERSION:
#     from tensorflow.python.keras.layers import Input, GRU, Dense, Conv1D,\
#         BatchNormalization, Flatten, RepeatVector, concatenate, MaxPooling1D,\
#         UpSampling1D, GaussianNoise, Embedding, Conv2DTranspose, Lambda, Reshape, Add, Activation
#     from tensorflow.python.keras.models import Model
#     from tensorflow.python.keras.optimizers import Adam
#     from tensorflow.python.keras.preprocessing.text import Tokenizer
#     import tensorflow.python.keras.backend as K
#     from tensorflow.python.keras.utils import plot_model
# elif '1.12' in tf.VERSION:
#     from tensorflow.keras.layers import Input, GRU, Dense, Conv1D,\
#         BatchNormalization, Flatten, RepeatVector, concatenate, MaxPooling1D,\
#         UpSampling1D, GaussianNoise, Embedding, Conv2DTranspose, Lambda, Reshape, Add, Activation
#     from tensorflow.keras.models import Model
#     from tensorflow.keras.optimizers import Adam
#     from tensorflow.keras.preprocessing.text import Tokenizer
#     import tensorflow.keras.backend as K
#     from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Input, GRU, Dense, Conv1D,\
    BatchNormalization, Flatten, RepeatVector, concatenate, MaxPooling1D,\
    UpSampling1D, GaussianNoise, Embedding, Conv2DTranspose, Lambda, Reshape, Add, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
# from keras.preprocessing.text import Tokenizer
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils.data_utils import Sequence
# from keras.utils import plot_model
# from keras.utils import multi_gpu_model

path = "~/anaconda3/envs/Drug_Response/Data/RNAi/Train_Data/"
# path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/RNAi/Train_Data/"

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 10GB of memory
    try:
        tensorflow.config.experimental.set_memory_growth(gpus[0], enable=True)
        # tensorflow.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=14336)])
        print('First GPU configured successfully.')
        tensorflow.config.experimental.set_memory_growth(gpus[1], enable=True)
        # tensorflow.config.experimental.set_virtual_device_configuration(
        #       gpus[1],
        #       [tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=14336)])
        print('Second GPU configured successfully.')
    except RuntimeError as e:
        # Virtual devices must be set at program startup
        print(e)

tensorflow.config.threading.set_intra_op_parallelism_threads(1)
tensorflow.config.threading.set_inter_op_parallelism_threads(1)
print('Parallelism threads configured successfully.')

# gpus = tensorflow.config('GPU')
# tensorflow.config.gpu.set_per_process_memory_fraction(0.85)
# tensorflow.config.experimental.set_memory_growth(True)
# tensorflow.config.gpu.set_per_process_memory_growth(True)
# tensorflow.config.experimental.set_visible_devices(gpus[1], 'GPU')


def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same', activation='relu'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1),
                        strides=(strides, 1), padding=padding, activation=activation)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x


def Conv1DBN(input_tensor, filters, kernel_size, strides=1, padding='same', activation='linear', use_bias=False, name=None):
    x = Conv1D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(input_tensor)

    if not use_bias:
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=2,
                               scale=False,
                               name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def InceptionStem(input_tensor, x_filters):
    tower_1 = Conv1DBN(input_tensor, filters=32*x_filters, kernel_size=3, strides=1, padding='same', activation='linear')

    return tower_1


def StemDecoder(input_tensor, x_filters):
    tower_1 = Conv1DBN(input_tensor, filters=32*x_filters, kernel_size=3, strides=1, padding='same', activation='linear')
    return tower_1


def InputDecoder(input_tensor):
    # 4 filters for 4 bases
    final_input = Conv1DBN(input_tensor, filters=4, kernel_size=3, strides=1, padding='same', activation='linear')
    return final_input


def InceptionDecoder(input_tensor, scale, block_type, block_idx, x_filters):
    if block_type == 'A':
        tower_1 = Conv1DBN(input_tensor, filters=16*x_filters, kernel_size=3, strides=1, padding='same', activation='linear')
        tower_1 = Conv1DBN(tower_1, filters=16*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')
        tower_2 = Conv1DBN(input_tensor, filters=16*x_filters, kernel_size=5, strides=1, padding='same', activation='linear')
        tower_2 = Conv1DBN(tower_2, filters=16*x_filters, kernel_size=5, strides=1, padding='same', activation='linear')
        tower_2 = Conv1DBN(tower_2, filters=16*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')
        tower_3 = Conv1DBN(input_tensor, filters=16*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')

        # TODO: SHOULD THE CONCATENATION ALSO BE FLIPPED?
        concat = concatenate([tower_1, tower_2, tower_3], axis=2)
    elif block_type == 'B':
        tower_1 = Conv1DBN(input_tensor, filters=32*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')
        tower_2 = Conv1DBN(input_tensor, filters=32*x_filters, kernel_size=7, strides=1, padding='same', activation='linear')
        tower_2 = Conv1DBN(tower_2, filters=32*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')
        concat = concatenate([tower_1, tower_2], axis=2)
    elif block_type == 'C':
        tower_1 = Conv1DBN(input_tensor, filters=48*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')

        tower_2 = Conv1DBN(input_tensor, filters=48*x_filters, kernel_size=3, strides=1, padding='same', activation='linear')
        tower_2 = Conv1DBN(tower_2, filters=48*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')

        concat = concatenate([tower_1, tower_2], axis=2)
        # concat = Conv1DBN(concat, filters=448, kernel_size=1, strides=1, padding='same', activation='linear')

    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "A", "B" or "C", '
                         'but got: ' + str(block_type))

    # Match number of filters with input tensor channels
    NinN = Conv1DBN(concat, filters=int(input_tensor.shape[2]), kernel_size=1, strides=1, padding='same', activation='linear')
    block_name = block_type + '_' + str(block_idx)

    # Add reduced channels to input tensor with after scaling, then relu
    added = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=[None, int(input_tensor.shape[2])],
               arguments={'scale': scale},
               name=block_name)([input_tensor, NinN])
    final = Activation(activation='relu')(added)

    return final


def InceptionV4Module(input_tensor, scale, block_type, block_idx, x_filters):
    if block_type == 'A':
        tower_1 = Conv1DBN(input_tensor, filters=16*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')
        tower_1 = Conv1DBN(tower_1, filters=16*x_filters, kernel_size=3, strides=1, padding='same', activation='linear')
        tower_2 = Conv1DBN(input_tensor, filters=16*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')
        tower_2 = Conv1DBN(tower_2, filters=16*x_filters, kernel_size=5, strides=1, padding='same', activation='linear')
        tower_2 = Conv1DBN(tower_2, filters=16*x_filters, kernel_size=5, strides=1, padding='same', activation='linear')
        tower_3 = Conv1DBN(input_tensor, filters=16*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')
        concat = concatenate([tower_1, tower_2, tower_3], axis=2)
    elif block_type == 'B':
        tower_1 = Conv1DBN(input_tensor, filters=32*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')
        tower_2 = Conv1DBN(input_tensor, filters=32*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')
        tower_2 = Conv1DBN(tower_2, filters=32*x_filters, kernel_size=7, strides=1, padding='same', activation='linear')
        concat = concatenate([tower_1, tower_2], axis=2)
    elif block_type == 'C':
        tower_1 = Conv1DBN(input_tensor, filters=48*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')

        tower_2 = Conv1DBN(input_tensor, filters=48*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')
        tower_2 = Conv1DBN(tower_2, filters=48*x_filters, kernel_size=3, strides=1, padding='same', activation='linear')

        concat = concatenate([tower_1, tower_2], axis=2)
        # concat = Conv1DBN(concat, filters=448, kernel_size=1, strides=1, padding='same', activation='linear')
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "A", "B" or "C", '
                         'but got: ' + str(block_type))

    # Match number of filters with input tensor channels, with a network in a network module
    NinN = Conv1DBN(concat, filters=int(input_tensor.shape[2]), kernel_size=1, strides=1, padding='same', activation='linear')
    block_name = block_type + '_' + str(block_idx)

    # Add reduced channels to input tensor with after scaling, then relu
    added = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=[None, int(input_tensor.shape[2])],
               arguments={'scale': scale},
               name=block_name)([input_tensor, NinN])
    final = Activation(activation='relu')(added)

    return final


def ReductionDecoder(input_tensor, block_type, block_idx, x_filters):
    block_name = block_type + '_' + str(block_idx)
    if block_type == 'A':
        # Match the number of filters, strides, kernel size and padding with the reduction layer blocks
        tower_1 = Conv1DTranspose(input_tensor, filters=int(input_tensor.shape[2]), kernel_size=2, strides=2, padding='same',
                                  activation='linear')
        tower_2 = Conv1DTranspose(input_tensor, filters=16*x_filters, kernel_size=3, strides=2, padding='same',
                                  activation='linear')
        tower_3 = Conv1DTranspose(input_tensor, filters=16*x_filters, kernel_size=3, strides=2, padding='same', activation='linear')
        tower_3 = Conv1DBN(tower_3, filters=16*x_filters, kernel_size=3, strides=1, padding='same', activation='linear')
        # Down-sample the channels (?)
        tower_3 = Conv1DBN(tower_3, filters=8*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')

        concat = concatenate([tower_1, tower_2, tower_3], axis=2, name=block_name)

    elif block_type == 'B':
        tower_1 = Conv1DTranspose(input_tensor, filters=int(input_tensor.shape[2]), kernel_size=2, strides=2, padding='same',
                                  activation='linear')
        # TODO FIX too many transposed convolutions
        tower_2 = Conv1DTranspose(input_tensor, filters=48*x_filters, kernel_size=3, strides=2, padding='same', activation='linear')
        tower_2 = Conv1DTranspose(tower_2, filters=32*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')

        tower_3 = Conv1DTranspose(input_tensor, filters=32*x_filters, kernel_size=3, strides=2, padding='same', activation='linear')
        tower_3 = Conv1DTranspose(tower_3, filters=32*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')

        tower_4 = Conv1DTranspose(input_tensor, filters=32*x_filters, kernel_size=3, strides=2, padding='same', activation='linear')
        tower_4 = Conv1DTranspose(tower_4, filters=32*x_filters, kernel_size=3, strides=1, padding='same', activation='linear')
        tower_4 = Conv1DBN(tower_4, filters=32*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')

        concat = concatenate([tower_1, tower_2, tower_3, tower_4], axis=2, name=block_name)

    else:
        raise ValueError('Unknown Inception-ResNet reduction block type. '
                         'Expects "A" or "B"'
                         'but got: ' + str(block_type))
    return concat


def ReductionModule(input_tensor, block_type, block_idx, x_filters):
    block_name = block_type + '_' + str(block_idx)
    if block_type == 'A':
        tower_1 = MaxPooling1D(pool_size=2, data_format='channels_last')(input_tensor)

        tower_2 = Conv1DBN(input_tensor, filters=16*x_filters, kernel_size=3, strides=2, padding='same', activation='linear')

        tower_3 = Conv1DBN(input_tensor, filters=8*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')
        tower_3 = Conv1DBN(tower_3, filters=16*x_filters, kernel_size=3, strides=1, padding='same', activation='linear')
        tower_3 = Conv1DBN(tower_3, filters=16*x_filters, kernel_size=3, strides=2, padding='same', activation='linear')

        concat = concatenate([tower_1, tower_2, tower_3], axis=2, name=block_name)
    elif block_type == 'B':
        tower_1 = MaxPooling1D(pool_size=2, data_format='channels_last')(input_tensor)

        tower_2 = Conv1DBN(input_tensor, filters=32*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')
        tower_2 = Conv1DBN(tower_2, filters=48*x_filters, kernel_size=3, strides=2, padding='same', activation='linear')

        tower_3 = Conv1DBN(input_tensor, filters=32*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')
        tower_3 = Conv1DBN(tower_3, filters=32*x_filters, kernel_size=3, strides=2, padding='same', activation='linear')

        tower_4 = Conv1DBN(input_tensor, filters=32*x_filters, kernel_size=1, strides=1, padding='same', activation='linear')
        tower_4 = Conv1DBN(tower_4, filters=32*x_filters, kernel_size=3, strides=1, padding='same', activation='linear')
        tower_4 = Conv1DBN(tower_4, filters=32*x_filters, kernel_size=3, strides=2, padding='same', activation='linear')

        concat = concatenate([tower_1, tower_2, tower_3, tower_4], axis=2, name=block_name)
    else:
        raise ValueError('Unknown Inception-ResNet reduction block type. '
                         'Expects "A" or "B"'
                         'but got: ' + str(block_type))
    return concat


def InceptionBlock(input_tensor, block_idx, x_filters):
    # BlockA
    Inc_A = InceptionV4Module(input_tensor, scale=0.3, block_type='A', block_idx=block_idx, x_filters=x_filters)
    # Reduction A
    Red_A = ReductionModule(Inc_A, block_type='A', block_idx='Reduction_'+block_idx, x_filters=x_filters)
    # BlockB
    Inc_B = InceptionV4Module(Red_A, scale=0.3, block_type='B', block_idx=block_idx, x_filters=x_filters)
    # Reduction B
    Red_B = ReductionModule(Inc_B, block_type='B', block_idx='Reduction_'+block_idx, x_filters=x_filters)
    # BlockC
    Inc_C = InceptionV4Module(Red_B, scale=0.3, block_type='C', block_idx=block_idx, x_filters=x_filters)
    # Another Reduction B
    # Red_B_2 = ReductionModule(Inc_C, block_type='B', block_idx='Reduction_B2_'+block_idx)

    return Inc_C


def DecoderBlock(input_tensor, block_idx, x_filters):
    # Transposed Reduction B
    # T_Red_B_2 = ReductionDecoder(input_tensor, block_type='B', block_idx='Reduction_Decoder_B2_'+block_idx)
    # Transposed Module C
    T_C = InceptionDecoder(input_tensor, scale=0.3, block_type='C', block_idx='Decoder_'+block_idx, x_filters=x_filters)
    # Transposed Reduction B
    T_Red_B = ReductionDecoder(T_C, block_type='B', block_idx='Reduction_Decoder_'+block_idx, x_filters=x_filters)
    # Transposed Module B
    T_B = InceptionDecoder(T_Red_B, scale=0.3, block_type='B', block_idx='Decoder_'+block_idx, x_filters=x_filters)
    # Transposed Reduction A
    T_Red_A = ReductionDecoder(T_B, block_type='A', block_idx='Reduction_Decoder_'+block_idx, x_filters=x_filters)
    # Transposed Module A
    T_A = InceptionDecoder(T_Red_A, scale=0.3, block_type='A', block_idx='Decoder_'+block_idx, x_filters=x_filters)

    return T_A


def create_model(x_filters):
    print('Defining model...')

    cdna_input = Input(shape=(None, 4))
    # Stem
    stem = InceptionStem(cdna_input, x_filters=x_filters)

    Block_1_Result = InceptionBlock(stem, block_idx='1', x_filters=x_filters)

    # Bottleneck layer to be used as input to other models
    encoder_result = Conv1DBN(Block_1_Result, filters=4, kernel_size=1, strides=1, padding='same', activation='linear')

    # Red_B = ReductionModule(Block_1_Result, block_type='B', block_idx='Reduction_'+'2')

    # Block_2_Result = InceptionBlock(Block_1_Result, block_idx='2')

    Decoder_1_Result = DecoderBlock(encoder_result, block_idx='1', x_filters=x_filters)
    # T_Red_B = ReductionDecoder(Decoder_1_Result, block_type='B', block_idx='Reduction_Decoder_'+'2')

    # Decoder_2_Result = DecoderBlock(Decoder_1_Result, block_idx='2')

    T_Stem = StemDecoder(Decoder_1_Result, x_filters=x_filters)

    T_Input = InputDecoder(T_Stem)

    Inc_Model = Model(cdna_input, T_Input)

    # run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    # Inc_Model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy')

    Inc_Model.summary()

    return Inc_Model


def extract_layers(model, layer_name):
    output_layer = cur_model.get_layer(layer_name)
    model_subset = Model(model.input, output_layer.output)
    for i, l in enumerate(model_subset.layers):
        l.trainable = False
        l._name = 'FCAE_Encoder_'+str(i)
    # fcae_encoder = K.function([gene_fcae_model.layers[0].input, K.learning_phase()],
#                               [gene_fcae_model.layers[3].output])
    return model_subset


class ThreadSafeIter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):  # Py3
        with self.lock:
            return next(self.it)

    def next(self):
        with self.lock:
            return self.it.next()


def ThreadSafeGenerator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))
    return g

# path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/RNAi/Train_Data/"
# transcript_path = 'all_transcript_seqs.txt'
# cur_transcripts = pd.read_csv(path+transcript_path, engine='c', sep=',', nrows=100)
def dna_to_onehot(seqs_list):
    base_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0],
             'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    list_alt = [list(x) for x in seqs_list]
    # Use a dictionary to perform multiple replacements at once
    # ~4x faster than using list comprehension in succession
    encoded = [[base_dict.get(x, x) for x in y] for y in list_alt]
    # t1 = [[[1,0,0,0] if x=='A' else x for x in y] for y in list_alt]
    # t2 = [[[0,1,0,0] if x=='T' else x for x in y] for y in t1]
    # t3 = [[[0,0,1,0] if x=='C' else x for x in y] for y in t2]
    # t4 = [[[0,0,0,1] if x=='G' else x for x in y] for y in t3]
    one_hot_alt = [np.array(x).reshape((1, len(x), 4)) for x in encoded]
    return one_hot_alt


def pad_for_divisibility(onehot_seqs, by=8):
    # all_lengths = np.unique([x.shape[1] for x in onehot_seqs])
    # Group them by the same length
    # for length in all_lengths:
    #     cur_sub = np.array([x for x in onehot_seqs if x.shape[1] == length])

    onehot_seqs = onehot_seqs.reshape(onehot_seqs.shape[0], onehot_seqs.shape[2], 4)
    assert not np.isnan(np.sum(onehot_seqs))
    # for i in range(len(one_hot_alt)):
    if by == 8:
        # Detect odd length input
        if (onehot_seqs.shape[1] & 1) == 1:
            # Duplicate last base or just add a zero vector
            # x_train[i][len(x_train[i])-1]
            onehot_seqs = np.concatenate((onehot_seqs,
                                     np.tile(np.array([0,0,0,0]),
                                             (onehot_seqs.shape[0], 1)).reshape((onehot_seqs.shape[0], 1, 4))), axis=1)
            # Since we're doing 3 transposed convolutions, we're dividing by 48, so must add 2 more
        if onehot_seqs.shape[1] % 4 == 2:
            onehot_seqs = np.concatenate((onehot_seqs,
                                      np.tile(np.array([0,0,0,0]),
                                              (onehot_seqs.shape[0]*2, 1)).reshape((onehot_seqs.shape[0], 2, 4))), axis=1)
        if onehot_seqs.shape[1] % 8 == 4:
            onehot_seqs = np.concatenate((onehot_seqs,
                                      np.tile(np.array([0,0,0,0]),
                                              (onehot_seqs.shape[0]*4, 1)).reshape((onehot_seqs.shape[0], 4, 4))), axis=1)
    else:
        raise Exception('Padding for other denominators not yet implemented!')

    return onehot_seqs


def encode_all_transcripts(encoder_model, transcript_path, save_path, batch_overall_size):
    start = time.time()
    cur_transcripts = pd.read_csv(path+transcript_path, engine='c', sep=',')
    print("Took", round(time.time() - start, 2), "seconds to read file")
    cur_alt_lists = cur_transcripts['transcript'].tolist()
    all_lengths = np.unique([len(x) for x in cur_alt_lists])

    all_encoded = []
    for length in all_lengths:
        cur_sub = np.array([x for x in cur_alt_lists if len(x) == length])
        # Convert LIST of dna sequences to their onehot encoded form
        cur_sub = dna_to_onehot(cur_sub)
        cur_sub = np.array(cur_sub)
        cur_sub = pad_for_divisibility(cur_sub, by=8)

        cur_size = np.prod(cur_sub.shape)
        if cur_size > batch_overall_size:
            num_parts = np.int(np.ceil(cur_size/batch_overall_size))
            advance = np.int(np.ceil(cur_sub.shape[0]/num_parts))
            for i in range(0, cur_sub.shape[0], advance):
                # This automatically ensures only valid indices are kept,
                # so num_parts can be larger than what remains
                cur_sub_sub = cur_sub[i:i+advance, :]
                cur_encoded = encoder_model.predict_on_batch(cur_sub_sub)
                all_encoded.append(cur_encoded)

        else:
            cur_encoded = encoder_model.predict_on_batch(cur_sub)
            all_encoded.append(cur_encoded)
        print('Finished predicting on batch of length', length)

    with open(save_path, 'wb') as fp:
        pickle.dump(all_encoded, fp)

# make_example(cur_encoded, 1)
# def make_example(features, image_id):
#
#     ex = tensorflow.train.SequenceExample()
#     ex.context.feature['image_id'].int64_list.value.append(image_id)
#     fl_features = ex.feature_lists.feature_list['features/fc8']
#
#     for feature in features:
#         fl_features.feature.add().bytes_list.value.append(feature.tostring())
#     return ex
#
#
# def get_writable( arr ):
#     """
#         this function returns a serialized string
#         for input array of integers
#         arr : input array
#     """
#     arr = tensorflow.train.Int64List( value = arr)
#     arr = tensorflow.train.Feature(int64_list= arr )
#     arr = tensorflow.train.Features(feature =  { 'seqs': arr})
#     arr = tensorflow.train.Example( features = arr)
#     return arr.SerializeToString()
#
# filename = "./test2.tfrecords"
#         writer = tensorflow.io.TFRecordWriter(filename)
#
#         #writing 3 different sized arrays
#         writer.write( get_writable(cur_encoded))
#         writer.close()
def my_pred_generator(transcript_path, chunk_size):
        while True:  # keras requires all generators to be infinite
            for cur_transcripts in pd.read_csv(path+transcript_path, engine='c', sep=',', chunksize=chunk_size):
                cur_alt_lists = cur_transcripts['transcript'].tolist()
                all_lengths = np.unique([len(x) for x in cur_alt_lists])
                for length in all_lengths:
                    cur_sub = np.array([x for x in cur_alt_lists if len(x) == length])
                    # Convert LIST of dna sequences to their onehot encoded form
                    cur_sub = dna_to_onehot(cur_sub)
                    cur_sub = pad_for_divisibility(cur_sub, by=8)

                    yield cur_sub


def transcript_generator(transcript_path, chunk_size, per_yield_size):
    # while True:  # keras requires all generators to be infinite
        # base_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0],
        #              'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    for cur_transcripts in pd.read_csv(path+transcript_path, engine='c', sep=',', chunksize=chunk_size):
        cur_alt_lists = cur_transcripts['transcript'].tolist()
        # Convert LIST of dna sequences to their onehot encoded form
        cur_onehot = dna_to_onehot(cur_alt_lists)

        # Group sequences by the same length as batches need to be of the same size
        all_lengths = np.unique([x.shape[1] for x in cur_onehot])
        for length in all_lengths:
            cur_sub = np.array([x for x in cur_onehot if x.shape[1] == length])

            cur_sub = pad_for_divisibility(cur_sub, by=8)

            # Limit the overall size of each released batch to fit on the VRAM
            cur_size = np.prod(cur_sub.shape)
            if cur_size > per_yield_size:
                num_parts = np.int(np.ceil(cur_size/per_yield_size))
                advance = np.int(np.ceil(cur_sub.shape[0]/num_parts))
                for i in range(0, cur_sub.shape[0], advance):
                    # This automatically ensures only valid indices are kept,
                    # so num_parts can be larger than what remains
                    cur_train = cur_sub[i:i+advance, :]
                    yield cur_train, cur_train
            else:
                yield (cur_sub, cur_sub)

        # Garbage collection to reduce possibility of memory leak (?)
        gc.collect()

# @ThreadSafeGenerator
class my_generator(Sequence):
    def __init__(self, transcript_path, chunk_size, per_yield_size):
        self.transcript_path, self.chunk_size = transcript_path, chunk_size
        self.per_yield_size = per_yield_size
        self.transcript_generator = transcript_generator(self.transcript_path, self.chunk_size, self.per_yield_size)

    def __len__(self):
        return np.int(np.ceil(self.chunk_size / float(self.per_yield_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = next(self.transcript_generator)

        return batch_x, batch_y


if __name__ == '__main__':
    K.clear_session()
    cur_chunk_size = sys.argv[1]
    num_epochs = sys.argv[2]
    action = sys.argv[3]
    checkpoint_callback = ModelCheckpoint('inception_cae.h5', monitor='loss', verbose=1, save_best_only=False,
                                          save_weights_only=False, mode='auto', period=1)

    # NOTE: Have to use 0 workers while using multiprocessing with the generators when submitting jobs or using qrsh
    if action == 'resume':
        cur_model = load_model('inception_cae.h5')
        print('Loaded model successfully.')
        cur_model.fit_generator(generator=my_generator(transcript_path='all_transcript_seqs.txt',
                                                       chunk_size=int(cur_chunk_size), per_yield_size=1e5),
                                steps_per_epoch=22500, epochs=int(num_epochs), use_multiprocessing=True, workers=0,
                                verbose=True,
                                max_queue_size=50, callbacks=[checkpoint_callback])
        cur_model.save('inception_cae.h5')

    elif action == 'encode':
        cur_model = load_model('inception_cae.h5')
        encoder_model = extract_layers(cur_model, layer_name='C_1')

        encode_all_transcripts(encoder_model=encoder_model, transcript_path='all_transcript_seqs.txt',
                               save_path='encoded_transcripts', batch_overall_size=5e5)

    else:
        cur_model = create_model(x_filters=2)

        # parallel_model = multi_gpu_model(cur_model, gpus=2, cpu_merge=True, cpu_relocation=False)
        cur_model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy')

        # plot_model(cur_model, to_file='inception_cae.png', show_shapes=True, show_layer_names=True)
        cur_model.fit_generator(generator=my_generator(transcript_path='all_transcript_seqs.txt',
                                                       chunk_size=int(cur_chunk_size), per_yield_size=1e5),
                                steps_per_epoch=22500, epochs=int(num_epochs), use_multiprocessing=True, workers=0,
                                verbose=True,
                                max_queue_size=50, callbacks=[checkpoint_callback])

        cur_model.save('inception_cae.h5')



