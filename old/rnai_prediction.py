import pandas as pd
import numpy as np
import time

from tensorflow.python.keras.layers import Input, GRU, Dense, Conv1D,\
    BatchNormalization, Flatten, RepeatVector, concatenate, MaxPooling1D,\
    UpSampling1D, GaussianNoise, Embedding, Conv2DTranspose, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
import tensorflow.python.keras.backend as K

from tensorflow.python.keras.utils import plot_model
import numpy.lib.recfunctions


path = "Data/RNAi/Train_Data/"


def Conv1DTranspose(input_tensor, filters, kernel_size, activation, name=None, strides=2, padding='valid'):
    """
    Define a 1D deconvolution layer
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1),
                        padding=padding, activation=activation)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2), name=name)(x)
    return x


def reconstuction_loss(y_true, y_pred):
    """
    Define the reconstruction lost
    """
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


# Create the model
h_SIZE = 17
# ShRNA Autoencoder
SH_Input = Input(shape=(21, ), name='shRNA_Input')
SH_Embedding = Embedding(output_dim=8, input_dim=4+1,
                         input_length=21, name='shRNA_Embedding')(SH_Input)
SH = Conv1D(filters=200, kernel_size=4, padding='valid', activation='relu')(SH_Embedding)
SH = BatchNormalization()(SH)
SH = Conv1D(filters=300, kernel_size=2, padding='valid', activation='relu')(SH)
SH = BatchNormalization()(SH)

# As a trick, use a convolutional layer with filter size equal to
# sh_encoded = Conv1D(filters=1, kernel_size=h_SIZE)(SH)
# SH = MaxPooling1D(pool_size=2, name='shRNA_Embedding')(SH)
# sh_encoded = Flatten()(SH)

DCNN1 = Conv1DTranspose(input_tensor=sh_encoded, filters=1, kernel_size=h_SIZE, strides=1,
                        activation='relu')
DCNN2 = Conv1DTranspose(input_tensor=DCNN1, filters=1, kernel_size=2, strides=1,
                        activation='relu')

reconstruction_output = Conv1DTranspose(input_tensor=DCNN2, filters=1, kernel_size=4, strides=1,
                                        activation='relu', name='reconstruction_output')

sh_model = Model(inputs=SH_Input, outputs=reconstruction_output)
sh_model.compile(optimizer=Adam(lr=0.001),
                 loss='cosine_proximity')

sh_model.summary()
plot_model(sh_model, show_shapes=True)

sh_model.fit(x=cur_shrna_train_X, y=cur_shrna_train_y,
             batch_size=32, epochs=5, verbose=1)
temp = cur_shrna_train_X[0, :].reshape((21, )).shape
sh_model.predict(x=cur_shrna_train_X)
temp=cur_shrna_train_y[0,:,:]


cur_shrna_train_X = cur_shrna_train.reshape((cur_shrna_train.shape[0], cur_shrna_train.shape[1]))
cur_shrna_train_y = cur_shrna_train.reshape((cur_shrna_train.shape[0], cur_shrna_train.shape[1], 1))
cur_shrna_train_X.shape, cur_shrna_train_y.shape


SH_Input = Input(shape=(21, ), name='shRNA_Input')
SH_Embedding = Embedding(output_dim=4, input_dim=4+1,
                         input_length=21, name='shRNA_Embedding')(SH_Input)
SH = Conv1D(filters=64, kernel_size=4, padding='valid', activation='relu')(SH_Embedding)
SH = BatchNormalization()(SH)
SH = Conv1D(filters=32, kernel_size=2, padding='valid', activation='relu')(SH)
SH = BatchNormalization()(SH)

sh_encoded = Flatten()(SH)
SH = RepeatVector(21)(sh_encoded)
SH = GRU(units=128, activation='tanh', return_sequences=True)(SH)
SH = GRU(units=128, activation='tanh', return_sequences=True)(SH)
SH_Output = GRU(units=128, activation='softmax', return_sequences=True, name='shRNA_Output')(SH)

sh_model = Model(inputs=SH_Input, outputs=SH_Output)
sh_model.compile(optimizer=Adam(lr=0.001),
                 loss='sparse_categorical_crossentropy')
#
# sh_model.summary()
# plot_model(sh_model, show_shapes=True, to_file='gru_model.png')
#
sh_model.fit(x=cur_shrna_train, y=cur_shrna_train, epochs=1, batch_size=64)
#
# pred = sh_model.predict(cur_shrna_train[0:1])
# pred.argmax(axis=2)
# SH = Conv1D(filters=32, kernel_size=4, padding='valid', activation='relu')(sh_encoded)
# SH = UpSampling1D(size=2)(SH)
# SH = BatchNormalization()(SH)
# SH = Conv1D(filters=64, kernel_size=2, padding='valid', activation='relu')(SH)
# SH = BatchNormalization()(SH)
# SH = UpSampling1D(size=2)(SH)
# sh_decoded = Conv1D(filters=1, kernel_size=2, padding='same', activation='relu', name='shRNA_Output')(SH)

# Expression Autoencoder
EXP_Input = Input(shape=(57820, ), name='EXP_Input')
EXP = GaussianNoise(stddev=1)(EXP_Input)
EXP = Dense(units=2048, activation='relu')(EXP)
EXP = BatchNormalization()(EXP)
EXP = Dense(units=512, activation='relu')(EXP)
EXP = BatchNormalization()(EXP)
EXP = Dense(units=128, activation='relu')(EXP)
exp_encoded = BatchNormalization(name='Exp_Embedding')(EXP)

EXP = Dense(units=512, activation='relu')(exp_encoded)
EXP = BatchNormalization()(EXP)
EXP = Dense(units=2048, activation='relu')(EXP)
EXP = BatchNormalization()(EXP)
EXP_Output = Dense(units=57820, activation='relu', name='EXP_Output')(EXP)

TUM = Dense(64, activation='relu', name='TUM_Input')(exp_encoded)
TUM = BatchNormalization()(TUM)
TUM_Output = Dense(33, activation='softmax', name='TUM_Output')

exp_model = Model(inputs=EXP_Input, outputs=[EXP_Output, TUM_Output])

exp_model.compile(optimizer=Adam(lr=0.001),
                  loss={'EXP_Output': 'mse',
                        'TUM_Output': 'sparse_categorical_cross_entropy'})

merged = concatenate([sh_encoded, exp_encoded])

# LFC Predictor
LFC = Dense(units=256, activation='linear', name='LFC_Input')(merged)
LFC = BatchNormalization()(LFC)
LFC = Dense(units=128, activation='linear')(LFC)
LFC = BatchNormalization()(LFC)
LFC_Output = Dense(units=1, activation='linear', name='LFC_Output')(LFC)

model = Model(inputs=[SH_Input, EXP_Input], outputs=[SH_Output, EXP_Output, LFC_Output])

model.compile(optimizer=Adam(lr=0.001),
              loss={'shRNA_Output': 'sparse_categorical_crossentropy',
                    'Exp_Output': 'mse',
                    'LFC_Output': 'mse'})
model.summary()
plot_model(model, show_shapes=True, to_file='sh_exp_model.png')

# Read Mutational Data
# type_dict = dict(zip((np.arange(15) + 1).astype('str'), ["int8"]*15))
# long_ccle_mut = pd.read_csv(path+"/labeled_ccle_mut_data.txt", engine='c', sep=',', dtype={"Variant_Labels": "int8"})
#
# start = time.time()
# dcast = long_ccle_mut.pivot_table(index=["cell_line"],
#                                   columns=["Hugo_Symbol", "Variant_Labels"],
#                                   aggfunc=len, fill_value=0, dropna=False)
# dcast.columns = ["{0}_{1}".format(l1, l2) for l1, l2 in dcast.columns]
# mut_int8_readtime = time.time() - start
#
# dcast.shape
# dcast.memory_usage(index=True).sum()
# np_dcast = dcast.to_records()
#
# np_dcast['cell_line']
# np_dcast['A1BG_1']
# np.lib.recfunctions.join_by(key="cell_line", np_dcast, np_sub_ach)
# ccle_mut.memory_usage(index=True).sum()

# Create the type dictionary for genes once
# gene_names = pd.read_csv(path+"/achilles_exp.txt", engine='c', sep=',', nrows=1).columns[1:]
# type_dict = dict(zip(list(gene_names), ["float16"]*len(gene_names)))

cur_tokenizer = Tokenizer(num_words=5, filters="", lower=False, split=' ')
cur_tokenizer.fit_on_texts('ATCG ')

# Read shRNA Data
# Have more shRNAs than cell lines, thus, for each shRNA, we get the corresponding cell line and its associated data
achilles = pd.read_csv(path+"/achilles_shrna_cell_lfc.txt", engine='c', sep=',')

cur_exp_path = "achilles_exp.txt"
cur_exp = pd.read_csv(path+cur_exp_path, engine='c', sep=',', nrows=1)
cur_shrna_data = pd.read_csv(path+"achilles_shrna_cell_lfc.txt")

def my_generator(cur_shrna_data, cur_exp_path, batch_size):
    while True:  # keras requires all generators to be infinite
        for cur_exp in pd.read_csv(path+cur_exp_path, engine='c', sep=',', chunksize=1):
            # Subset shRNA data based on the current cell line
            shrna_sub = cur_shrna_data[cur_shrna_data['cell_line'] == cur_exp['cell_line'][0]]
            cur_lfc_train = shrna_sub['lfc'].values

            cur_shrna_lists = [' '.join(list(x)) for x in shrna_sub['shRNA'].tolist()]

            cur_shrna_lists_seqs = cur_tokenizer.texts_to_sequences(cur_shrna_lists)
            cur_shrna_train = np.array(cur_shrna_lists_seqs)
            cur_shrna_train = cur_shrna_train.reshape(cur_shrna_train.shape[0], 21)
            # Treat characters as labels...
            # cur_shrna_train = cur_shrna_train - 1
            # Expression values for current cell line
            np_exp = cur_exp.iloc[:, 1:].values
            # Pass matched expression and shRNA data
            for i in range(0, cur_shrna_train.shape[0], batch_size):
                x_shrna = cur_shrna_train[i:i+batch_size, :]
                y_lfc = cur_lfc_train[i:i+batch_size]
                x_exp = np.tile(np_exp, (x_shrna.shape[0], 1))
                # x_shrna.shape, y_lfc.shape, x_exp.shape
                # Pass shRNA and Exp as both X and y to autoencoders, LFC to LFC model
                yield ({'shRNA_Input': x_shrna, 'Exp_Input': x_exp},
                       {'shRNA_Output': x_shrna, 'Exp_Output': x_exp, 'LFC_Output': y_lfc})
                # cur_exp_train = np.tile(np_exp, (cur_shrna_train.shape[0], 1))
                # cur_exp_train = cur_exp_train.reshape(cur_exp_train.shape[0], 57820)


# model.fit_gen(x={"shRNA_Input": cur_shrna_train, "Exp_Input": cur_exp_train},
#           y={"shRNA_Output": cur_shrna_train, "Exp_Output": cur_exp_train,
#              'LFC_Output': cur_lfc_train},
#           epochs=1, verbose=1, batch_size=128, validation_split=0.1)
model.fit_generator(generator=my_generator(cur_shrna_data=achilles, cur_exp_path='achilles_exp.txt', batch_size=16),
                    steps_per_epoch=int(np.ceil(achilles.shape[0] / 16)), epochs=1, verbose=1, use_multiprocessing=True)
# Read expression data in chunks
# for cur_exp in pd.read_csv(path+"/achilles_exp.txt", engine='c', sep=',', chunksize=1):
#
#     # Subset shRNA data based on the current cell line
#     ach_sub = achilles[achilles['cell_line'] == cur_exp['cell_line'][0]]
#
#     # start = time.time()
#     cur_shrna_lists = [' '.join(list(x)) for x in ach_sub['shRNA'].tolist()]
#
#     cur_shrna_lists_seqs = cur_tokenizer.texts_to_sequences(cur_shrna_lists)
#     cur_shrna_train = np.array(cur_shrna_lists_seqs)
#     cur_shrna_train = cur_shrna_train.reshape(cur_shrna_train.shape[0], 21, 1)
#
#     # Duplicate expression data as necessary
#     np_exp = cur_exp.iloc[:, 1:].values
#     cur_exp_train = np.tile(np_exp, (cur_shrna_train.shape[0], 1))
#     cur_exp_train = cur_exp_train.reshape(cur_exp_train.shape[0], 57820)
#
#     cur_lfc_train = ach_sub['lfc'].values
#     # to_seq_time = time.time() - start

# model.fit(x={"shRNA_Input": cur_shrna_train, "Exp_Input": cur_exp_train},
#           y={"shRNA_Output": cur_shrna_train, "Exp_Output": cur_exp_train,
#              'LFC_Output': cur_lfc_train},
#           epochs=1, verbose=1, batch_size=128, validation_split=0.1)



# ccle_exp = pd.read_csv(path+"/ccle_exp_data.txt", engine='c', sep=',', dtype=type_dict)

# start = time.time()
# ccle_exp = pd.read_csv(path+"/ccle_long_exp.txt", engine='c', sep=',', dtype={"exp": "float16"})
# exp_float16_readtime = time.time() - start


# exp_genes = ccle_exp.columns[1:]
# ccle_exp.set_index("cell_line", inplace=True)
# ccle_exp = ccle_exp.reset_index()
# np_exp = ccle_exp.to_records()
# np_exp = np_exp.view(np.recarray)
# np.rec.from
# np_exp.nbytes

# ccle_exp.memory_usage(index=True).sum()
# temp = ccle_exp.astype(type_dict)

# np_temp = temp.to_records()

# temp.memory_usage(index=True).sum()
# ccle_exp.iloc[:, 1]
# temp.iloc[1, 1]

# le = LabelEncoder()
# le.fit(ccle_mut['Variant_Classification'])
# ccle_mut['label'] = le.transform(ccle_mut["Variant_Classification"])

# mut_np = ccle_mut[["Hugo_Symbol", "cell_name", "label"]].values

# achilles.set_index("cell_line", inplace=True)
# achilles = achilles.reset_index()
np_ach = achilles.to_records()

np_exp['cell_line']

# Randomly sample 1000 rows from achilles data
cur_idx = np.random.choice(achilles.shape[0], 10000)
cur_sub = achilles.iloc[cur_idx, :]

# cur_sub = cur_sub[cur_sub[:, 1].argsort()]

np_sub_ach = cur_sub.iloc[:, 1].to_frame().to_records()
np_sub_ach['cell_line']

# First, subset the expression data
exp_sub = np_exp[np.isin(np_exp['cell_line'], np_sub_ach['cell_line'])]

start = time.time()
np_joined = np.lib.recfunctions.join_by("cell_line", np_exp, np_sub_ach, "inner")
float16_total = time.time() - start

start = time.time()
np_joined = np.lib.recfunctions.join_by("cell_line", np_temp, np_sub_ach, "inner")
float16_total = time.time() - start



cur_merged = pd.merge(cur_sub, ccle_exp, on="cell_line", how="inner")

# Get CCLE Expression data for these cell lines
exp_sub = ccle_exp[np.isin(ccle_exp[:, 0], cur_sub[:, 1]), :]
exp_sub = exp_sub[exp_sub[:, 0].argsort()]
match = lambda a, b: [ b.index(x)+1 if x in b else None for x in a ]
match(cur_sub[:, 1], ccle_exp[:, 0])

cur_sub[:, 1] == exp_sub[:, 0]

exp_sub.shape
ccle_exp.shape
np.select(ccle_exp[:, 0] in cur_sub[:, 1])
ccle_exp[ccle_exp[0, :].isin(cur_sub[:, 1])]
