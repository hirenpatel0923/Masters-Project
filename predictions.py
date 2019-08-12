from random import seed

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input
from keras.utils.data_utils import OrderedEnqueuer
from keras.callbacks import TerminateOnNaN, ModelCheckpoint
from keras.optimizers import RMSprop, Adam
from keras.losses import mse, binary_crossentropy
from keras.layers import LeakyReLU

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# importing networks and it's parameters
from network_params import *
from functions import *

# importing dataloader and datagenerator
from data_preprocessing.data_params import *
from data_preprocessing.data_loader_pred import *
from data_preprocessing.data_generator_rgb_pred import *

from encoder_network import EncoderNetwork
from decoder_network import DecoderNetwork
from discrimintor_network import DiscriminatorNetwork


s = 23
seed(s)

epochs = 50
version = 'v-01-epoch-'+str(epochs)


#gan params
networkParams = NetworkParams()

#getting Data
dataParams = DataParams()
dataLoader = DataLoader()


def get_au_list():
    temp_au_lst = [str(int(au.replace('AU', ''))) for au in dataParams.allOccAUs_BP]
    occ_col_lst = []

    for col in df.columns:
        if col in temp_au_lst and col != '0':
            occ_col_lst.append(col)

    return occ_col_lst



print('Getting images and AU_OCC information.......')
df = dataLoader.load_data(dataParams.testSubjects_BP, dataParams.allTasks_BP)



encoder_input = Input(shape=(networkParams.modifiedHeight, networkParams.modifiedWidth, networkParams.numChannels))
au_input = Input(shape=(12,))
fm_input = Input(shape=(1,))


encoderModel, flat_shape_before, flat_shape_after = EncoderNetwork().buildModel()
decoderModel = DecoderNetwork().buildModel(flat_shape_before, flat_shape_after)
discriminatorModel = DiscriminatorNetwork().buildModel()

vae_output = decoderModel([encoderModel(encoder_input), au_input, fm_input])

VAE = Model([encoder_input, au_input, fm_input], vae_output)

filepath = 'models/VAE-'+version+'.h5'

VAE.load_weights(filepath)

for pred_batch in range(len(dataParams.testSubjects_BP)):
    temp_df = df.groupby(['subject','task']).apply(lambda x: x.sample(1)).reset_index(drop=True)
    filename = 'prediction/VAE-prediction-' + str(pred_batch)
    predict_results(VAE, 
                    (temp_df['subject'].values, temp_df['task'].values, temp_df['0'].values, temp_df['path'].values, temp_df[get_au_list()].values, temp_df['MF'].values, len(dataParams.testSubjects_BP), len(dataParams.allTasks_BP)), 
                    filename)


filepath = 'models/VAE_after-'+version+'.h5'

VAE.load_weights(filepath)

for pred_batch in range(len(dataParams.testSubjects_BP)):
    temp_df = df.groupby(['subject','task']).apply(lambda x: x.sample(1)).reset_index(drop=True)
    filename = 'prediction/VAE_after-prediction-' + str(pred_batch)
    predict_results(VAE, 
                    (temp_df['subject'].values, temp_df['task'].values, temp_df['0'].values, temp_df['path'].values, temp_df[get_au_list()].values, temp_df['MF'].values, len(dataParams.testSubjects_BP), len(dataParams.allTasks_BP)), 
                    filename)
