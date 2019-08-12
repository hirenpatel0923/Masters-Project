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
#seed(s)

epochs = 50
version = 'v-01-epoch-'+str(epochs)

AUs = '1,2,4,6,7,10,12,14,15,17,23,24'.split(',')

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
filename = 'VAE_'+version + '--' + str(epochs)

VAE.load_weights('models/VAE_after-RAUFMv-01-epoch-50.h5')

samples = 100

for numsample in range(1, samples + 1):
    temp_df = df.sample()
    df = df.drop(temp_df.index)
    df_repeted = pd.concat([temp_df]*(len(AUs)+2), ignore_index=True)
    
    au_index = 0
    for index, row in df_repeted.iterrows():
        if index == 0:
            pass
        elif index == 1:
            val = row['MF']
            if val == 0:
                df_repeted.set_value(index, 'MF', 1)
            else:
                df_repeted.set_value(index, 'MF', 0)
        else:
            for i, val in enumerate(AUs):
                if i != au_index:
                    df_repeted.set_value(index, val, 0)
                else:
                    df_repeted.set_value(index, val, 1)
                
            au_index += 1
    filename = 'prediction-3.0-after/VAE-prediction-sample-'+str(numsample)
    predict_result_3(VAE, 
                     (df_repeted['subject'].values, df_repeted['task'].values, df_repeted['0'].values, df_repeted['path'].values, df_repeted[get_au_list()].values, df_repeted['MF'].values, len(dataParams.testSubjects_BP), len(dataParams.allTasks_BP)), 
                     filename)

#VAE.load_weights('models/VAE_after-RAUFMv-01-epoch-50.h5')