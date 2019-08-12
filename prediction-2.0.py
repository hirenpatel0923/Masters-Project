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

VAE.load_weights(filepath)

plot_results(VAE, (df['path'].values, df[get_au_list()].values, df['MF'].values), batch_size=networkParams.batch_size, filename=filename+'_2.0_prediction')

VAE.load_weights('models/VAE_after-RAUFMv-01-epoch-50.h5')

plot_results(VAE, (df['path'].values, df[get_au_list()].values, df['MF'].values), batch_size=networkParams.batch_size, filename=filename+'RAUFM_2.0_prediction')

# samples = 10
# numRandomAus = 5
# #getting predictions ground truths 
# for numSample in range(1, numRandomAus + 1):
#     for samp in range(samples):

#         temp_df = df.sample()
#         df = df.drop(temp_df.index)
#         mf_df = temp_df.copy()
#         au_df = temp_df.copy()

#         mf_df['MF'] = mf_df['MF'].apply(lambda x: 0 if x==1 else 1)
#         random_au = random.sample(AUs, numSample)
#         for au in random_au:
#             au_df[au] = au_df[au].apply(lambda x:0 if x==1 else 1)
#         temp_df = pd.concat((temp_df, mf_df, au_df))
#         #print(temp_df)

#         au_occurred = []
#         for au in AUs:
#             tp = mf_df[au].apply(lambda x: au_occurred.append(au) if x==1 else 0)


#         filename = 'prediction-2.0-after/VAE-prediction-sample-'+str(numSample)+'-'+str(samp)
#         predict_result_2(VAE, 
#                         (temp_df['subject'].values, temp_df['task'].values, temp_df['0'].values, temp_df['path'].values, temp_df[get_au_list()].values, temp_df['MF'].values, len(dataParams.testSubjects_BP), len(dataParams.allTasks_BP)), 
#                         (random_au, au_occurred),
#                         filename)

# temp_df = df.groupby(['subject','task']).apply(lambda x: x.sample(1)).reset_index(drop=True)
# new_df = temp_df.copy()
# new_df['MF'] = new_df['MF'].apply(lambda x: 0 if x==1 else 1)
# print(new_df.head(5))

# filename = 'prediction-2.0/VAE-prediction-'
# predict_results(VAE, 
#                 (temp_df['subject'].values, temp_df['task'].values, temp_df['0'].values, temp_df['path'].values, temp_df[get_au_list()].values, temp_df['MF'].values, len(dataParams.testSubjects_BP), len(dataParams.allTasks_BP)), 
#                 filename)
