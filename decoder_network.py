import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense, Concatenate, Conv2D, MaxPool2D, Flatten, Conv2DTranspose, Reshape, Dropout, BatchNormalization
from keras.utils.data_utils import OrderedEnqueuer
from keras.callbacks import TerminateOnNaN, ModelCheckpoint
from keras.optimizers import RMSprop, Adam
from keras.losses import mse, binary_crossentropy
from keras.layers import LeakyReLU

from network_params import *

networkParams = NetworkParams()

class DecoderNetwork:
    def __init__(self):
        pass

    def buildModel(self, flat_shape_before, flat_shape_after):
        decoder_input = Input(shape=(networkParams.z_dim,))
        au_input = Input(shape=(12,))
        fm_input = Input(shape=(1,))


        concatenate = Concatenate(axis=-1)([decoder_input, au_input, fm_input])

        decoder = Dense(256, activation='tanh')(concatenate)
        decoder = Dropout(0.3)(decoder)
        decoder = BatchNormalization()(decoder)

        decoder = Dense(512, activation='tanh')(concatenate)
        decoder = Dropout(0.3)(decoder)
        
        decoder = Dense(flat_shape_after[-1])(decoder)

        decoder = Reshape((flat_shape_before[1], flat_shape_before[2], flat_shape_before[3]))(decoder)

        decoder = Conv2DTranspose(256, (2,2), strides=(2,2), padding='SAME')(decoder)
        decoder = LeakyReLU(alpha=0.2)(decoder)
        decoder = BatchNormalization()(decoder)

        decoder = Conv2DTranspose(128, (2,2), strides=(2,2), padding='SAME')(decoder)
        decoder = LeakyReLU(alpha=0.2)(decoder)
        decoder = BatchNormalization()(decoder)

        decoder = Conv2DTranspose(64, (2,2), strides=(2,2), padding='SAME')(decoder)
        decoder = LeakyReLU(alpha=0.2)(decoder)
        decoder = BatchNormalization()(decoder)

        decoder = Conv2DTranspose(3, (2,2), strides=(2,2), padding='SAME')(decoder)


        DecoderModel = Model([decoder_input, au_input, fm_input], decoder)
        DecoderModel.compile(optimizer='adam',
                            loss='categorical_crossentropy')

        #print(DecoderModel.summary())
        
        return DecoderModel
