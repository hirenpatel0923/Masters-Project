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

class EncoderNetwork:
    def __init__(self):
        pass

    def buildModel(self):
        encoder_input = Input(shape=(networkParams.modifiedHeight, networkParams.modifiedWidth, networkParams.numChannels))

        encoder = Conv2D(64, (4,4), strides=(1,1), padding='SAME')(encoder_input)
        encoder = LeakyReLU(alpha=0.2)(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = MaxPool2D((2,2), padding='SAME')(encoder)

        encoder = Conv2D(128, (4,4), padding='SAME')(encoder)
        encoder = LeakyReLU(alpha=0.2)(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = MaxPool2D((2,2), padding='SAME')(encoder)

        encoder = Conv2D(256, (4,4), strides=(1,1), padding='SAME')(encoder)
        encoder = LeakyReLU(alpha=0.2)(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = MaxPool2D((2,2), padding='SAME')(encoder)

        encoder = Conv2D(512, (4,4), padding='SAME')(encoder)
        encoder = LeakyReLU(alpha=0.2)(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = MaxPool2D((2,2), padding='SAME')(encoder)

        flat_shape_before = K.int_shape(encoder)

        encoder = Flatten()(encoder)
        flat_shape_after = K.int_shape(encoder)

        encoder = Dense(512, activation='tanh')(encoder)
        encoder = Dropout(0.3)(encoder)


        encoder = Dense(256, activation='tanh')(encoder)
        encoder = Dropout(0.3)(encoder)
        encoder = BatchNormalization()(encoder)

        z = Dense(networkParams.z_dim)(encoder)

        EncoderModel = Model(encoder_input, z)
        EncoderModel.compile(optimizer='adam',
                            loss='categorical_crossentropy')
        #print(EncoderModel.summary())
        
        return EncoderModel, flat_shape_before, flat_shape_after
