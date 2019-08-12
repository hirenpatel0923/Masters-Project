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

class DiscriminatorNetwork:
    def __init__(self):
        pass

    def buildModel(self):
        discriminator_input = Input(shape=(networkParams.modifiedHeight, networkParams.modifiedWidth, networkParams.numChannels))

        discriminator = Conv2D(64, (4,4), strides=(1,1), padding='SAME')(discriminator_input)
        discriminator = LeakyReLU(alpha=0.02)(discriminator)
        discriminator = BatchNormalization()(discriminator)
        discriminator = MaxPool2D((2,2), padding='SAME')(discriminator)

        discriminator = Conv2D(128, (4,4), padding='SAME')(discriminator)
        discriminator = LeakyReLU(alpha=0.02)(discriminator)
        discriminator = BatchNormalization()(discriminator)
        discriminator = MaxPool2D((2,2), padding='SAME')(discriminator)

        discriminator = Conv2D(256, (4,4), strides=(1,1), padding='SAME')(discriminator)
        discriminator = LeakyReLU(alpha=0.02)(discriminator)
        discriminator = BatchNormalization()(discriminator)
        discriminator = MaxPool2D((2,2), padding='SAME')(discriminator)

        discriminator = Conv2D(512, (4,4), padding='SAME')(discriminator)
        discriminator = LeakyReLU(alpha=0.02)(discriminator)
        discriminator = BatchNormalization()(discriminator)
        discriminator = MaxPool2D((2,2), padding='SAME')(discriminator)

        flat_shape_before = K.int_shape(discriminator)

        discriminator = Flatten()(discriminator)
        flat_shape_after = K.int_shape(discriminator)

        discriminator = Dense(512, activation='tanh')(discriminator)
        discriminator = Dropout(0.3)(discriminator)
        discriminator = BatchNormalization()(discriminator)

        discriminator = Dense(256)(discriminator)
        discriminator = Dropout(0.3)(discriminator)

        discriminator_realfake = Dense(1, activation='sigmoid')(discriminator)

        discriminator_au = Dense(12, activation='softmax')(discriminator)

        discriminator_identity = Dense(1, activation='sigmoid')(discriminator)


        discriminatorModel = Model(discriminator_input, [discriminator_realfake, discriminator_au ,discriminator_identity])
        discriminatorModel.compile(optimizer='adam',
                                   loss=['binary_crossentropy', 'categorical_crossentropy', 'binary_crossentropy'])
        print(discriminatorModel.summary())
        
        return discriminatorModel
