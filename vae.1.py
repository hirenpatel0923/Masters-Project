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
from data_preprocessing.data_loader import *
from data_preprocessing.data_generator_rgb import *

from encoder_network import EncoderNetwork
from decoder_network import DecoderNetwork
from discrimintor_network_1 import DiscriminatorNetwork

s = 23
seed(s)

epochs = 50
version = 'v-01-epoch-'+str(epochs)



#gan params
networkParams = NetworkParams()

#getting Data
dataParams = DataParams()
dataLoader = DataLoader()

print('Getting images and AU_OCC information.......')
Images, AU_OCC_array, MF_array = dataLoader.load_data(dataParams.trainSubjects_BP, dataParams.allTasks_BP)

train_images, test_images, train_au_occ, test_au_occ, train_mf_array, test_mf_array = train_test_split(Images, AU_OCC_array, MF_array, test_size=0.3)

#displaying ground truths....
plot_true_results((test_images, test_au_occ, test_mf_array), batch_size=networkParams.batch_size)
print('grounf truths saved....')


train_dataGenerator =  DataGenerator(train_images, train_au_occ, train_mf_array)
val_dataGenerator =  DataGenerator(test_images, test_au_occ, test_mf_array)


#############################################################################################
#### VAE model and Discriminator #########################
#############################################################################################
# #randomNormal = keras.initializers.RandomNormal(mean=0, stddev=0.05, seed=s)

encoder_input = Input(shape=(networkParams.modifiedHeight, networkParams.modifiedWidth, networkParams.numChannels))
au_input = Input(shape=(12,))
fm_input = Input(shape=(1,))


encoderModel, flat_shape_before, flat_shape_after = EncoderNetwork().buildModel()
decoderModel = DecoderNetwork().buildModel(flat_shape_before, flat_shape_after)
discriminatorModel = DiscriminatorNetwork().buildModel()

vae_output = decoderModel([encoderModel(encoder_input), au_input, fm_input])

VAE = Model([encoder_input, au_input, fm_input], vae_output)

#print('############## VAE Model ################')
#print(VAE.summary())

# mean_squared_error =  K.mean(K.square(decoder - encoder_input))
# vae_loss = mean_squared_error

VAE.compile(optimizer='adam', loss='mse')

#print('VAE Training starts..........')


# filepath = 'VAE-'+version+'.h5'
# filepath = 'models/VAE_checkpoint'+version+'.h5'
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint, TerminateOnNaN()]


# vae_history = VAE.fit_generator(train_dataGenerator, 
#                   epochs=epochs,
#                   verbose=1,
#                   validation_data=val_dataGenerator,
#                   callbacks=callbacks_list
#                   )

# plot_history(vae_history)


# # custom training function for vae
# orderedEnqueuerTrain = OrderedEnqueuer(train_dataGenerator)
# orderedEnqueuerTrain.start(workers=1, max_queue_size=10)
# output_generator_train = orderedEnqueuerTrain.get()


# orderedEnqueuerVal = OrderedEnqueuer(train_dataGenerator)
# orderedEnqueuerVal.start(workers=1, max_queue_size=10)
# output_generator_val = orderedEnqueuerVal.get()


# print('VAE Training starts...........')

# VAE_train_losses = {"VAE":[]}
# VAE_val_losses = {"VAE":[]}

# VAE_train_acc = {"VAE":[]}
# VAE_val_acc = {"VAE":[]}
# #losses['VAE'] = vae_history.history

# vae_train_loss = [0]
# vae_val_loss = [0]

# steps_per_epoch_train = np.floor(len(train_images)/networkParams.batch_size)
# steps_per_epoch_val = np.floor(len(test_images)/networkParams.batch_size)

# for epoch in range(1, epochs+1):

#     steps_done = 0

#     while steps_done < steps_per_epoch_train:
#         print('epoch: '+str(epoch)+'/'+str(epochs)+' - batch: '+str(steps_done)+'/'+str(steps_per_epoch_train),'---------'+'VAE_loss: '+str(vae_train_loss)+', VAEGAN_loss: '+str(vae_val_loss), end="\r")
        
#         generator_opt_train = next(output_generator_train)
#         generator_opt_val = next(output_generator_val)

#         if len(generator_opt_train) == 2:
#             combined_data, images = generator_opt_train

#             vae_train_loss = VAE.train_on_batch(combined_data, images)

#             VAE_train_losses["VAE"].append(vae_train_loss)

#             #VAE_train_acc['VAE'].append(vae_train_loss[1])


#             if steps_done % (steps_per_epoch_train//steps_per_epoch_val) == 0:
#                 combined_data, images = generator_opt_val

#                 vae_val_loss = VAE.test_on_batch(combined_data, images)

#                 VAE_val_losses["VAE"].append(vae_val_loss)

#                 #VAE_val_acc['VAE'].append(vae_val_loss[1])


#             steps_done += 1

#     if epochs % 1 == 0:
#         filename = 'VAE_'+version
#         plot_loss(VAE_train_losses, filename, 'train_loss')
#         plot_loss(VAE_val_losses, filename, 'val_loss')
#         #plot_loss(VAE_train_acc, filename, 'VAE_train_acc')
#         #plot_loss(VAE_val_acc, filename, 'VAE_val_acc')
        


#     # Update the plots
#     if epoch == 1 or epoch % 5 == 0:
#         filename = 'VAE_'+version + '--' + str(epoch)
#         plot_results(VAE, (test_images, test_au_occ, test_mf_array), batch_size=networkParams.batch_size, filename=filename+'_train')
#         VAE.save_weights('models/VAE-'+version+'.h5')
#         encoderModel.save_weights('models/Encoder_before-'+version+'.h5')
#         decoderModel.save_weights('models/Decoder_before-'+version+'.h5')
        



# print('Encoder and Decoder model saved...')
# encoderModel.save_weights('models/Encoder-VAE-before'+version+'.h5')
# decoderModel.save_weights('models/Decoder-VAE-before'+version+'.h5')

# VAE.load_weights(filepath)

#filepath = 'VAE-'+version+'.h5'
#VAE.save_weights(filepath)

######

filepath = 'models/VAE-'+version+'.h5'
VAE.load_weights(filepath)

discriminatorModel.trainable = False

vaegan_output = discriminatorModel(VAE.output)
print(vaegan_output)
VAEGAN = Model([encoder_input, au_input, fm_input], vaegan_output)

adam = Adam(lr=0.00001)

VAEGAN.compile(loss=['binary_crossentropy', 'categorical_crossentropy', 'binary_crossentropy'],
               optimizer=adam,
               metrics=['accuracy'])

print(VAEGAN.summary())
####################################################################################
########## custom training for VAEGAN ##############################################
####################################################################################

orderedEnqueuerTrain = OrderedEnqueuer(train_dataGenerator)
orderedEnqueuerTrain.start(workers=1, max_queue_size=10)
output_generator_train = orderedEnqueuerTrain.get()


orderedEnqueuerVal = OrderedEnqueuer(train_dataGenerator)
orderedEnqueuerVal.start(workers=1, max_queue_size=10)
output_generator_val = orderedEnqueuerVal.get()


print('VAEGAN Training starts...........')

train_losses = {"Discriminator":[], "VAEGAN":[]}
val_losses = {"Discriminator":[], "VAEGAN":[]}

train_acc = {"VAEGAN":[]}
val_acc = {"VAEGAN":[]}
#losses['VAE'] = vae_history.history

d_loss = [0]
vaegan_loss = [0]

steps_per_epoch_train = np.floor(len(train_images)/networkParams.batch_size)
steps_per_epoch_val = np.floor(len(test_images)/networkParams.batch_size)

for epoch in range(1, epochs+1):

    steps_done = 0

    while steps_done < steps_per_epoch_train:
        print('epoch: '+str(epoch)+'/'+str(epochs)+' - batch: '+str(steps_done)+'/'+str(steps_per_epoch_train),'---------'+'Discriminator_loss: '+str(d_loss[0])+', VAEGAN_loss: '+str(vaegan_loss[0]), end="\r")
        
        generator_opt_train = next(output_generator_train)
        generator_opt_val = next(output_generator_val)

        if len(generator_opt_train) == 2:
            combined_data, images = generator_opt_train

            vae_generated_images = VAE.predict(combined_data)

            X = np.concatenate((images, vae_generated_images))

            y_realfake = np.zeros(2*networkParams.batch_size)
            y_realfake[:networkParams.batch_size] = 1

            y_au = np.concatenate((combined_data[1], combined_data[1]))
            y_mf = np.concatenate((combined_data[2], combined_data[2]))

            # y = np.concatenate((y_realfake, y_au), axis=1)
            # y = np.concatenate((y, y_mf), axis=1)

            discriminatorModel.trainable = True
            d_loss = discriminatorModel.train_on_batch(X, [y_realfake, y_au, y_mf])

            ##### VAEGAN training generator
            discriminatorModel.trainable = False
            y1 = np.ones(networkParams.batch_size)

            vaegan_loss = VAEGAN.train_on_batch(combined_data, [y1, combined_data[1], combined_data[-1]])

            train_losses["Discriminator"].append(d_loss)
            train_losses['VAEGAN'].append(vaegan_loss)

            train_acc['VAEGAN'].append(vaegan_loss[1])


            if steps_done % (steps_per_epoch_train//steps_per_epoch_val) == 0:
                combined_data_val, images_val = generator_opt_val

                vae_generated_images = VAE.predict(combined_data_val)

                X = np.concatenate((images_val, vae_generated_images))

                y_realfake = np.zeros(2*networkParams.batch_size)
                y_realfake[:networkParams.batch_size] = 1

                y_au = np.concatenate((combined_data[1], combined_data[1]))
                y_mf = np.concatenate((combined_data[2], combined_data[2]))

                # y = np.concatenate((y_realfake, y_au), axis=1)
                # y = np.concatenate((y, y_mf), axis=1)

                discriminatorModel.trainable = True
                d_loss = discriminatorModel.test_on_batch(X, [y_realfake, y_au, y_mf])

                ##### VAEGAN training generator
                discriminatorModel.trainable = False
                y1 = np.ones(networkParams.batch_size)
                
                vaegan_loss = VAEGAN.test_on_batch(combined_data_val, [y1, combined_data[1], combined_data[-1]])

                val_losses["Discriminator"].append(d_loss)
                val_losses['VAEGAN'].append(vaegan_loss)

                val_acc['VAEGAN'].append(vaegan_loss[1])


            steps_done += 1

    if epochs % 1 == 0:
        filename = 'VAEGAN-D-RAUFM' + version
        plot_loss(train_losses, filename, 'train_loss')
        plot_loss(val_losses, filename, 'val_loss')
        plot_loss(train_acc, filename, 'train_acc')
        plot_loss(train_acc, filename, 'val_acc')

    # Update the plots
    if epoch == 1 or epoch % 5 == 0:
        #plot_generated()
        filename = 'VAE_'+version + '--' + str(epoch)
        plot_results(VAE, (test_images, test_au_occ, test_mf_array), batch_size=networkParams.batch_size, filename='VAE_after-RAUFM'+version)
        discriminatorModel.save_weights('models/Discriminator-RAUFM'+version+'.h5')
        VAEGAN.save_weights('models/VAEGAN-RAUFM'+version+'.h5')
        VAE.save_weights('models/VAE_after-RAUFM'+version+'.h5')
        encoderModel.save_weights('models/Encoder_after-RAUFM'+version+'.h5')
        decoderModel.save_weights('models/Decoder_after-RAUFM'+version+'.h5')

print('Training completed....')