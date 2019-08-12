import numpy as np
from PIL import Image
import os
import cv2
import keras
from keras.utils import to_categorical
from network_params import *

class DataGenerator(keras.utils.Sequence):
    'Generates data for keras model'
    def __init__(self, Images_list, AU_OCC_list, MF_list):
        self.networkParams = NetworkParams()

        self.indexes = [i for i in range(len(Images_list))]
        self.Images_list = Images_list
        self.AU_OCC_list = AU_OCC_list
        self.MF_list = MF_list

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.networkParams.batch_size))

    def __getitem__(self, index):
        # generate indexex of batch
        indexes = self.indexes[index*self.networkParams.batch_size:(index+1)*self.networkParams.batch_size]

        #generate data
        Images, (AU_OCC, MF_label) = self.__data_generation(indexes)

        return [Images, AU_OCC, MF_label], Images

    def on_epoch_end(self):
        #self.indexes = np.arange(len(self.list_IDs))
        if self.networkParams.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        #Initialization
        Images = np.empty((self.networkParams.batch_size, *self.networkParams.dim, self.networkParams.numChannels))
        AU_OCC = np.empty((self.networkParams.batch_size, self.AU_OCC_list.shape[-1]), dtype=int)
        MF_list = np.empty((self.networkParams.batch_size), dtype=int)
        nan_file_lst = []

        for i, ID in enumerate(list_IDs_temp):
 
            if os.path.isfile(self.Images_list[ID]):
                img = cv2.imread(self.Images_list[ID])
                img = cv2.resize(img, dsize=self.networkParams.dim)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                #img = self.normalize_meanstd(img, axis=(0,1,2))
                img = ((img / 255.) * 2.) - 1.

                #nun check
                if np.isnan(img).any():
                    print('nun check and file found : ', self.Images_list[ID])
                    nan_file_lst.append(i)

                Images[i, ] = img

                # store classes
                AU_OCC[i] = self.AU_OCC_list[ID]
                MF_list[i] = self.MF_list[ID]

        Images = np.delete(Images, nan_file_lst, 0)
        AU_OCC = np.delete(AU_OCC, nan_file_lst, 0)
        MF_list = np.delete(MF_list, nan_file_lst, 0)

        return Images, (AU_OCC, MF_list)

    def normalize_meanstd(self, a, axis=None): 
        # axis param denotes axes along which mean & std reductions are to be performed
        mean = np.mean(a, axis=axis, keepdims=True)
        std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
        return (a - mean) / std