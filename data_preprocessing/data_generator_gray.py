import numpy as np
from PIL import Image
import os
import cv2
import keras
import scipy
from keras.utils import to_categorical

class DataGenerator(keras.utils.Sequence):
    'Generates data for keras model'
    def __init__(self, Images_list, AU_OCC_list, MF_list, dim, num_channel, batch_size = 32, shuffle=True):
        self.indexes = [i for i in range(len(Images_list))]
        self.dim = dim
        self.num_channel = num_channel
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.Images_list = Images_list
        self.AU_OCC_list = AU_OCC_list
        self.MF_list = MF_list

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        # generate indexex of batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        #generate data
        Images, (AU_OCC, MF_label) = self.__data_generation(indexes)
        #Images, [Subjects, Tasks, AU_OCC, AU_INT] = self.__data_generation(indexes)

        return Images, Images # Images, Images # [Images, AU_OCC, MF_label], Images #[Subjects, Tasks, AU_OCC, AU_INT]

    # def on_epoch_end(self):
    #     self.indexes = np.arange(len(self.list_IDs))
    #     if self.shuffle:
    #         np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        #Initialization
        #Images = []#np.empty((self.batch_size), dtype=str)
        Images = np.empty((self.batch_size, *self.dim, self.num_channel))
        AU_OCC = np.empty((self.batch_size, self.AU_OCC_list.shape[-1]), dtype=int)
        MF_list = np.empty((self.batch_size), dtype=int)
        #Image_name_list = np.empty((self.batch_size))
        nan_file_lst = []
        

        for i, ID in enumerate(list_IDs_temp):

            #if os.path.isfile(self.Images_list[ID]):
            # store samples
            # Images[i, ] = np.array(Image.open(self.Images_list[ID]))
            
            if os.path.isfile(self.Images_list[ID]):
                #print(self.Images_list[ID])
                img = cv2.imread(self.Images_list[ID], cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, dsize=self.dim)
                img = img.reshape(self.dim[0], self.dim[1], self.num_channel)
                #img = self.normalize_meanstd(img, axis=(0,1,2))
                img = ((img - 255.) * 2.) - 1.
                #img = img / 255.

                #nun check
                if np.isnan(img).any():
                    print('nan check : ',self.Images_list[ID])
                    nan_file_lst.append(i)

                Images[i, ] = img

                # store classes
                AU_OCC[i] = self.AU_OCC_list[ID]
                MF_list[i] = self.MF_list[ID]
                #Image_name_list[i] = self.Images_list[ID]

        Images = np.delete(Images, nan_file_lst, 0)
        AU_OCC = np.delete(AU_OCC, nan_file_lst, 0)
        MF_list = np.delete(MF_list, nan_file_lst, 0)
        #Image_name_list = np.delete(Image_name_list, nan_file_lst, 0)

        return Images, (AU_OCC, MF_list) #[to_categorical(Subjects, num_classes=self.subject_classes), to_categorical(Tasks, num_classes=self.task_classes), AU_OCC, AU_INT]

    def normalize_meanstd(self, a, axis=None): 
        # axis param denotes axes along which mean & std reductions are to be performed
        #a *= (1./255)
        mean = np.mean(a, axis=axis, keepdims=True)
        std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
        return (a - mean) / std