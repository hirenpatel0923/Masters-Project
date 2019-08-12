import numpy as np
import random
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from network_params import NetworkParams
from scipy.misc import toimage

networkParams = NetworkParams()

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('output/vae_loss.png')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('output/vae_acc.png')
    plt.close()
    #plt.show()


def plot_loss(losses, filename, name='loss'):
    """
    @losses.keys():
        0: loss
        1: accuracy
    """
    plt.figure(figsize=(10,8))

    for key in losses.keys():
        loss = [v for v in losses[key]]
        plt.plot(loss, label=key)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('output/'+str(filename)+'_'+name+'.png')
    plt.close()
    #plt.show()


def plot_results(models,
                 data,
                 batch_size=128,
                 filename="vae_predicted_results"):

    vae = models
    images, au_occ, fm_int = data

    n = 12
    sample_size = 25
    

    random_index = [random.randint(0, len(images)) for i in range(25)]

    combined_data = get_samples(random_index, images, au_occ, fm_int, batch_size)

    fig =plt.figure(figsize=(10, 10))

    counter = 0
    for i in range(len(random_index)):
        img = combined_data[0][counter].reshape(1,networkParams.modifiedHeight, networkParams.modifiedWidth, networkParams.numChannels)
        au = combined_data[1][counter].reshape(1,12)
        fm = combined_data[2][counter].reshape(1,1)
        
        x_decoded = vae.predict([img, au, fm])
        #print(x_decoded)
        reshaped_pred = x_decoded.reshape(networkParams.modifiedHeight, networkParams.modifiedWidth, networkParams.numChannels)
        pred = cv2.resize(reshaped_pred, dsize=(sample_size, sample_size))
        #pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        pred = (pred / 2.) + (1./2.)
        #pred = ((pred + 1.) / 2.) * 255.

        fig.add_subplot(5, 5, counter+1)
        plt.imshow(pred)
        counter += 1

    # plt.figure(figsize=(10, 10))
    start_range = sample_size // 2
    end_range = n * sample_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, sample_size)
    sample_range_x = np.round([i for i in range(n)], 1)
    sample_range_y = np.round([i for i in range(n)], 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    #plt.imshow(figure)
    plt.savefig('output/'+filename+'.png')
    plt.close()
    #plt.show()

def plot_true_results(data,
                 batch_size=128,
                 filename="ground_truths"):

    images, au_occ, fm_int = data

    n = 12
    sample_size = 25
    

    random_index = [random.randint(0, len(images)) for i in range(25)]

    combined_data = get_samples(random_index, images, au_occ, fm_int, batch_size)

    fig =plt.figure(figsize=(10, 10))

    counter = 0
    for i in range(len(random_index)):
        img = combined_data[0][counter].reshape(1,networkParams.modifiedHeight, networkParams.modifiedWidth, networkParams.numChannels)
        au = combined_data[1][counter].reshape(1,12)
        fm = combined_data[2][counter].reshape(1,1)
        img = img.reshape(networkParams.modifiedHeight, networkParams.modifiedWidth, networkParams.numChannels)
        #x_decoded = vae.predict([img, au, fm])
        #print(x_decoded)
        img = cv2.resize(img, dsize=(sample_size, sample_size))
        img = toimage(img)

        fig.add_subplot(5, 5, counter+1)
        plt.imshow(img)
        counter += 1

    # plt.figure(figsize=(10, 10))
    start_range = sample_size // 2
    end_range = n * sample_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, sample_size)
    sample_range_x = np.round([i for i in range(n)], 1)
    sample_range_y = np.round([i for i in range(n)], 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    #plt.imshow(figure)
    plt.savefig('output/'+filename+'.png')
    plt.close()
    #plt.show()


def get_samples(random_index, images, au_occ, fm_int, batch_size):
    Images = np.empty((batch_size, *networkParams.dim, networkParams.numChannels))
    AU_OCC = np.empty((batch_size, au_occ.shape[-1]), dtype=int)
    MF_list = np.empty((batch_size), dtype=int)

    for i, ID in enumerate(random_index):
        if os.path.isfile(images[ID]):
            #print(self.Images_list[ID])
            img = cv2.imread(images[ID])
            img = cv2.resize(img, dsize=networkParams.dim)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            #img = img.reshape(32, 32, 1)
            #img = self.normalize_meanstd(img, axis=(0,1,2))
            img = (img / 255.) * 2. - 1.
            #cv2.imshow('Img',img)
            #cv2.waitKey()
            Images[i, ] = img

            # store classes
            AU_OCC[i] = au_occ[ID]
            MF_list[i] = fm_int[ID]

    return [Images,AU_OCC,MF_list]


def plot_results_gray(models,
                 data,
                 batch_size=128,
                 filename="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    vae = models
    images, au_occ, fm_int = data

    n = 25
    sample_size = 32
    

    random_index = [random.randint(0, len(images)) for i in range(n)]

    combined_data = get_samples(random_index, images, au_occ, fm_int, batch_size)

    fig =plt.figure(figsize=(10, 10))

    counter = 0
    for i in range(len(random_index)):
        img = combined_data[0][counter].reshape(1,32,32,1)
        au = combined_data[1][counter].reshape(1,12)
        fm = combined_data[2][counter].reshape(1,1)
        
        x_decoded = vae.predict(img)
        #print(x_decoded.shape)
        reshaped_pred = x_decoded.reshape(32,32)
        pred = Image.fromarray(reshaped_pred)
        pred = np.asarray(pred)
        pred = cv2.resize(pred, dsize=(sample_size, sample_size))
       
    
        fig.add_subplot(5, 5, counter+1)
        plt.imshow(pred, interpolation='nearest', cmap='gray_r')

        counter += 1

    # plt.figure(figsize=(10, 10))
    start_range = sample_size // 2
    end_range = n * sample_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, sample_size)
    sample_range_x = np.round([i for i in range(n)], 1)
    sample_range_y = np.round([i for i in range(n)], 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.tight_layout()
    plt.close()
    #plt.imshow(figure)
    #plt.savefig('output/'+filename+'.png')
    #plt.show()


def predict_results(model, data, filename=''):
    subjects_list, tasks_list, frame_list, images_list, au_list, mf_list, subject_count, task_count = data
    
    random_index = [i for i in range(len(images_list))]
    combined_data = get_samples(random_index, images_list, au_list, mf_list, len(random_index))

    pred = model.predict_on_batch(combined_data)
    
    f, axes = plt.subplots(task_count,subject_count, figsize=(subject_count, task_count))

    count = 0

    for i in range(subject_count):
        for j in range(task_count):
            subject = str(subjects_list[count])
            task = str(tasks_list[count])
            frame = str(frame_list[count])
            label = subject + '-' + task

            img = pred[count]
            img = img.reshape(networkParams.modifiedHeight, networkParams.modifiedWidth, networkParams.numChannels)

            img = (img / 2.) + (1./2.)

            axes[j][i].imshow(img, interpolation='nearest')
            axes[j][i].set_xticks([])
            axes[j][i].set_yticks([])

            #axes[j][i].set_title(label, fontsize=12)
            axes[j][i].set_xlabel(label)
            axes[j][i].set_ylabel(frame)
            
            
            #labels = [str(subject), '-', str(task)]
            #axes[j][i].set_xticks()
            #axes[j][i].set_yticklabels(r"task")

            count += 1

    plt.tight_layout()
    plt.savefig(filename)
    #plt.show()
    plt.close()



def predict_result_2(model, data, AUs, filename = ''):
    model = model

    random_au, au_occurred = AUs

    subjects_list, tasks_list, frame_list, images_list, au_list, mf_list, subject_count, task_count = data
    
    random_index = [i for i in range(len(images_list))]
    combined_data = get_samples(random_index, images_list, au_list, mf_list, len(random_index))

    pred = model.predict_on_batch(combined_data)
    
    f, axes = plt.subplots(1, 4, figsize=(10, 1))


    original_img = cv2.imread(images_list[0])
    original_img = cv2.resize(original_img, dsize=networkParams.dim)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = original_img / 255.
    
    axes[0].imshow(original_img, interpolation='nearest')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    title = str(subjects_list[0]) + ' - ' + str(tasks_list[0]) + ' - ' + str(frame_list[0]) + ' => '
    for au in au_occurred:
        title += 'AU' + au + ' '

    f.suptitle(title)

    count = 0
    #for i in range(1):
    for j in range(1, 4):

        img = pred[count]
        img = img.reshape(networkParams.modifiedHeight, networkParams.modifiedWidth, networkParams.numChannels)

        img = (img / 2.) + (1./2.)

        axes[j].imshow(img, interpolation='nearest')
        axes[j].set_xticks([])
        axes[j].set_yticks([])

        if j == 1:
            axes[j].set_xlabel('Generated')
        elif j == 2:
            axes[j].set_xlabel('MF')
        elif j == 3:
            rand_au = ''
            for au in random_au:
                rand_au += 'AU' + au + ' '
            axes[j].set_xlabel(rand_au)
        else:
            pass
        
        
        count += 1

    plt.tight_layout()
    plt.savefig(filename)
    #plt.show()
    plt.close()


def predict_result_3(model, data, filename = ''):
    AUs = '1,2,4,6,7,10,12,14,15,17,23,24'.split(',')
    
    model = model  

    subjects_list, tasks_list, frame_list, images_list, au_list, mf_list, subject_count, task_count = data
    
    random_index = [i for i in range(len(images_list))]
    combined_data = get_samples(random_index, images_list, au_list, mf_list, len(random_index))

    pred = model.predict_on_batch(combined_data)
    
    f, axes = plt.subplots(1, 15, figsize=(14, 1))


    original_img = cv2.imread(images_list[0])
    original_img = cv2.resize(original_img, dsize=networkParams.dim)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = original_img / 255.
    
    axes[0].imshow(original_img, interpolation='nearest')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    title = str(subjects_list[0]) + ' - ' + str(tasks_list[0]) + ' - ' + str(frame_list[0]) + ' => '
    
    f.suptitle(title)

    count = 0
    au_count = 0
    #for i in range(1):
    for j in range(1, 15):

        img = pred[count]
        img = img.reshape(networkParams.modifiedHeight, networkParams.modifiedWidth, networkParams.numChannels)

        img = (img / 2.) + (1./2.)

        axes[j].imshow(img, interpolation='nearest')
        axes[j].set_xticks([])
        axes[j].set_yticks([])

        if j == 1:
            axes[j].set_xlabel('Generated')
        elif j == 2:
            axes[j].set_xlabel('MF') 
        else:
            rand_au = 'AU' + AUs[au_count]
            axes[j].set_xlabel(rand_au)
            au_count += 1
        
        
        count += 1

    plt.tight_layout()
    plt.savefig(filename)
    #plt.show()
    plt.close()

