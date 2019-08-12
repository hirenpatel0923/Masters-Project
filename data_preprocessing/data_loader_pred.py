import os
import csv
import pandas as pd
import numpy as np
import cv2

from data_preprocessing.data_params import *
from network_params import *

class DataLoader:
    def __init__(self):
        self.dataParams = DataParams()
        self.networkParams = NetworkParams()
        self.AU_OCC_files = []
        self.AU_INT_files = []
        self.combined_df = None
        self._badfiles()

    def _getImageFiles(self, Subjects, Tasks):

        first_array = True

        for subject in Subjects:
            
            for task in Tasks:

                base_path = os.path.join(BASE_DIR, subject)
                base_path = os.path.join(base_path, task)
                #AU_INT_path = os.path.join(BASE_GROUND_PATH, AU_INT)
                AU_OCC_path = os.path.join(BASE_GROUND_PATH, AU_OCC)

                #getting AU_OCC dataframe
                AU_OCC_filename = subject + '_' + task + '.csv'

                AU_OCC_df = self._get_AU_OCC_dataframe(AU_OCC_path, AU_OCC_filename, base_path, subject, task)

                #adding subject and task column to dataframe
                AU_OCC_df['subject'] = subject
                AU_OCC_df['task'] = task

                # Adding M/F to dataframe
                if 'F' in subject:
                    AU_OCC_df['MF'] = 1
                if 'M' in subject:
                    AU_OCC_df['MF'] = 0 
                
                # #getting AU_INT dataframe
                # AU_INT_df = None
                # first = True

                # for AU in self.dataParams.allIntAUs_BP:

                #     AU_INT_filename = subject + '_' + task + '_' + AU + '.csv'
                #     AU_INT_filepath = os.path.join(AU_INT_path, AU)

                #     temp_df = self._get_AU_Int_dataframe(AU_INT_filepath, AU_INT_filename, base_path, subject, task, AU)

                #     if first:
                #         AU_INT_df = temp_df.copy()
                #         first = False

                #     else:
                #         AU_INT_df = pd.merge(AU_INT_df, temp_df, on='0')


                # #merging both AU_OCC and AU_INT dataframes
                # combined_df = pd.merge(AU_OCC_df, AU_INT_df, on='0')

                if first_array:
                    self.combined_df = AU_OCC_df.copy()
                    first_array = False
                                    
                else:
                    self.combined_df = pd.concat([self.combined_df, AU_OCC_df])

        return self.combined_df

    def _get_AU_OCC_dataframe(self, AU_OCC_path, AU_OCC_filename, base_path, subject, task):

        csv_df = pd.read_csv(os.path.join(AU_OCC_path, AU_OCC_filename), dtype={'0': str})
        temp_au_lst = [str(int(au.replace('AU', ''))) for au in self.dataParams.allOccAUs_BP]
        drop_col_lst = []

        for col in csv_df.columns:
            if col not in temp_au_lst and col != '0':
                drop_col_lst.append(col)

        csv_df = csv_df.drop(drop_col_lst, axis=1)

        padding_num = self.get_padding_number(base_path)

        #check bad files
        
        csv_df['0'] = csv_df['0'].apply(lambda x: x.zfill(padding_num))
        mask = csv_df['0'].apply(lambda x: subject + '/' + task + '/' + x not in self.dataParams.lost_files)
        csv_df = csv_df[mask]

        csv_df['path'] = csv_df['0'].apply(lambda x: os.path.join(base_path, (x + IMAGE_TYPE)))
        
        #print(subject + '/' + task + ' shape : ' + str(csv_df.shape))
        return csv_df

    def _get_AU_Int_dataframe(self, AU_INT_path, AU_INT_filename, base_path, subject, task, AU):
        csv_df = pd.read_csv(os.path.join(AU_INT_path, AU_INT_filename), names=['0', AU])

        #check bad files
        mask = csv_df['0'].apply(lambda x: subject + '/' + task + '/' + str(x) not in self.dataParams.lost_files)
        csv_df = csv_df[mask]

        csv_df['0'] = csv_df['0'].apply(lambda x: os.path.join(base_path, str(x) + IMAGE_TYPE))
        #print(subject + '/' + task + '/' + AU + ' shape : ' + str(csv_df.shape))
        return csv_df

    def _badfiles(self):
        with open(BAD_FILES_PATH, 'r') as f:
            self.dataParams.lost_files = f.readlines()
            
    def get_padding_number(self, path):
        file_counts = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
        if file_counts > 999:
            return 4
        elif file_counts > 99:
            return 3
        elif file_counts > 9:
            return 2

    def subject_task_mapping(self, Subjects, Tasks):
        # subjects
        subject_to_int = {}
        int_to_subject = {}

        for i, subject in enumerate(Subjects):
            subject_to_int[subject] = i
            int_to_subject[i] = subject

        #tasks
        task_to_int = {}
        int_to_task = {}

        for i, task in enumerate(Tasks):
            task_to_int[task] = i
            int_to_task[i] = task

        return subject_to_int, int_to_subject, task_to_int, int_to_task

    # def get_images_from_list(self, array):
    #     images_array_list = []
    #     file_not_found_index = []
    #     for i, a in enumerate(array):
    #         img = cv2.imread(a)
    #         if os.path.isfile(a):
    #             print(a)
    #             img = cv2.resize(img, dsize=self.networkParams.dim)
    #             images_array_list.append(img)
    #         else:
    #             print('file not found : ',a)
    #             file_not_found_index.append(i)

    #     return np.array(images_array_list), file_not_found_index

    def load_data(self, Subjects, Tasks):
        df = self._getImageFiles(Subjects, Tasks)

        #print(df['0'].values)
        # Images, file_not_found_list = self.get_images_from_list(df['0'].values)
        # print(Images.shape)
    
        #getting AU_OCC
        temp_au_lst = [str(int(au.replace('AU', ''))) for au in self.dataParams.allOccAUs_BP]
        occ_col_lst = []

        for col in df.columns:
            if col in temp_au_lst and col != '0':
                occ_col_lst.append(col)

        AU_OCC_df = df[occ_col_lst]


        # #getting AU_INT
        # int_col_lst = []
        # for col in df.columns:
        #     if col in self.dataParams.allIntAUs_BP and col != '0':
        #         int_col_lst.append(col)

        # AU_INT_df = df[int_col_lst]

        # # Images_df.to_csv(filename + '_images.csv')
        # # AU_OCC_df.to_csv(filename + '_AU_OCC.csv')
        # # AU_INT_df.to_csv(filename + '_AU_INT.csv')

        # for tasks and subjects 
        #subject_to_int, int_to_subject, task_to_int, int_to_task = self.subject_task_mapping(self.dataParams.allSubjects_BP, self.dataParams.allTasks_BP) 

        #df['subject'] = df['subject'].apply(lambda x: subject_to_int[x])
        #df['task'] = df['task'].apply(lambda x: task_to_int[x])


        return df
        # return df['subject'].values, df['task'].values, df['0'].values, AU_OCC_df.values, df['MF'].values #, AU_INT_df.values


        #return df['0'].values, AU_OCC_df.values, df['MF'].values