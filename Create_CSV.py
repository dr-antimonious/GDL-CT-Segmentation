import pandas as pd
import nibabel as nib
from random import sample
from Excel_Processing import ProcessSpreadsheets

DATASET_INFO_PATH = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\imageCHD_dataset_info.xlsx"
SCAN_INFO_PATH = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\imagechd_dataset_image_info.xlsx"

dataset_info = ProcessSpreadsheets(DATASET_INFO_PATH, SCAN_INFO_PATH)

try:
    dataset_info.to_csv(path_or_buf = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\patient_info.csv",
                        index = False, mode = 'x')
    print('Patient CSV saved.')
except:
    print('Patient CSV already present. Moving on.')

dataset_info = dataset_info.drop(['ASD', 'VSD', 'AVSD', 'ToF', 'TGA', 'CA',
                                  'PA', 'PDA', 'COUNT', 'PatientSex',
                                  'PatientBirthDate', 'AcquisitionDate'],
                                  axis = 1)

axial_count = [nib.load("C:\\Users\\leotu\\Downloads\\ImageCHD_dataset\\ImageCHD_dataset\\ct_" \
                         + str(x) + "_image.nii.gz") \
               .header['dim'][3] for x in dataset_info['index'].sort_values()]
dataset_info['Axial_count'] = axial_count
train = list()
evaluation = list()
test = list()

for index, row in dataset_info.iterrows():
    temp = row.copy(deep = True)
    ax_c = temp['Axial_count']
    temp = temp.rename({'Axial_count': 'Adjacency_count'})

    temp_sagittal = list()
    temp_coronal = list()
    temp_axial = list()

    for i in range(0, 512):
        temp['Type'] = 'S'
        temp['Indice'] = i
        temp_sagittal.append(temp.copy(deep = True))
    
    for i in range(0, 512):
        temp['Type'] = 'C'
        temp['Indice'] = i
        temp_coronal.append(temp.copy(deep = True))
    
    for i in range(0, ax_c):
        temp['Type'] = 'A'
        temp['Indice'] = i
        temp['Adjacency_count'] = 512
        temp_axial.append(temp.copy(deep = True))
    
    new_train = sample(temp_sagittal, k = round(512*0.7))
    temp_sagittal = list(filter(lambda i: , temp_sagittal))
    train.append(new_train)

    new_train = sample(temp_coronal, k = round(512*0.7))
    temp_coronal = list(filter(lambda i: i not in new_train, temp_coronal))
    train.append(new_train)

    new_train = sample(temp_axial, k = round(ax_c*0.7))
    temp_axial = list(filter(lambda i: i not in new_train, temp_axial))
    train.append(new_train)

    new_eval = sample(temp_sagittal, k = round(512*0.2))
    temp_sagittal = list(filter(lambda i: i not in new_eval, temp_sagittal))
    evaluation.append(new_eval)

    new_eval = sample(temp_coronal, k = round(512*0.2))
    temp_coronal = list(filter(lambda i: i not in new_eval, temp_coronal))
    evaluation.append(new_eval)

    new_eval = sample(temp_axial, k = round(ax_c*0.2))
    temp_axial = list(filter(lambda i: i not in new_eval, temp_axial))
    evaluation.append(new_eval)

    test.append(temp_sagittal)
    test.append(temp_coronal)
    test.append(temp_axial)

train_dataset = pd.DataFrame(train).reset_index().drop('level_0', axis = 1)
eval_dataset = pd.DataFrame(evaluation).reset_index().drop('level_0', axis = 1)
test_dataset = pd.DataFrame(test).reset_index().drop('level_0', axis = 1)

try:
    train_dataset.to_csv(path_or_buf = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\train_dataset_info.csv",
                         index = False, mode = 'x')
    print('Train dataset CSV saved.')
except:
    print('Train dataset CSV already present.')

try:
    eval_dataset.to_csv(path_or_buf = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\eval_dataset_info.csv",
                        index = False, mode = 'x')
    print('Eval dataset CSV saved.')
except:
    print('Eval dataset CSV already present.')

try:
    test_dataset.to_csv(path_or_buf = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\test_dataset_info.csv",
                        index = False, mode = 'x')
    print('Test dataset CSV saved.')
except:
    print('Test dataset CSV already present.')