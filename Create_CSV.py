from datetime import datetime
import pandas as pd
import nibabel.loadsave as nib
from numpy import floor, ceil
from random import sample
from tqdm import tqdm

from Excel_Processing import ProcessSpreadsheets

DIRECTORY           = "/home/ubuntu/proj/ImageCHD_dataset/"
DATASET_INFO_PATH   = DIRECTORY + "imageCHD_dataset_info.xlsx"
SCAN_INFO_PATH      = DIRECTORY + "imagechd_dataset_image_info.xlsx"
TRAIN_PORTION       = 0.7
EVAL_PORTION        = 0.2
TEST_PORTION        = 0.1

def get_counts(data: pd.DataFrame, names: list) -> dict[str, dict[str, int]]:
    chd_counts = [int(len(data[data[chd] == 1])) for chd in names]
    chds = sorted(zip(names, chd_counts), key = lambda x: (x[1], x[0]))
    print(chds)

    chd, count = chds[0]
    train_count = int(floor(count * TRAIN_PORTION))
    eval_count = int(ceil(count * EVAL_PORTION))
    test_count = int(ceil(count * TEST_PORTION))

    if (train_count + eval_count + test_count) > count:
        train_count -= 1

    print('chd: ' + chd + ', train: ' + str(train_count) \
           + ', eval: ' + str(eval_count) + ', test: ' + str(test_count) \
            + ', should be sum: ' + str(count))

    data.drop(data.loc[data[chd] == 1].index, inplace = True)
    names.remove(chd)

    result = {
        chd: {
            'train': train_count,
            'eval': eval_count,
            'test': test_count
        }
    }

    if len(names) != 0:
        result.update(get_counts(data, names))
    return result

def main():
    dataset_info = ProcessSpreadsheets(DIRECTORY + 'imageCHD_dataset_info.xlsx',
                                       DIRECTORY + 'imagechd_dataset_image_info.xlsx')
    mask = [(x.date() - y.date()).days > 10 * 365 \
            for x, y in zip(dataset_info['AcquisitionDate'], dataset_info['PatientBirthDate'])] 
    dataset_info.drop(dataset_info.loc[mask].index, inplace = True)
    print(dataset_info)
    chd_names = ["ASD", "VSD", "AVSD", "ToF", "DORV", "CA", "PA", "DSVC", "PDA"]

    chds = get_counts(dataset_info, chd_names)
    print(chds)
    
    dataset_info = ProcessSpreadsheets(DIRECTORY + 'imageCHD_dataset_info.xlsx',
                                       DIRECTORY + 'imagechd_dataset_image_info.xlsx')
    dataset_info.drop(dataset_info.loc[mask].index, inplace = True)
    chd_names = ["ASD", "VSD", "AVSD", "ToF", "DORV", "CA", "PA", "DSVC", "PDA"]

    for chd_split in chds.keys():
        mask = dataset_info[chd_split] == 1
        chd_info = dataset_info[mask]
        print(chd_info)
        dataset_info.drop(dataset_info.loc[mask].index, inplace = True)

if __name__ == '__main__':
    main()
# axial_count = [nib.load("C:\\Users\\leotu\\Downloads\\ImageCHD_dataset\\ImageCHD_dataset\\ct_" \
#                          + str(x) + "_image.nii.gz") \
#                .header['dim'][3] for x in dataset_info['index'].sort_values()]
# dataset_info['Axial_count'] = axial_count
# train = list()
# evaluation = list()
# test = list()

# for index, row in tqdm(dataset_info.iterrows()):
#     temp = row.copy(deep = True)
#     ax_c = temp['Axial_count']
#     temp = temp.rename({'Axial_count': 'Adjacency_count'})

#     temp_sagittal = list()
#     temp_coronal = list()
#     temp_axial = list()

#     for i in range(0, 512):
#         temp['Type'] = 'S'
#         temp['Indice'] = i
#         temp_sagittal.append(temp.copy(deep = True))
    
#     for i in range(0, 512):
#         temp['Type'] = 'C'
#         temp['Indice'] = i
#         temp_coronal.append(temp.copy(deep = True))
    
#     for i in range(0, ax_c):
#         temp['Type'] = 'A'
#         temp['Indice'] = i
#         temp['Adjacency_count'] = 512
#         temp_axial.append(temp.copy(deep = True))
    
#     new_train = sample(temp_sagittal, k = round(512*0.7))
#     tmp = list()

#     good = False
#     for sag in temp_sagittal:
#         good = True
#         for tr in new_train:
#             if sag.equals(tr):
#                 good = False
#                 break
#         if good:
#             tmp.append(sag)
    
#     temp_sagittal = tmp.copy()
#     for tr in new_train:
#         train.append(tr)

#     new_train = sample(temp_coronal, k = round(512*0.7))
#     tmp = list()

#     good = False
#     for cor in temp_coronal:
#         good = True
#         for tr in new_train:
#             if cor.equals(tr):
#                 good = False
#                 break
#         if good:
#             tmp.append(cor)
    
#     temp_coronal = tmp.copy()
#     for tr in new_train:
#         train.append(tr)

#     new_train = sample(temp_axial, k = round(ax_c*0.7))
#     tmp = list()

#     good = False
#     for ax in temp_axial:
#         good = True
#         for tr in new_train:
#             if ax.equals(tr):
#                 good = False
#                 break
#         if good:
#             tmp.append(ax)
    
#     temp_axial = tmp.copy()
#     for tr in new_train:
#         train.append(tr)

#     new_eval = sample(temp_sagittal, k = round(512*0.2))
#     tmp = list()

#     good = False
#     for sag in temp_sagittal:
#         good = True
#         for ev in new_eval:
#             if sag.equals(ev):
#                 good = False
#                 break
#         if good:
#             tmp.append(sag)
    
#     temp_sagittal = tmp.copy()
#     for ev in new_eval:
#         evaluation.append(ev)

#     new_eval = sample(temp_coronal, k = round(512*0.2))
#     tmp = list()

#     good = False
#     for cor in temp_coronal:
#         good = True
#         for ev in new_eval:
#             if cor.equals(ev):
#                 good = False
#                 break
#         if good:
#             tmp.append(cor)
    
#     temp_coronal = tmp.copy()
#     for ev in new_eval:
#         evaluation.append(ev)

#     new_eval = sample(temp_axial, k = round(ax_c*0.2))
#     tmp = list()

#     good = False
#     for ax in temp_axial:
#         good = True
#         for ev in new_eval:
#             if ax.equals(ev):
#                 good = False
#                 break
#         if good:
#             tmp.append(ax)
    
#     temp_axial = tmp.copy()
#     for ev in new_eval:
#         evaluation.append(ev)

#     for te in temp_sagittal:
#         test.append(te)
#     for te in temp_coronal:
#         test.append(te)
#     for te in temp_axial:
#         test.append(te)

# train_dataset = pd.DataFrame(train).reset_index().drop('level_0', axis = 1)
# eval_dataset = pd.DataFrame(evaluation).reset_index().drop('level_0', axis = 1)
# test_dataset = pd.DataFrame(test).reset_index().drop('level_0', axis = 1)

# print("Train: ", str(train_dataset.__len__()))
# print("Eval: ", str(eval_dataset.__len__()))
# print("Test: ", str(test_dataset.__len__()))

# try:
#     train_dataset.to_csv(path_or_buf = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\train_dataset_info.csv",
#                          index = False, mode = 'x')
#     print('Train dataset CSV saved.')
# except:
#     print('Train dataset CSV already present.')

# try:
#     eval_dataset.to_csv(path_or_buf = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\eval_dataset_info.csv",
#                         index = False, mode = 'x')
#     print('Eval dataset CSV saved.')
# except:
#     print('Eval dataset CSV already present.')

# try:
#     test_dataset.to_csv(path_or_buf = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\test_dataset_info.csv",
#                         index = False, mode = 'x')
#     print('Test dataset CSV saved.')
# except:
#     print('Test dataset CSV already present.')