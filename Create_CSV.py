import pandas as pd
import nibabel.loadsave as nib
from random import sample
from tqdm import tqdm

DIRECTORY = "/home/ubuntu/proj/ImageCHD_dataset/"
DATASET_INFO_PATH = DIRECTORY + "imageCHD_dataset_info.xlsx"
SCAN_INFO_PATH = DIRECTORY + "imagechd_dataset_image_info.xlsx"
DATASET_INFO = pd.read_csv(filepath_or_buffer = DIRECTORY + "patient_info.csv")

CHDS = ['ASD', 'VSD', 'AVSD', 'ToF', 'TGA', 'CA', 'PA', 'PDA']
CHD_COUNTS = [DATASET_INFO[chd].value_counts()[1.0] for chd in CHDS]
print(CHD_COUNTS)

print(sorted(zip(CHDS, CHD_COUNTS), key = lambda x: x[1]))

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