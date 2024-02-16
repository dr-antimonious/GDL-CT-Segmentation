import os
import pandas as pd
import numpy as np
import nibabel as nib

DATASET_PATH = "D:\\ImageCHD_dataset\\ImageCHD_dataset\\"
FILES = os.listdir(DATASET_PATH)
FILES.remove('imageCHD_dataset_info.xlsx')
FILES.sort()

SHEET = pd.read_excel(DATASET_PATH + "imageCHD_dataset_info.xlsx", dtype = pd.UInt16Dtype) \
          .drop(['Unnamed: 17', 'ONLY FIRST 8', 'FIRST 8 + MORE', 'NORMAL', 'IGNORED'], axis = 1) \
          .drop(110, axis = 0)

n_cols = np.array([], dtype = np.uint16)
for f in FILES:
    if f.split('_')[2].split('.')[0] == 'image':
        n_cols = np.append(n_cols, nib.load(DATASET_PATH + f).header['dim'][3])

metadata = pd.concat([SHEET, pd.DataFrame({'n_cols': n_cols})], axis = 1) \
             .drop(SHEET[(SHEET['DORV'] == 1) |
                         (SHEET['CAT'] == 1) |
                         (SHEET['AAH'] == 1) |
                         (SHEET['DAA'] == 1) |
                         (SHEET['IAA'] == 1) |
                         (SHEET['APVC'] == 1) |
                         (SHEET['DSVC'] == 1) |
                         (SHEET['PAS'] == 1)].index,
                         axis = 0) \
             .dropna(axis = 0, thresh = 3) \
             .reset_index() \
             .drop(['level_0', 'DORV', 'CAT', 'AAH', 'DAA', 'IAA', 'APVC', 'DSVC', 'PAS'],
                   axis = 1)

metadata[metadata[['ASD', 'VSD', 'AVSD', 'ToF', 'TGA', 'CA', 'PA', 'PDA']] == 0] = 1
metadata[metadata[['ASD', 'VSD', 'AVSD', 'ToF', 'TGA', 'CA', 'PA', 'PDA']].isna()] = 0
metadata.to_csv("D:\\ImageCHD_dataset\\metadata.csv", sep = ',', index = False)