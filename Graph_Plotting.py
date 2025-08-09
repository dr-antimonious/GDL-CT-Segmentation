import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Excel_Processing import ProcessSpreadsheets

DATASET_INFO_PATH = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\imageCHD_dataset_info.xlsx"
SCAN_INFO_PATH = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\imagechd_dataset_image_info.xlsx"

dataset_info = ProcessSpreadsheets(DATASET_INFO_PATH, SCAN_INFO_PATH)

nan_cols = ["ASD", "VSD", "AVSD", "ToF", "TGA", "CA", "PA", "PDA"]

for i in range(0, 54):
    dataset_info.loc[i, 'AGE'] = (dataset_info.loc[i, 'AcquisitionDate'] - dataset_info.loc[i, 'PatientBirthDate']).days / 365
    if dataset_info.loc[i, 'AGE'] < 1/12:
        dataset_info.loc[i, 'AgeCategory'] = '0-1m'
    elif (dataset_info.loc[i, 'AGE'] < 0.25) & (dataset_info.loc[i, 'AGE'] >= 1/12):
        dataset_info.loc[i, 'AgeCategory'] = '1-3m'
    elif (dataset_info.loc[i, 'AGE'] < 0.5) & (dataset_info.loc[i, 'AGE'] >= 0.25):
        dataset_info.loc[i, 'AgeCategory'] = '3-6m'
    elif (dataset_info.loc[i, 'AGE'] < 0.75) & (dataset_info.loc[i, 'AGE'] >= 0.5):
        dataset_info.loc[i, 'AgeCategory'] = '6-9m'
    elif (dataset_info.loc[i, 'AGE'] < 1) & (dataset_info.loc[i, 'AGE'] >= 0.75):
        dataset_info.loc[i, 'AgeCategory'] = '9-12m'
    elif (dataset_info.loc[i, 'AGE'] < 2) & (dataset_info.loc[i, 'AGE'] >= 1):
        dataset_info.loc[i, 'AgeCategory'] = '1-2y'
    elif (dataset_info.loc[i, 'AGE'] < 3) & (dataset_info.loc[i, 'AGE'] >= 2):
        dataset_info.loc[i, 'AgeCategory'] = '2-3y'
    elif (dataset_info.loc[i, 'AGE'] < 5) & (dataset_info.loc[i, 'AGE'] >= 3):
        dataset_info.loc[i, 'AgeCategory'] = '3-5y'
    elif (dataset_info.loc[i, 'AGE'] < 9) & (dataset_info.loc[i, 'AGE'] >= 5):
        dataset_info.loc[i, 'AgeCategory'] = '5-9y'
    elif (dataset_info.loc[i, 'AGE'] < 21) & (dataset_info.loc[i, 'AGE'] >= 9):
        dataset_info.loc[i, 'AgeCategory'] = '9-21y'
    else:
        dataset_info.loc[i, 'AgeCategory'] = '21y+'

# y = [dataset_info[dataset_info['AGE'] < 1/12].__len__(),
#      dataset_info[(dataset_info['AGE'] < 0.25) & (dataset_info['AGE'] >= 1/12)].__len__(),
#      dataset_info[(dataset_info['AGE'] < 0.5) & (dataset_info['AGE'] >= 0.25)].__len__(),
#      dataset_info[(dataset_info['AGE'] < 0.75) & (dataset_info['AGE'] >= 0.5)].__len__(),
#      dataset_info[(dataset_info['AGE'] < 1) & (dataset_info['AGE'] >= 0.75)].__len__(),
#      dataset_info[(dataset_info['AGE'] < 2) & (dataset_info['AGE'] >= 1)].__len__(),
#      dataset_info[(dataset_info['AGE'] < 3) & (dataset_info['AGE'] >= 2)].__len__(),
#      dataset_info[(dataset_info['AGE'] < 5) & (dataset_info['AGE'] >= 3)].__len__(),
#      dataset_info[(dataset_info['AGE'] < 9) & (dataset_info['AGE'] >= 5)].__len__(),
#      dataset_info[(dataset_info['AGE'] < 21) & (dataset_info['AGE'] >= 9)].__len__(),
#      dataset_info[(dataset_info['AGE'] >= 21)].__len__()]

y = np.array([])
for col in nan_cols:
    y = np.append(y, dataset_info[dataset_info[col] == 1].__len__())

plt.figure(figsize=(8, 4.8))
sns.countplot(data = dataset_info, x = 'AgeCategory', hue = 'PatientSex',
              order = ['0-1m', '1-3m', '3-6m', '6-9m', '9-12m', '1-2y', '2-3y', '3-5y', '5-9y', '9-21y', '21y+'])
plt.xlabel('Patient age')
plt.ylabel('Count')
plt.title('Pre-processed ImageCHD patient distribution')
plt.legend(title = 'Patient sex')
# plt.savefig('C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\Graphs\\pre-proc_patient_distribution.png')
plt.show()