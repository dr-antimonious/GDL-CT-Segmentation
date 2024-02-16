from Adj_Logic import Generate_Adj_Matrix
import os

DATASET_PATH = "D:\\ImageCHD_dataset\\ImageCHD_dataset\\"
FILES = os.listdir(DATASET_PATH)
FILES.remove('imageCHD_dataset_info.xlsx')

for f in FILES:
    if f.split('_')[2].split('.')[0] == 'label':
        FILES.remove(f)

