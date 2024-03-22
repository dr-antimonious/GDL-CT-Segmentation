import pandas as pd
from Dataset_Filtering import Filter_ImageCHD
import numpy as np

dataset = pd.read_csv("C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\train_dataset_info.csv")
if (Filter_ImageCHD(np.unique(dataset['index']))):
    print("Filtered dataset successfully.")
else:
    print("Filtration failed.")