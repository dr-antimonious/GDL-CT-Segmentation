import pandas as pd

sheet = pd.read_excel("D:\\ImageCHD_dataset\\ImageCHD_dataset\\imageCHD_dataset_info.xlsx") \
          .drop(['Unnamed: 17', 'ONLY FIRST 8', 'FIRST 8 + MORE', 'NORMAL', 'IGNORED'], axis = 1) \
          .drop(['DORV', 'CAT', 'AAH', 'DAA', 'IAA', 'APVC', 'DSVC', 'PAS'], axis = 1) \
          .drop(110, axis = 0) \
          .dropna(axis = 0, thresh = 2) \
          .reset_index() \
          .drop('level_0', axis = 1)

print(sheet)