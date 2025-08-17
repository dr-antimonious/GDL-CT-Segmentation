from datetime import datetime
from nibabel.loadsave import load
from numpy import floor, ceil, unique
from os import listdir
from os.path import exists
import pandas as pd
from random import sample, seed
from tqdm import tqdm

from Excel_Processing import ProcessSpreadsheets
from Dataset_Filtering import Filter_ImageCHD
from Adj_Logic import Generate_All_Adj_Matrices

DIRECTORY           = "D:\\CTProject\\CTProject\\GDL-CT-Segmentation\\"
SRC_DIR             = "D:\\ImageCHD_dataset\\"
DEST_DIR            = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\"
DATASET_INFO_PATH   = DIRECTORY + "imageCHD_dataset_info.xlsx"
SCAN_INFO_PATH      = DIRECTORY + "imagechd_dataset_image_info.xlsx"
DEST_CSVS           = ['new_train_dataset_info.csv',
                       'new_eval_dataset_info.csv',
                       'new_test_dataset_info.csv']
SPLITS              = ['train', 'eval', 'test']
PLANES              = ['A', 'C', 'S']
TRAIN_PORTION       = 0.7
EVAL_PORTION        = 0.2
TEST_PORTION        = 0.1

def get_counts(data: pd.DataFrame, names: list) -> \
    dict[str, dict[str, int]]:
    chd_counts = [int(len(data[data[chd] == 1])) for chd in names]
    chds = sorted(zip(names, chd_counts), key = lambda x: (x[1], x[0]))

    chd, count = chds[0]
    train_count = int(floor(count * TRAIN_PORTION))
    eval_count = int(ceil(count * EVAL_PORTION))
    test_count = int(ceil(count * TEST_PORTION))

    if (train_count + eval_count + test_count) > count:
        eval_count -= 1
    
    if (eval_count > 1) and (count < 10):
        eval_count -= 1
        train_count += 1
    
    if test_count > 1:
        test_count -= 1
        train_count += 1

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
    chd_names = ["ASD", "VSD", "AVSD", "ToF",
                 "PA", "PDA", "APVC", "DAA"]
    
    if not exists(DIRECTORY + 'patient_info.csv'):
        dataset_info = ProcessSpreadsheets(
            DIRECTORY + 'imageCHD_dataset_info.xlsx',
            DIRECTORY + 'imagechd_dataset_image_info.xlsx'
        )

        ## Remove patients older than 10y
        age_mask = [(x.date() - y.date()).days > 10 * 366 \
                for x, y in zip(dataset_info['AcquisitionDate'],
                                dataset_info['PatientBirthDate'])]

        dataset_info = dataset_info.drop(dataset_info.loc[age_mask].index) \
                .reset_index() \
                    .drop("level_0", axis = 1)
        
        chd_mask = dataset_info[chd_names[0]] == 0
        for c in chd_names[1:]:
            chd_mask &= dataset_info[c] == 0
        
        dataset_info = dataset_info.drop(dataset_info.loc[chd_mask].index) \
                .reset_index() \
                    .drop("level_0", axis = 1)
        dataset_info.to_csv(DIRECTORY + 'patient_info.csv', index = False)
    else:
        dataset_info = pd.read_csv(DIRECTORY + 'patient_info.csv')

    chds = get_counts(dataset_info, chd_names)
    
    dataset_info = pd.read_csv(DIRECTORY + 'patient_info.csv',
                               index_col = False)
    chd_names = ["ASD", "VSD", "AVSD", "ToF",
                 "PA", "PDA", "APVC", "DAA"]
    
    # indices = unique(array(
    #     [x for xx in \
    #      [dataset_info[dataset_info[c] == 1]['index'].values.tolist() for c in chd_names] \
    #           for x in xx]
    # ))
    # Filter_ImageCHD(indices)

    axial_count = {x: load(SRC_DIR + "ct_" + str(x) + "_image.nii.gz") \
                   .header['dim'][3] for x in dataset_info['index'].sort_values()}
    
    print(axial_count)

    datasets = {split: [] for split in SPLITS}

    for chd in tqdm(chds.keys()):
        seed(datetime.now().microsecond)

        for split in SPLITS:
            mask = dataset_info[chd] == 1
            chd_info = dataset_info[mask]
            
            samples = sample(chd_info['index'].values.tolist(),
                             k = chds[chd][split])
            
            for sam in samples:
                for plane in PLANES:
                    for idx in range(512 if plane != 'A' else axial_count[sam]):
                        datasets[split].append({
                            'index': sam,
                            'Adjacency_count': axial_count[sam] if plane != 'A' else 512,
                            'Type': plane,
                            'Indice': idx
                        })

            sample_mask = dataset_info['index'].isin(samples)
            dataset_info.drop(dataset_info.loc[sample_mask].index, inplace = True)
    
    df_datasets = {
        split: pd.DataFrame(datasets[split]) for split in SPLITS
    }

    for i, split in enumerate(SPLITS):
        print(df_datasets[split])
        print(len(df_datasets[split]))
        df_datasets[split].to_csv(DEST_DIR + DEST_CSVS[i], index = False)

if __name__ == '__main__':
    main()