import pandas as pd

def ProcessSpreadsheets(dataset_info_path, scan_info_path) -> pd.DataFrame:
    r"""
    Method for processing ImageCHD's two Excel spreadsheets.
    SPREADSHEETS NEED TO BE PRE-PROCESSED THEMSELVES BEFORE CALLING THIS METHOD!

    Arguments:
      dataset_info_path (string): path to the imageCHD_dataset_info.xlsx file
      scan_info_path (string): path to the imageCHD_dataset_image_info.xlsx file
    """
    dataset_info = pd.read_excel(io = dataset_info_path, sheet_name = 'classification dataset')
    scan_info = pd.read_excel(io = scan_info_path, sheet_name = 'Sheet1')
    dataset_info = pd.concat([dataset_info, scan_info], axis = 1)

    # ignore = dataset_info[dataset_info["IGNORED"] > 0].index
    drop_cols = ["NORMAL", "ONLYFIRST8", "FIRST8+MORE", "NORMAL.1", "IGNORED",
                "TGA", "AAH", "IAA", "PAS", "CAT", "CA", "DORV", "DSVC", "idx",
                "PatientBirthDate1", "AcquisitionDate1",
                "PixelSpacing1", "PixelSpacing2", "calculate_z_thick",
                "ManufacturerModelName", "AGE", "UNKNOWN"]

    dataset_info = dataset_info.drop(drop_cols, axis = 1) \
      .reset_index() \
        .drop("level_0", axis = 1)

    nan_cols = ["ASD", "VSD", "AVSD", "ToF", "PA", "PDA", "APVC", "DAA"]
    dataset_info['COUNT'] = 0
    for col in nan_cols:
        mask = dataset_info[dataset_info[col] != 1].index
        dataset_info.loc[mask, col] = 0
        dataset_info['COUNT'] += dataset_info[col]
    
    return dataset_info