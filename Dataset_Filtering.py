import shutil
from numpy import ndarray as arr
from tqdm import tqdm

def Filter_ImageCHD(image_ids: arr) -> bool:
    try:
        for id in tqdm(image_ids):
            shutil.copy("C:\\Users\\leotu\\Downloads\\ImageCHD_dataset\\ImageCHD_dataset\\ct_" \
                        + str(id) + "_image.nii.gz",
                        "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\IMAGES\\" \
                        + str(id) + ".nii.gz")
            shutil.copy("C:\\Users\\leotu\\Downloads\\ImageCHD_dataset\\ImageCHD_dataset\\ct_" \
                        + str(id) + "_label.nii.gz",
                        "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\LABELS\\" \
                        + str(id) + ".nii.gz")
        return True
    except Exception as e:
        print(e)
        return False