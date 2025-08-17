from numpy import ndarray as arr
from os import listdir
import shutil

def Filter_ImageCHD(image_ids: arr) -> bool:
    try:
        SRC_DIR = "D:\\ImageCHD_dataset\\"
        IMAGE_DIR = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\IMAGES\\"
        LABEL_DIR = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\LABELS\\"
        RM_IMAGE_DIR = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\RM_IMAGES\\"
        RM_LABEL_DIR = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\RM_LABELS\\"
        IMAGE_DEST = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\NEW_IMAGES\\"
        LABEL_DEST = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\NEW_LABELS\\"

        image_list = listdir(IMAGE_DIR)

        for id in image_ids:
            filename = str(id) + ".nii.gz"
            if str(id) + ".nii.gz" not in image_list:
                print("Add: ", id)
                shutil.copy(SRC_DIR + "ct_" + str(id) + "_image.nii.gz",
                            IMAGE_DEST + filename)
                shutil.copy(SRC_DIR + "ct_" + str(id) + "_label.nii.gz",
                            LABEL_DEST + filename)
            else:
                print("Exists: ", id)
                image_list.pop(image_list.index(filename))
        
        if len(image_list) != 0:
            for file in image_list:
                print("Remove: ", file)
                shutil.copy(IMAGE_DIR + file, RM_IMAGE_DIR + file)
                shutil.copy(LABEL_DIR + file, RM_LABEL_DIR + file)

        return True
    except Exception as e:
        print(e)
        return False