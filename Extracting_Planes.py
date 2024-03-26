from nibabel import load
from numpy import ndarray, array, int32, uint8, append, flip

def Convert_To_Graph(image: ndarray, label: ndarray) -> tuple[ndarray, ndarray]:
    r"""
        Arguments:
            image (numpy.ndarray): Source coronary-CT image.
            label (numpy.ndarray): Ground truth segmentation.
        
        Returns:
            out (tuple[numpy.ndarray, numpy.ndarray]): Source coronary-CT image and ground truth segmentation as graphs.
    """
    img_array = array([], dtype = int32)
    label_array = array([], dtype = uint8)

    for i in range(0, image.shape[0]):
        if i % 2 == 1:
            img_array = append(img_array, flip(image[i, :]))
            label_array = append(label_array, flip(label[i, :]))
        else:
            img_array = append(img_array, image[i, :])
            label_array = append(label_array, label[i, :])
            
    label_array[label_array > 7] = 0
    return (img_array, label_array)

def Extract_And_Convert(path_to_image: str, path_to_label: str,
                        plane_type: str, plane_index: int) \
                        -> tuple[ndarray, ndarray]:
    r"""
        Arguments:
            path_to_image (str): Full path to the coronary-CT .nii.gz file.
            path_to_label (str): Full path to the segmentation label .nii.gz file.
            plane_type (str): One-character string with a value of 'A', 'C', or 'S'.
            plane_index (int): Index of plane to be extracted from the image and label.
        
        Returns:
            out (tuple[numpy.ndarray, numpy.ndarray]): Source coronary-CT image and ground truth segmentation as graphs.
    """
    match plane_type:
        case 'A': # Axial plane
            return Convert_To_Graph(image = load(path_to_image)\
                                    .get_fdata()[:, :, plane_index],
                                    label = load(path_to_label)\
                                    .get_fdata()[:, :, plane_index])
        case 'C': # Coronal plane
            return Convert_To_Graph(image = load(path_to_image)\
                                    .get_fdata()[:, plane_index, :],
                                    label = load(path_to_label)\
                                    .get_fdata()[:, plane_index, :])
        case 'S': # Sagittal plane
            return Convert_To_Graph(image = load(path_to_image)\
                                    .get_fdata()[plane_index, :, :],
                                    label = load(path_to_label)\
                                    .get_fdata()[plane_index, :, :])