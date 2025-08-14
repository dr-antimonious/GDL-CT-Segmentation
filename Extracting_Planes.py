from numpy import ndarray, array, int32, uint8

def Convert_To_Graph(image: ndarray, label: ndarray) -> tuple[ndarray, ndarray]:
    r"""
        Arguments:
            image (numpy.ndarray): Source coronary-CT image.
            label (numpy.ndarray): Ground truth segmentation.
        
        Returns:
            out (tuple[numpy.ndarray, numpy.ndarray]): Source coronary-CT image and ground truth segmentation as graphs.
    """
    img = array(image, dtype = int32)
    lab = array(label, dtype = uint8)
    img[1::2, :] = image[1::2, ::-1]
    lab[1::2, :] = label[1::2, ::-1]
    
    img = img.flatten()
    img = img.reshape((img.shape[0], 1))

    lab[lab > 7] = 0
    lab = lab.flatten()
    return (img, lab)

def Extract_And_Convert(im, la, plane_type: str, plane_index: int) \
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
            image = im[:, :, plane_index]
            label = la[:, :, plane_index]
        case 'C': # Coronal plane
            image = im[:, plane_index, :]
            label = la[:, plane_index, :]
        case 'S': # Sagittal plane
            image = im[plane_index, :, :]
            label = la[plane_index, :, :]
        case _:
            raise ValueError('Invalid plane_type in Extract_And_Convert', plane_type)
    return Convert_To_Graph(image, label)