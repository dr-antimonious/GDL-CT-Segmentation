from numpy import ndarray, array, int32, uint8, float32, \
    hypot, sqrt, maximum, indices, stack
from scipy.ndimage import sobel, uniform_filter

def Compute_Additional_Features(image: ndarray) -> ndarray:
    height, width = image.shape
    features = []

    intensity = image.astype(float32)
    features.append(intensity)

    grad_x = sobel(intensity, axis = 0)
    grad_y = sobel(intensity, axis = 1)
    grad = hypot(grad_x, grad_y)
    features.append(grad)

    mean = uniform_filter(intensity)
    features.append(mean)

    sq = uniform_filter(intensity ** 2)
    std = sqrt(maximum(sq - mean ** 2, 1e-6))
    features.append(std)

    y_idx, x_idx = indices((height, width))
    norm_x_idx = x_idx / width
    norm_y_idx = y_idx / height
    features.append(norm_x_idx)
    features.append(norm_y_idx)

    return stack(features, axis = 2)

def Convert_To_Graph(image: ndarray, label: ndarray) -> tuple[ndarray, ndarray]:
    r"""
        Arguments:
            image (numpy.ndarray): Source coronary-CT image.
            label (numpy.ndarray): Ground truth segmentation.
        
        Returns:
            out (tuple[numpy.ndarray, numpy.ndarray]): Source coronary-CT image and ground truth segmentation as graphs.
    """
    img = array(image, dtype = int32)
    img = Compute_Additional_Features(img)
    lab = array(label, dtype = uint8)
    img[1::2, :, :] = img[1::2, ::-1, :]
    lab[1::2, :] = lab[1::2, ::-1]
    
    img = img.flatten()
    img = img.reshape((-1, 6))

    lab[lab > 7] = 0
    lab = lab.flatten()
    return (img, lab)

def Extract_And_Convert(im, la, idx, plane_type: str, plane_index: int) \
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
    print("Extract_And_Convert ", idx, plane_index, plane_type)
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