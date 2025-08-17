import numpy as np
from tqdm import tqdm

def Calculate_Index(i: int, j: int, n_cols: int) -> int:
    return n_cols * i + j * (i % 2 == 0) + (n_cols - 1 - j) * (i % 2 == 1)

def Generate_Adj_Matrix(n_cols: int) -> np.ndarray:
    row_array = np.zeros(8 * 512 * n_cols, dtype = np.uint32)
    col_array = np.zeros(8 * 512 * n_cols, dtype = np.uint32)

    for i in range(0, 512):
        index = 8 * i * n_cols if i % 2 == 0 else 8 * (i + 1) * n_cols - 1

        for j in range(0, n_cols):
            mask = list(range(index, index + 8)) if i % 2 == 0 else list(range(index, index - 8, -1))
            mask.sort()

            row_array[mask] = Calculate_Index(i, j, n_cols)

            i_m = 511 if i == 0 else i - 1
            i_p = 0 if i == 511 else i + 1
            
            j_m = n_cols - 1 if j == 0 else j - 1
            j_p = 0 if j == n_cols - 1 else j + 1
            
            con_indices = np.array([Calculate_Index(i, j_m, n_cols),
                                    Calculate_Index(i_m, j_m, n_cols),
                                    Calculate_Index(i_m, j, n_cols),
                                    Calculate_Index(i_m, j_p, n_cols),
                                    Calculate_Index(i, j_p, n_cols),
                                    Calculate_Index(i_p, j_p, n_cols),
                                    Calculate_Index(i_p, j, n_cols),
                                    Calculate_Index(i_p, j_m, n_cols)],
                                    dtype = np.uint32)
            con_indices.sort()

            col_array[mask] = con_indices
            index = index + 8 if i % 2 == 0 else index - 8

    return np.array([row_array, col_array], dtype = np.uint32)

def Generate_All_Adj_Matrices(n_cols_arr: np.ndarray) -> bool:
    try:
        for n_cols in tqdm(n_cols_arr):
            adj_mat = Generate_Adj_Matrix(n_cols = n_cols)
            np.save(file = "C:\\Users\\leotu\\OneDrive\\Documents\\ImageCHD_dataset\\NEW_ADJACENCY\\Adj_Mat_" \
                    + str(n_cols) + ".npy", arr = adj_mat)
            print(str(n_cols) + " saved")
        return True
    except Exception as e:
        print(e)
        return False