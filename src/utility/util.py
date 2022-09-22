import numpy as np


def str_to_ndarray(str: str) -> np.ndarray:
    rows = str.split('\n')
    arr = np.ndarray((len(rows), len(rows[0].split())))
    
    for idx, row in enumerate(rows):
        arr[idx, :] = row.split()
    
    return arr
