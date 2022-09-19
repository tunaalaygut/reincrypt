import sys
import matplotlib.pyplot as plt
import numpy as np


INPUT_RIMG_PATH = sys.argv[1]


def main():
    pixels = ""
    with open(INPUT_RIMG_PATH, "r+") as f:
        pixels = f.read().split("$")[0]

    plt.imshow(__str_to_ndarray(pixels.strip()))
    plt.show()

# TODO: Date reader uses this function as well. Make it available as util.
def __str_to_ndarray(str: str) -> np.ndarray:
    rows = str.split('\n')
    arr = np.ndarray((len(rows), len(rows[0].split())))
    
    for idx, row in enumerate(rows):
        arr[idx, :] = row.split()
    
    return arr


if __name__ == "__main__":
    main()
