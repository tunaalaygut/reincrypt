import sys
import matplotlib.pyplot as plt
from util import str_to_ndarray


INPUT_RIMG_PATH = sys.argv[1]


def main():
    pixels = ""
    with open(INPUT_RIMG_PATH, "r+") as f:
        pixels = f.read().split("$")[0]

    plt.imshow(str_to_ndarray(pixels.strip()))
    plt.show()


if __name__ == "__main__":
    main()
