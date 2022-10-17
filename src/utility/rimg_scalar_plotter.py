import os
import sys
from natsort import natsorted
from plotter import plot_daily
from datetime import datetime

INPUT_RIMG_DIR = sys.argv[1]
CURRENCY = os.path.basename(INPUT_RIMG_DIR)


def main():
    dates = []
    scalars = []
    for rimg in natsorted(os.listdir(INPUT_RIMG_DIR)):
        with open(os.path.join(INPUT_RIMG_DIR, rimg)) as r:
            rimg_str = r.read()
            scalars.append(float(rimg_str.split("$")[1].strip()))
            dates.append(rimg_str.split("$")[-1].strip())
    plot_daily(daily_data=scalars, date_begin=dates[0], date_end=dates[-1], 
               title=f"Scalars by Date for {CURRENCY}", y_label="Scalar",
               output_dir=".", output_fname=f"{CURRENCY}_scalar",
               file_suffix=int(datetime.now().timestamp()), color="blue")


if __name__ == "__main__":
    main()
