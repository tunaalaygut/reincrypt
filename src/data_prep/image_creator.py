import os
import sys
sys.path.append("..")
import talib as ta
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
from utility.r_utils import clean_anomalies, bound_scalar, neutralize_series
import io
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--indir', required=True, type=str, 
                    help="Input (raw) data directory.")
parser.add_argument('-o', '--outdir', required=True, type=str, 
                    help="Output (rimg) data directory.")
parser.add_argument('-t', '--type', required=True, type=str, 
                    help="Image creation type <train> or <verification>")
args = vars(parser.parse_args())

INPUT_DIR = args["indir"]
OUTPUT_DIR = args["outdir"]
TYPE = args["type"]
IS_TRAIN = (TYPE == "train")
IMAGE_SIZE = 32
INTERVAL_START = 7
TICKERS = []

if IS_TRAIN:
    with open("../../training_currencies.txt", "r+") as f: 
        TICKERS = f.read().splitlines()
else:
    with open("../../verification_currencies.txt", "r+") as f: 
        TICKERS = f.read().splitlines()

def main():
    print(f"Reading files from {INPUT_DIR}")
    for input_file in os.listdir(INPUT_DIR):
        try:
            currency_symbol = input_file.split("_")[1].split(".")[0]
            if currency_symbol not in TICKERS:
                continue
            df = pd.read_csv(os.path.join(INPUT_DIR, input_file))
            df = (clean_anomalies(df) if IS_TRAIN else df)
            
            print(f"\nTrying to create image data for {currency_symbol}.")
            tis = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(df)))
            intervals = range(INTERVAL_START, INTERVAL_START + IMAGE_SIZE)
            
            for interval in intervals:
                i = interval - intervals.start
                # Cluster 0 based on technical indicator clustering
                tis[i][0] = ta.EMA(real=df.Close, timeperiod=interval)  
                tis[i][1] = ta.KAMA(real=df.Close, timeperiod=interval)  
                tis[i][2] = ta.MIDPOINT(real=df.Close, timeperiod=interval)  
                tis[i][3] = ta.SMA(real=df.Close, timeperiod=interval)  
                tis[i][4] = ta.T3(real=df.Close, timeperiod=interval, 
                                  vfactor=0.7)  
                tis[i][5] = ta.TRIMA(real=df.Close, timeperiod=interval)  
                tis[i][6] = ta.WMA(real=df.Close, timeperiod=interval)  
                tis[i][7] = ta.MA(real=df.Close, timeperiod=interval)  
                tis[i][8] = ta.MIDPRICE(high=df.High, low=df.Low, 
                                        timeperiod=interval)  
                # Cluster 1 based on technical indicator clustering
                tis[i][9] = ta.DEMA(real=df.Close, timeperiod=interval)  
                tis[i][10] = ta.CMO(real=df.Close, timeperiod=interval)
                tis[i][11] = ta.RSI(real=df.Close, timeperiod=interval)
                tis[i][12] = ta.ULTOSC(high=df.High, low=df.Low, close=df.Close, 
                                       timeperiod1=interval, 
                                       timeperiod2=interval*2,
                                       timeperiod3=interval*4)
                tis[i][13] = ta.PLUS_DI(high=df.High, low=df.Low, 
                                        close=df.Close, timeperiod=interval)
                tis[i][14] = ta.ADOSC(high=df.High, low=df.Low, close=df.Close, 
                                      volume=df.Volume, fastperiod=interval-4,
                                      slowperiod=interval+3) 
                # Cluster 2 based on technical indicator clustering
                tis[i][15] = ta.ADX(high=df.High, low=df.Low, close=df.Close, 
                                    timeperiod=interval)
                tis[i][16] = ta.CCI(high=df.High, low=df.Low, close=df.Close, 
                                    timeperiod=interval)
                tis[i][17] = ta.DX(high=df.High, low=df.Low, close=df.Close, 
                                   timeperiod=interval)
                tis[i][18] = ta.MOM(real=df.Close, timeperiod=interval)
                tis[i][19] = ta.MFI(high=df.High, low=df.Low, close=df.Close, 
                                    volume=df.Volume, timeperiod=interval)
                tis[i][20] = ta.WILLR(high=df.High, low=df.Low, close=df.Close, 
                                      timeperiod=interval)
                tis[i][21] = ta.ADXR(high=df.High, low=df.Low, close=df.Close, 
                                     timeperiod=interval)
                tis[i][22] = ta.AROONOSC(high=df.High, low=df.Low,
                                         timeperiod=interval)
                tis[i][23] = ta.ROC(real=df.Close, timeperiod=interval)
                tis[i][24] = ta.ROCP(real=df.Close, timeperiod=interval)
                tis[i][25] = ta.TSF(real=df.Close, timeperiod=interval)
                tis[i][26] = ta.LINEARREG(real=df.Close, timeperiod=interval)
                # Cluster 3 based on technical indicator clustering
                tis[i][27] = ta.PLUS_DM(high=df.High, low=df.Low,
                                        timeperiod=interval)
                tis[i][28] = ta.ATR(high=df.High, low=df.Low, close=df.Close, 
                                    timeperiod=interval)
                tis[i][29] = ta.NATR(high=df.High, low=df.Low, close=df.Close, 
                                     timeperiod=interval)
                tis[i][30] = ta.STDDEV(real=df.Close, timeperiod=interval)
                tis[i][31] = ta.VAR(real=df.Close, timeperiod=interval)

            images = np.zeros((len(df), IMAGE_SIZE, IMAGE_SIZE))

            for day_idx in range(len(df)):
                images[day_idx] = tis[:, :, day_idx]
                
            images_new = images.copy()

            delete_idx = []
            for idx in range(len(images_new)):
                if np.isnan(images[idx]).any():
                    delete_idx.append(idx)
            images_new = np.delete(images_new, delete_idx, axis=0)

            scalars = []
            dates = []
            for i in range(len(df)):
                if i == len(df) - 1:
                    delta = 0
                else:
                    delta = df.Close[i+1] - df.Close[i]

                if df.Close[i] != 0:
                    scalar = 100 * delta / df.Close[i]
                elif df.Close[i] == 0 and df.Close[i+1] - df.Close[i] == 0:
                    scalar = 0
                else:
                    scalar = 100
                scalar = (bound_scalar(scalar) if IS_TRAIN else scalar)
                scalars.append(f'{str(scalar)}\n')
                dates.append(f"{str(df.Date[i])}")

            scalars = np.delete(np.array(scalars), delete_idx).astype(float)\
                .tolist()
            scalars = (neutralize_series(scalars) if IS_TRAIN else scalars)
            dates = np.delete(np.array(dates), delete_idx).tolist()

            os.makedirs(os.path.join(OUTPUT_DIR, currency_symbol), 
                        exist_ok=True)

            scaled_images = []
            for idx, im in enumerate(images_new):
                scaled_im = minmax_scale(im, (0,255)).astype(int)
                scaled_images.append(scaled_im)

            for idx, (scalar, image) in enumerate(zip(scalars, scaled_images),
                                                  start=0):
                image_bytes = io.BytesIO()
                np.savetxt(image_bytes, image, fmt="%03d")
                output_str = image_bytes.getvalue().decode() \
                    + "$\n" + str(scalar) \
                    + "\n$\n" + dates[idx]
                image_bytes.close()

                with open(f'{OUTPUT_DIR}/{currency_symbol}/image_{idx}.rimg', 
                        'w+') as f:
                    f.write(output_str)
            print(f"Image data for {currency_symbol} created successfully.\n")
        except Exception as e:
            print(f"Failed {currency_symbol}. {e}\n")
            pass


if __name__ == "__main__":
    main()
