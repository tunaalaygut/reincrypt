import os
import sys
import talib as ta
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
import io

from util import clean_anomalies


INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
IMAGE_SIZE = 32
INTERVAL_START = 7

def main():
    print(f"Reading files from {INPUT_DIR}")
    for input_file in os.listdir(INPUT_DIR):
        try:
            df = clean_anomalies(
                pd.read_csv(os.path.join(INPUT_DIR, input_file)))
            currency_symbol = input_file.split("_")[1].split(".")[0]
            print(f"\nTrying to create image data for {currency_symbol}.")
            tis = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(df)))
            intervals = range(INTERVAL_START, INTERVAL_START + IMAGE_SIZE)
            
            for interval in intervals:
                i = interval - intervals.start
                # Overlap Studies
                tis[i][0] = ta.EMA(real=df.Close, timeperiod=interval)  
                tis[i][1] = ta.DEMA(real=df.Close, timeperiod=interval)  
                tis[i][2] = ta.KAMA(real=df.Close, timeperiod=interval)  
                tis[i][3] = ta.MIDPOINT(real=df.Close, timeperiod=interval)  
                tis[i][4] = ta.SMA(real=df.Close, timeperiod=interval)  
                tis[i][5] = ta.T3(real=df.Close, timeperiod=interval, 
                                  vfactor=0.7)  
                tis[i][6] = ta.TRIMA(real=df.Close, timeperiod=interval)  
                tis[i][7] = ta.WMA(real=df.Close, timeperiod=interval)  
                tis[i][8] = ta.MA(real=df.Close, timeperiod=interval)  
                tis[i][9] = ta.MIDPRICE(high=df.High, low=df.Low, 
                                        timeperiod=interval)  
                # Momentum Indicators
                tis[i][10] = ta.ADX(high=df.High, low=df.Low, close=df.Close, 
                                    timeperiod=interval)
                tis[i][11] = ta.CCI(high=df.High, low=df.Low, close=df.Close, 
                                    timeperiod=interval)
                tis[i][12] = ta.CMO(real=df.Close, timeperiod=interval)
                tis[i][13] = ta.DX(high=df.High, low=df.Low, close=df.Close, 
                                   timeperiod=interval)
                tis[i][14] = ta.MOM(real=df.Close, timeperiod=interval)
                tis[i][15] = ta.MFI(high=df.High, low=df.Low, close=df.Close, 
                                    volume=df.Volume, timeperiod=interval)
                tis[i][16] = ta.RSI(real=df.Close, timeperiod=interval)
                tis[i][17] = ta.ULTOSC(high=df.High, low=df.Low, close=df.Close, 
                                       timeperiod1=interval, 
                                       timeperiod2=interval*2,
                                       timeperiod3=interval*4)
                tis[i][18] = ta.WILLR(high=df.High, low=df.Low, close=df.Close, 
                                      timeperiod=interval)
                tis[i][19] = ta.ADXR(high=df.High, low=df.Low, close=df.Close, 
                                     timeperiod=interval)
                tis[i][20] = ta.AROONOSC(high=df.High, low=df.Low,
                                         timeperiod=interval)
                tis[i][21] = ta.PLUS_DI(high=df.High, low=df.Low, 
                                        close=df.Close, timeperiod=interval)
                tis[i][22] = ta.PLUS_DM(high=df.High, low=df.Low,
                                        timeperiod=interval)
                tis[i][23] = ta.ROC(real=df.Close, timeperiod=interval)
                tis[i][24] = ta.ROCP(real=df.Close, timeperiod=interval)
                # Volatility Indicators
                tis[i][25] = ta.ATR(high=df.High, low=df.Low, close=df.Close, 
                                    timeperiod=interval)
                tis[i][26] = ta.NATR(high=df.High, low=df.Low, close=df.Close, 
                                     timeperiod=interval)
                # Statistic Functions
                tis[i][27] = ta.STDDEV(real=df.Close, timeperiod=interval)
                tis[i][28] = ta.LINEARREG(real=df.Close, timeperiod=interval)
                tis[i][29] = ta.VAR(real=df.Close, timeperiod=interval)
                tis[i][30] = ta.TSF(real=df.Close, timeperiod=interval)
                # Volume Indicators
                tis[i][31] = ta.ADOSC(high=df.High, low=df.Low, close=df.Close, 
                                      volume=df.Volume, fastperiod=interval-4,
                                      slowperiod=interval+3)  

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

                scalars.append(f'{str(scalar)}\n')
                dates.append(f"{str(df.Date[i])}")

            scalars = np.delete(np.array(scalars), delete_idx).tolist()
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
                    + "$\n" + scalar \
                    + "$\n" + dates[idx]
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
