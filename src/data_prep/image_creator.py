import os
import talib as ta
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
import io

#TODO: Make this parametric. Obviously...
INPUT_FILE = "../../crypto_data/2908_BTC-USD.csv"

df = pd.read_csv(INPUT_FILE)
tis = np.zeros((18, 18, len(df)))
intervals = range(7, 25)

def main():
    for interval in intervals:
        i = interval - 7
        tis[i][0] = ta.EMA(real=df.Close, timeperiod=interval)
        tis[i][1] = ta.DEMA(real=df.Close, timeperiod=interval)
        tis[i][2] = ta.KAMA(real=df.Close, timeperiod=interval)
        tis[i][3] = ta.MIDPOINT(real=df.Close, timeperiod=interval)
        tis[i][4] = ta.SMA(real=df.Close, timeperiod=interval)
        #TODO: What is vfactor?
        tis[i][5] = ta.T3(real=df.Close, timeperiod=interval, vfactor=0.7)
        tis[i][6] = ta.TRIMA(real=df.Close, timeperiod=interval)
        tis[i][7] = ta.WMA(real=df.Close, timeperiod=interval)
        tis[i][8] = ta.ADX(high=df.High, 
                        low=df.Low,
                        close=df.Close, 
                        timeperiod=interval)
        tis[i][9] = ta.CCI(high=df.High, 
                        low=df.Low, 
                        close=df.Close, 
                        timeperiod=interval)
        tis[i][10] = ta.CMO(real=df.Close, timeperiod=interval)
        tis[i][11] = ta.DX(high=df.High, 
                        low=df.Low, 
                        close=df.Close, 
                        timeperiod=interval)
        tis[i][12] = ta.MOM(real=df.Close, timeperiod=interval)
        tis[i][13] = ta.MFI(high=df.High, 
                            low=df.Low, 
                            close=df.Close, 
                            volume=df.Volume, 
                            timeperiod=interval)
        tis[i][14] = ta.RSI(real=df.Close, timeperiod=interval)
        #TODO: What are these timeperiods?
        tis[i][15] = ta.ULTOSC(high=df.High, 
                            low=df.Low, 
                            close=df.Close, 
                            timeperiod1=interval,
                            timeperiod2=interval*2,
                            timeperiod3=interval*4)
        tis[i][16] = ta.WILLR(high=df.High, 
                            low=df.Low, 
                            close=df.Close, 
                            timeperiod=interval)
        tis[i][17] = ta.ATR(high=df.High,
                            low=df.Low, 
                            close=df.Close, 
                            timeperiod=interval)
        
    images = np.zeros((len(df), 18, 18))

    for day_idx in range(len(df)):
        images[day_idx] = tis[:, :, day_idx]
        
    images_new = images.copy()

    delete_idx = []
    for idx in range(len(images_new)):
        if np.isnan(images[idx]).any():
            delete_idx.append(idx)
    images_new = np.delete(images_new, delete_idx, axis=0)

    scaled_images = []
    # TODO: Consider column based scaling
    for idx, im in enumerate(images_new):
        scaled_im = minmax_scale(im.astype(int), (0,255))
        scaled_images.append(scaled_im)


    # TODO: what is this value referred to as in the paper?
    gains = []

    for i in range(len(df)):
        if i == len(df) - 1:
            delta = 0
        else:
            delta = df.Close[i] - df.Close[i+1]
        gains.append(f'{str(100 * delta / df.Close[i])}\n')

    for d in delete_idx:
        del gains[d]

    os.makedirs('output', exist_ok=True)

    for idx, (gain, scaled_image) in enumerate(zip(gains, scaled_images), 
                                               start=0):
        image_bytes = io.BytesIO()
        np.savetxt(image_bytes, scaled_image, fmt="%03d")
        mystr = image_bytes.getvalue().decode() + "$\n" + gain
        image_bytes.close()

        with open(f'output/image_{idx}.rimg', 'w+') as f:
            f.write(mystr)


if __name__ == "__main__":
    main()
