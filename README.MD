# Reincrypt Project

## Introduction & Aim

In this project we aim to develop a `python` package that can:

- Download raw cryptocurrency data between given dates using Yahoo Finance API (with `yfinance` package) with the following format.
- For each date calculate various technical indicators using the raw data.
- Normalize the technical indicator values into interval [0-255].
- Create a `.RIMG` file for each date using normalized technical indicator values. A `.RIMG` file is a special file format that contains a grayscale image and the date.
- Train a DQN to map the state (grayscale image) into the optimal action (Long, Short, Neutral).
- Validate the models decision using Top/Bottom K and Market Neutral portfolios.


# Usage

## Setting Up the Environment

First, clone the repository with your preferred method. For example, to clone using **GitHub CLI**:
```bash
gh repo clone tunaalaygut/reincrypt && cd reincrypt
```

Then, create a virtual environment, activate it, and install the requirements 
```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

Now you are ready to move on.

## Data Preparation

This section covers the necessary steps to prepare your data for training.

### Downloading raw data

To download raw cryptocurrency data `ticker_downloader.py` module can be used. However, before you do that you need to create a list of currencies you want to download and write them to a `.txt` file. For example;

```
# currencies.txt
KCS-USD
WGR-USD
MYB-USD
...
```

```bash
cd src/data_prep
python ticker_downloader.py currencies.txt
```

Running the command above will produce an output similar to the following:

```bash
[*********************100%***********************]  1 of 1 completed
KCS-USD downloaded and written to file.
[*********************100%***********************]  1 of 1 completed
WGR-USD downloaded and written to file.
[*********************100%***********************]  1 of 1 completed
MYB-USD downloaded and written to file.
...
```

Downloaded raw data should have the following format for each crpytocurrency:
| Date                        | Open                     | High                     | Low                      | Close                    | Adj Close                | Volume |
|-----------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------|
| 2017-11-09    | 0.000224  | 0.000297   | 0.0002119   | 0.000282    | 0.0002   | 8605   |
| 2017-11-10    | 0.000284   | 0.000292   | 0.0001340    | 0.000264   | 0.0002    | 4201   |
| 2017-11-11    | 0.000263   | 0.000233   | 0.0001232    | 0.000194   | 0.0001   | 1266   |
| ... |  ... | ... | ... | ... | ... | ... |
|

Ideally, you would have two files `training_currencies.txt` and `verification_currencies.txt` to download training and verification currency data separately. Or you can separate them later. 

### Creating `.RIMG` files
Once the raw data is downloaded you would need to create `.RIMG` files. `image_creator.py` module does exactly that.

It works by calculating 32 pre-defined and clustered technical indicators for 32 different time intervals, normalizing the calculated values between [0, 255]. Hence, creating a 32x32 grayscale image for each day for each currency.

It also adds the date and a scalar. Scalar is the percentage difference of the currency's closing value from the previous day.

A *minimized* `.rimg` file looks like the following:

```
000 000 001 005 ...
005 008 000 000 ...
012 009 000 000 ...
020 010 011 005 ...
...
$
-4.694593226213477
$
2018-06-20
```

#### Visualizing `.RIMG` images
To visualize the created `.rimg` file(s) one can use `utility/rimg_viewer.py` module.

To use it:

```bash
python rimg_viewer.py <FILENAME>.rimg 
```

Which will produce an image similar to following image:

![alt text](sample_rimg.png)


## Training & Validation

### Train a model
Once the data is ready, a training configuration needs to be defined in a `.JSON` format with the following fields:

- `num_actions`: Number of actions the DQN will map states into
- `max_iterations`: Number of iterations the training will take 
- `learning_rate`: Learning rate for the gradient step
- `epsilon_init`: Initial epsilon value
- `epsilon_min`: Minimum epsilon value
- `memory_size`: Size of the experience replay memory
- `B`: Number of iterations before updating the online network weights
- `C`: Number of iterations before updating the target network weights
- `gamma`: Discount factor
- `batch_size`: Batch size
- `penalty`: Penalty to apply for position change
- `patch_size`: Transformer patch size
- `projection_dim`: Transformer projection dimension
- `enable_resizing`: Whether to enable image resizing
- `resized_image_size`: Resized image size if `enable_resizing` 
- `num_heads`: Number of heads in ViT
- `mlp_head_units`: Definition of MLP units in ViT
- `transformer_units`: Definition of transformer units in ViT
- `transformer_layers`: Number of transformer layers in ViT
- `description`: Description of the configuration

A sample training configuration is as follows:

```JSON
{
  "num_actions": 3,
  "max_iterations": 50000,
  "learning_rate": 0.0001,
  "epsilon_init": 1.0,
  "epsilon_min": 0.1,
  "memory_size": 1000,
  "B": 10,
  "C": 1000,
  "gamma": 0.99,
  "batch_size": 32,
  "penalty": 0.05,
  "patch_size": 6,
  "projection_dim": 64,
  "enable_resizing": false,
  "resized_image_size": 36,
  "num_heads": 4,
  "mlp_head_units": [
    2048,
    1024
  ],
  "transformer_units": [
    128,
    64
  ],
  "transformer_layers": 8,
  "description": "Using k-means clustered technical indicators."
}
```

With the configuration created, simply run 

```bash
python main.py -i <PATH_TO_TRAINING_DATA_DIR> -c <CONFIG_NAME>.json
```

- `-i` or `--input-path` argument is used to indicate the input data directory. 
- `-c` or `--config'` argument is used to specify which config file to use.

### Verify the model
Output of the trained model will be placed under the directory `$WORKSPACE/reincrypt/src/dqn/output/<CONFIG_NAME>` and it will have a structure similar to this:

```bash
<CONFIG_NAME>
├── <CONFIG_NAME>_model/
├── training_<TIMESTAMP>.json
└── training_<TIMESTAMP>.png
```

Where `<CONFIG_NAME>_model/` will be the keras model output, .json is the training logs and the .png file is the loss plot of the training. Keras model output can be used to load the model and make inferences.

To create portfolios and validate the model on the verification data set:

```bash
python main.py -v -i <PATH_TO_VERIFICATION_DATA_DIR> -c <CONFIG_NAME> -m <PATH_TO_CONFIG_NAME_model_DIR>
```

- `-i` or `--input-path` argument is used to indicate the input data directory. 
- `-v` or `--verification` flag is used to indicate the program will be run in verification mode.
- `-c` or `--config'` argument is used to specify which config file to use.
- `-m` or `--model` argument is used to specify the keras model output created in training phase. Required if `-v` flag is set.

After the verification ...
 