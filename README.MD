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

...

Downloaded raw data should have the following format for a crpytocurrency:
| Date                        | Open                     | High                     | Low                      | Close                    | Adj Close                | Volume |
|-----------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------|
| 2017-11-09    | 0.000224  | 0.000297   | 0.0002119   | 0.000282    | 0.0002   | 8605   |
| 2017-11-10    | 0.000284   | 0.000292   | 0.0001340    | 0.000264   | 0.0002    | 4201   |
| 2017-11-11    | 0.000263   | 0.000233   | 0.0001232    | 0.000194   | 0.0001   | 1266   |
| ... |  ... | ... | ... | ... | ... | ... |
|

### Creating `.RIMG` files

...
<!-- rimg viewer can be used to display the image, add this to paper and thesis -->


## Training & Validation

### Train a model
...

### Verify the model
...
