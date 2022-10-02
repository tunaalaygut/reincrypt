import numpy as np
from keras.models import load_model
from model import ViT


def test_mnp(X, y, config, model_path, logger):
    """
    Test using market neutralized portfolio
    """
    network = ViT(config)
    network.model = load_model(model_path, compile=False)
    outcome = __validate_neutralized_portfolio(network, X, y)
    
    logger.num_currencies = outcome[0]
    logger.position_change = outcome[1]
    logger.cumulative_asset = outcome[2]


def __validate_neutralized_portfolio(vit: ViT, X, y):
    num_currencies = len(X)
    days = len(X[0])
    cur_actions = np.zeros((num_currencies, 3))

    prev_alpha = np.zeros(num_currencies)
    cur_alpha = np.zeros(num_currencies)
    position_change = 0

    cur_reward = np.zeros(num_currencies)
    avg_daily_return = np.zeros(days)

    cumulative_asset = 1

    for t in range(days-1):
        for c in range(num_currencies):
            cur_state = X[c][t]
            _, eta = vit.q_value(cur_state, is_training=False)
            cur_actions[c] = eta

        cur_alpha = __get_neutralized_portfolio(cur_actions, num_currencies)

        for c in range(num_currencies):
            cur_reward[c] = np.round(cur_alpha[c] * y[c][t], 8)
            avg_daily_return[t] \
                = np.round(avg_daily_return[t] + cur_reward[c], 8)
            position_change = np.round(position_change
                                       + abs(cur_alpha[c] - prev_alpha[c]), 8)
            prev_alpha[c] = cur_alpha[c]

    for t in range(days):
        cumulative_asset = np.round(
            cumulative_asset
            + (cumulative_asset * avg_daily_return[t] * 0.01), 8)

    return num_currencies, position_change, cumulative_asset


def __get_neutralized_portfolio(actions, num_currencies):
    alpha = np.zeros(num_currencies)
    avg = 0

    for c in range(num_currencies):
        alpha[c] = 1 - np.argmax(actions[c])
        avg = avg + alpha[c]

    avg = np.round(avg / num_currencies, 4)

    sum_alpha = 0

    for c in range(num_currencies):
        alpha[c] = np.round(alpha[c] - avg, 4)
        sum_alpha = np.round(sum_alpha + abs(alpha[c]), 4)

    if sum_alpha == 0:
        return alpha

    for c in range(num_currencies):
        alpha[c] = np.round(alpha[c] / sum_alpha, 8)

    return alpha


if __name__ == "__main__":
    pass
