from rlogging.reincrypt_logger import VerificationLogger
from model import ViT
from keras.models import load_model
import numpy as np
import sys
sys.path.append("..")


def test_portfolio(X, y, config, model_path, logger, K=None):
    """
    Test using top/bottom k portfolio or market neutral portfolio 
    """
    network = ViT(config)
    network.model = load_model(model_path, compile=False)
    outcome = None

    if K:
        outcome = __validate_top_bottom_k_portfolio(network, X, y, K, logger)
        logger.portfolio_method = "top/bottom k"
    else:
        outcome = __validate_neutralized_portfolio(network, X, y, logger)
        logger.portfolio_method = "market neutral"

    logger.position_change = outcome[0]
    logger.final_cumulative_asset = outcome[1]


def __validate_top_bottom_k_portfolio(vit: ViT, X, y, K: float,
                                      logger: VerificationLogger):
    num_currencies = len(X)
    days = len(X[0])

    prev_alpha = np.zeros(num_currencies)
    cur_alpha = np.zeros(num_currencies)
    pos_change = 0

    cur_reward = np.zeros(num_currencies)
    avg_daily_ret = np.zeros(days)

    cumulative_asset = 1

    cur_action_value = np.zeros((num_currencies, 3))
    long_signals = np.zeros(num_currencies)

    upper_threshold = 0
    lower_threshold = 0

    X = np.array(X)

    for t in range(days):
        cur_action_value, _ = vit.q_value(X[:, t], is_training=False)
        long_signals = cur_action_value[:, 0] - cur_action_value[:, 2]

        upper_threshold, lower_threshold = __get_kth(long_signals, K)
        cur_alpha = __get_top_bottom_portfolio(upper_threshold, lower_threshold,
                                               long_signals, num_currencies)
        for c in range(num_currencies):
            cur_reward[c] = np.round(cur_alpha[c] * y[c][t], 8)
            avg_daily_ret[t] = np.round(avg_daily_ret[t] + cur_reward[c], 8)
            pos_change = \
                np.round(pos_change + abs(cur_alpha[c] - prev_alpha[c]), 8)
            prev_alpha[c] = cur_alpha[c]

        logger.add_daily_results(cumulative_asset, avg_daily_ret[t])
        cumulative_asset = \
            round(cumulative_asset
                  + (cumulative_asset * avg_daily_ret[t] * 0.01), 8)

    return pos_change, cumulative_asset


def __get_top_bottom_portfolio(upper_threshold, lower_threshold, long_signals,
                               num_currencies):
    alpha = np.zeros(num_currencies)
    sum = 0

    for c in range(num_currencies):
        if long_signals[c] >= upper_threshold:
            alpha[c] = 1
            sum = sum + 1
        elif long_signals[c] <= lower_threshold:
            alpha[c] = -1
            sum = sum+1
        else:
            alpha[c] = 0

    if sum == 0:
        return alpha

    for c in range(num_currencies):
        alpha[c] = np.round(alpha[c] / float(sum), 8)

    return alpha


def __get_kth(long_signals, K):
    num = int(len(long_signals) * K)
    sorted_long_signals = np.sort(long_signals)

    return (sorted_long_signals[len(long_signals) - num],
            sorted_long_signals[num-1])


def __validate_neutralized_portfolio(vit: ViT, X, y,
                                     logger: VerificationLogger):
    num_currencies = len(X)
    days = len(X[0])
    cur_actions = np.zeros((num_currencies, 3))

    prev_alpha = np.zeros(num_currencies)
    cur_alpha = np.zeros(num_currencies)
    position_change = 0

    cur_reward = np.zeros(num_currencies)
    avg_daily_return = np.zeros(days)

    cumulative_asset = 1
    X = np.array(X)  # Cast list to np array

    for t in range(days):
        _, cur_actions = vit.q_value(X[:, t], is_training=False)

        cur_alpha = __get_neutralized_portfolio(cur_actions, num_currencies)

        for c in range(num_currencies):
            cur_reward[c] = np.round(cur_alpha[c] * y[c][t], 8)
            avg_daily_return[t] \
                = np.round(avg_daily_return[t] + cur_reward[c], 8)
            position_change = np.round(position_change
                                       + abs(cur_alpha[c] - prev_alpha[c]), 8)
            prev_alpha[c] = cur_alpha[c]

        logger.add_daily_results(cumulative_asset, avg_daily_return[t])
        cumulative_asset = np.round(
            cumulative_asset
            + (cumulative_asset * avg_daily_return[t] * 0.01), 8)

    return position_change, cumulative_asset


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
