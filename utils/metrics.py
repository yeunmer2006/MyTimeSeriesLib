import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    # 防止除以0，加上1e-8稳定计算
    return np.mean(np.abs((true - pred) / (true + 1e-8)))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / (true + 1e-8)))


def ACC(pred, true):
    """
    计算涨跌方向预测准确率
    输入：
        pred: 预测序列，shape (batch, seq_len)
        true: 真实序列，shape (batch, seq_len)
    输出：
        预测涨跌方向与真实涨跌方向一致的比例
    """
    pred_direction = np.sign(pred[:, 1:] - pred[:, :-1])
    true_direction = np.sign(true[:, 1:] - true[:, :-1])
    acc = np.mean(pred_direction == true_direction)
    return acc


def SR(pred):
    """
    计算夏普比率
    基于预测收益率的平均收益 / 收益标准差
    输入：
        pred: 预测价格序列，shape (batch, seq_len)
    输出：
        夏普比率标量
    """
    returns = (pred[:, 1:] - pred[:, :-1]) / (pred[:, :-1] + 1e-8)
    mean_return = np.mean(returns)
    std_return = np.std(returns) + 1e-8
    return mean_return / std_return


def strategy_cr(pred, true):
    """
    计算策略累计收益率
    策略：预测明天上涨则当天买入，预测下跌则当天空仓
    输入：
        pred: 预测价格序列，shape (batch, seq_len)
        true: 真实价格序列，shape (batch, seq_len)
    输出：
        平均累计收益率
    """
    pred_diff = pred[:, 1:] - pred[:, :-1]
    pred_signal = (pred_diff > 0).astype(float)  # 买入信号

    true_diff = true[:, 1:] - true[:, :-1]
    true_return = true_diff / (true[:, :-1] + 1e-8)

    strategy_return = pred_signal * true_return  # 只在预测上涨时参与收益
    cumulative_return = np.prod(1 + strategy_return, axis=1) - 1  # 每条序列的累计收益率

    return np.mean(cumulative_return)


def metric(pred, true):
    """
    计算时间序列预测的多个评估指标（均基于原始尺度）
    输入：
        pred: 预测值，shape (batch, seq_len, 1) 或 (batch, seq_len)
        true: 真实值，shape 同 pred
    返回：
        mae, mse, rmse, mape, mspe, acc, cr, sr
    """
    pred = pred.reshape(pred.shape[0], -1)
    true = true.reshape(true.shape[0], -1)

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    acc = ACC(pred, true)
    cr = strategy_cr(pred, true)
    sr = SR(pred)

    return mae, mse, rmse, mape, mspe, acc, cr, sr
