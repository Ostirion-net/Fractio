# (c) 2021 Ostirion.net
# This code is licensed under MIT license (see LICENSE for details)


import numpy as np
import pandas as pd


def compute_weights(d: float,
                    size: int) -> pd.DataFrame:
    '''
    Compute the weights of individual data points
    for fractional differentiation:

    Args:
        d (float): Fractional differentiation value.
        size (int): Length of the data series.

    Returns:
        pd.DataFrame: Dataframe containing the weights for each point.
    '''

    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1]/k*(d-k+1))
    w = np.array(w[::-1]).reshape(-1, 1)

    return pd.DataFrame(w)


def standard_frac_diff(df: pd.DataFrame,
                       d: float,
                       thres: float=.01) -> pd.DataFrame:
    '''
    Compute the d fractional difference of the series.

    Args:
        df (pd.DataFrame): Dataframe with series to be differentiated.
        d (float): Order of differentiation.
        thres (float): threshold value to drop non-significant weights.

    Returns:
        pd.DataFrame: Dataframe containing differntiated series.
    '''
    w = compute_weights(d, len(df))
    w_ = np.cumsum(abs(w))
    w_ /= w_.iloc[-1]
    skip = int((w_ > thres).sum().values)
    results = {}
    index = df.index

    for name in df.columns:
        series_f = df[name].fillna(method='ffill').dropna()
        r = range(skip, series_f.shape[0])
        df_ = pd.Series(index=r)
        for idx in r:
            if not np.isfinite(df[name].iloc[idx]):
                continue
            results[idx] = np.dot(w.iloc[-(idx):, :].T, series_f.iloc[:idx])[0]

    result = pd.DataFrame(pd.Series(results), columns=['Frac_diff'])
    result.set_index(df[skip:].index, inplace=True)

    return result
