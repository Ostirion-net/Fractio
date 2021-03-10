# (c) 2021 Ostirion.net
# This code is licensed under MIT license (see LICENSE for details)


import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


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
        df (pd.DataFrame): Dataframe with series to be differentiated in a single
                           column.
        d (float): Order of differentiation.
        thres (float): threshold value to drop non-significant weights.

    Returns:
        pd.DataFrame: Dataframe containing differentiated series.
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


def compute_weights_fixed_window(d: float,
                                 threshold: float=1e-5) -> pd.DataFrame:
    '''
    Compute the weights of individual data points
    for fractional differentiation with fixed window:
    Args:
        d (float): Fractional differentiation value.
        threshold (float): Minimum weight to calculate.
    Returns:
        pd.DataFrame: Dataframe containing the weights for each point.
    '''

    w = [1.0]
    k = 1
    while True:
        v = -w[-1]/k*(d-k+1)
        if abs(v) < threshold:
            break
        w.append(v)
        k += 1

    w = np.array(w[::-1]).reshape(-1, 1)
    return pd.DataFrame(w)


def fixed_window_fracc_diff(df: pd.DataFrame,
                            d: float,
                            threshold: float=1e-5) -> pd.DataFrame:
    '''
    Compute the d fractional difference of the series with
    a fixed width window. It defaults to standard fractional
    differentiation when the length of the weights becomes 0.
    
    Args:
        df (pd.DataFrame): Dataframe with series to be differentiated in a single
                           column.
        d (float): Order of differentiation.
        threshold (float): threshold value to drop non-significant weights.
    Returns:
        pd.DataFrame: Dataframe containing differentiated series.
    '''

    w = compute_weights_fixed_window(d, threshold)
    l = len(w)
    results = {}
    names = df.columns
    for name in names:
        series_f = df[name].fillna(method='ffill').dropna()

        if l > series_f.shape[0]:
            return standard_frac_diff(df, d, threshold)
        r = range(l, series_f.shape[0])
        df_ = pd.Series(index=r)

        for idx in r:
            if not np.isfinite(df[name].iloc[idx]):
                continue
            results[idx] = np.dot(w.iloc[-(idx):, :].T,
                                  series_f.iloc[idx-l:idx])[0]

    result = pd.DataFrame(pd.Series(results), columns=['Frac_diff'])
    result.set_index(df[l:].index, inplace=True)

    return result


def find_stat_series(df: pd.DataFrame,
                     threshold: float=0.0001,
                     diffs: np.linspace=np.linspace(0.05, 0.95, 19),
                     p_value: float=0.05) -> pd.DataFrame:
    '''
    Find the series that passes the adf test at the given
    p_value.
    The time series must be a single column dataframe.
    Args:
        df (pd.DataFrame): Dataframe with series to be differentiated.
        threshold (float): threshold value to drop non-significant weights.
        diffs (np.linspace): Space for candidate d values.
        p_value (float): ADF test p-value limit for rejection of null
                         hypothesis.
    Returns:
        pd.DataFrame: Dataframe containing differentiated series. This series
                      is stationary and maintains maximum memory information.
    '''

    for diff in diffs:
        if diff == 0:
            continue
        s = fixed_window_fracc_diff(df, diff, threshold)
        adf_stat = adfuller(s, maxlag=1, regression='c', autolag=None)[1]
        if adf_stat < p_value:
            s.columns = ['d='+str(diff)]
            return s
