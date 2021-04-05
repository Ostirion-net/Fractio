# (c) 2021 Ostirion.net
# This code is licensed under MIT license (see LICENSE for details)


import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.stats import entropy


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
        df (pd.DataFrame): Dataframe with series to be differentiated in a
                           single column.
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
        df (pd.DataFrame): Dataframe with series to be differentiated in a
                           single column.
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


def compute_vol(df: pd.DataFrame,
                span: int=100) -> pd.DataFrame:
    '''
    Compute period volatility of returns as exponentially weighted
    moving standard deviation:
    Args:
        df (pd.DataFrame): Dataframe with price series in a single column.
        span (int): Span for exponential weighting.
    Returns:
        pd.DataFrame: Dataframe containing volatility estimates.
    '''
    df.fillna(method='ffill', inplace=True)
    r = df.pct_change()
    return r.ewm(span=span).std()


def triple_barrier_labels(
    df: pd.DataFrame,
    t: int,
    upper: float=None,
    lower: float=None,
    devs: float=2.5,
    join: bool=False,
    span: int=100) -> pd.DataFrame:
    '''
    Compute the triple barrier label for a price time series:
    Args:
        df (pd.DataFrame): Dataframe with price series in a single column.
        t (int): Future periods to obtain the lable for.
        upper (float): Returns for upper limit.
        lower (float): Returns for lower limit.
        devs (float): Standard deviations to set the upper and lower return
                      limits to when no limits passed.
        join (bool): Return a join of the input dataframe and the labels.
        span (int): Span for exponential weighting.
    Returns:
        pd.DataFrame: Dataframe containing labels and optinanlly (join=True)
                      input values.
    '''
    # Incorrect time delta:
    if t < 1:
        raise ValueError("Look ahead time invalid, t<1.")
    # Lower limit must be negative:
    if lower is not None:
        if lower > 0:
            raise ValueError("Lower limit must be a negative value.")

    df.fillna(method='ffill', inplace=True)

    lims = np.array([upper, lower])

    labels = pd.DataFrame(index=df.index, columns=['Label'])

    returns = df.pct_change()

    r = range(0, len(df)-1-t)
    for idx in r:
        s = returns.iloc[idx:idx+t]
        minimum = s.cumsum().values.min()
        maximum = s.cumsum().values.max()

        if not all(np.isfinite(s.cumsum().values)):
            labels['Label'].iloc[idx] = np.nan
            continue

        if any(lims == None):
            vol = compute_vol(df[:idx+t], span)

        if upper is None:
            u = vol.iloc[idx].values*devs
        else:
            u = upper

        if lower is None:
            l = -vol.iloc[idx].values*devs
        else:
            l = lower

        valid = np.isfinite(u) and np.isfinite(l)
        if not valid:
            labels['Label'].iloc[idx] = np.nan
            continue

        if any(s.cumsum().values >= u):
            labels['Label'].iloc[idx] = 1
        elif any(s.cumsum().values <= l):
            labels['Label'].iloc[idx] = -1
        else:
            labels['Label'].iloc[idx] = 0

    if join:
        df = df.join(labels)
        return df

    return labels


def get_entropic_labels(df: pd.DataFrame,
               side: str = 'max',
               future_space: np.linspace = np.linspace(2, 90, 40, dtype=int),
               tbl_settings: dict = {}) -> pd.DataFrame:
    '''
    Compute the series of triple barrier labels for a price series that
    results in the maximum or minimum entropy for label distribution.
    Args:
        df (pd.Dataframe): Dataframe with price series in a single column.
        side (str): 'max' or 'min' to select maximum or minimim entropies.
                    'min' entropy may not result in usable data.
        future_space (np.linspace): Space of future windows to analyze.
        tbl_settings (dict): Dictionary with settings for triple_barrier_labels
                             function.
    Returns:
        pd.DataFrame: Dataframe with the selected entropy distribution of
                      labels.
    '''

    if side not in ['max', 'min']:
        raise ValueError("Side must be 'max' or 'min'.")

    # Labels:
    l = {}
    for f in future_space:
        # Check this for references:
        l[f] = triple_barrier_labels(df, f, **tbl_settings)

    # Counts:
    c = {}
    for f in l.keys():
        s = l[f].squeeze()
        c[f] = s.value_counts(normalize=True)

    # Entropies:
    e = {}
    for f, c in c.items():
        e[f] = entropy(c)

    # Maximum and minimum entropies:
    max_e = [k for k, v in e.items() if v == max(e.values())][0]
    min_e = [k for k, v in e.items() if v == min(e.values())][0]

    if side == 'max':
        e_labels = l[max_e]
        t = max_e

    if side == 'min':
        e_labels = l[min_e]
        t = min_e

    e_labels.columns = ['t_delta='+str(t)]
    return e_labels


def cusum_events(df: pd.DataFrame,
                 h: float=None,
                 span: int=100,
                 devs: float=2.5) -> pd.DataFrame:
    '''
    Compute CUSUM events for a given price series.
    Args:
        df (pd.DataFrame): Dataframe with price time series
                           in a single column.
        h (float): Arbitrary cumulative returns value limit to trigger
                   the CUSUM filter. The filter is symmetric. If h
                   is None exponentially weighted standard deviation will
                   be used.
        span (int): Span for exponential weighting of standard deviation.
        devs (float): Standard deviations to compute variable
                      trigger limits if h is not defined.
    Returns:
        pd.DataFrame: Dataframe containing differentiated series.
    '''
    # Events e:
    e = pd.DataFrame(0, index=df.index,
                     columns=['CUSUM_Event'])
    s_pos = 0
    s_neg = 0
    r = df.pct_change()

    for idx in r.index:
        if h is None:
            h_ = r[:idx].ewm(span=span).std().values[-1][0]*devs
        else:
            h_ = h
        s_pos = max(0, s_pos+r.loc[idx].values)
        s_neg = min(0, s_neg+r.loc[idx].values)
        if s_neg < -h_:
            s_neg = 0
            e.loc[idx] = -1
        elif s_pos > h_:
            s_pos = 0
            e.loc[idx] = 1
    return e
