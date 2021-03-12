import pandas as pd
import numpy as np


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
    use_vol: bool=False,
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
        use_vol (bool): Use realized volatility to set limits.
        devs (float): Standard deviations to set the upper and lower return
                      limits to when use_vol is True.
        join (bool): Return a join of the input dataframe and the labels.
        span (int): Span for exponential weighting.

    Returns:
        pd.DataFrame: Dataframe containing labels and optinanlly (join=True)
                      input values.
    '''
    # Incorrect time delta:
    if t < 1:
        raise ValueError("Look ahead time invalid, t<1.")

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
