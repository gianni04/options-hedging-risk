"""Realised volatility estimators, walk-forward forecasts and the variance risk premium."""

import numpy as np
import pandas as pd
from arch import arch_model

TRADING_DAYS = 252
_TINY = 1e-300
_ANN = np.sqrt(TRADING_DAYS)


def _log_returns(close):
    return np.log(close.astype(float)).diff()


def _annualise(variance):
    return np.sqrt(variance.clip(lower=0.0) * TRADING_DAYS)


def close_to_close(close, window=21):
    """Annualised close-to-close realised volatility."""
    return _log_returns(close).rolling(window).std(ddof=1) * _ANN


def parkinson(high, low, window=21):
    """Annualised Parkinson high-low range volatility."""
    hl = np.log(high.astype(float) / low.astype(float)) ** 2
    return _annualise(hl.rolling(window).mean() / (4.0 * np.log(2.0)))


def garman_klass(ohlc, window=21):
    """Annualised Garman-Klass volatility from open, high, low and close."""
    hl = np.log(ohlc["high"] / ohlc["low"]) ** 2
    co = np.log(ohlc["close"] / ohlc["open"]) ** 2
    terms = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    return _annualise(terms.rolling(window).mean())


def _rogers_satchell_terms(ohlc):
    ho = np.log(ohlc["high"].astype(float) / ohlc["open"])
    lo = np.log(ohlc["low"].astype(float) / ohlc["open"])
    co = np.log(ohlc["close"].astype(float) / ohlc["open"])
    return ho * (ho - co) + lo * (lo - co)


def rogers_satchell(ohlc, window=21):
    """Annualised Rogers-Satchell volatility, robust to a drift in the price."""
    return _annualise(_rogers_satchell_terms(ohlc).rolling(window).mean())


def yang_zhang(ohlc, window=21):
    """Annualised Yang-Zhang volatility: overnight, open-close and Rogers-Satchell combined."""
    overnight = np.log(ohlc["open"].astype(float) / ohlc["close"].shift(1))
    open_close = np.log(ohlc["close"].astype(float) / ohlc["open"])
    v_o = overnight.rolling(window).var(ddof=1)
    v_c = open_close.rolling(window).var(ddof=1)
    v_rs = _rogers_satchell_terms(ohlc).rolling(window).mean()
    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    return _annualise(v_o + k * v_c + (1.0 - k) * v_rs)


def ewma(close, lam=0.94):
    """Annualised RiskMetrics EWMA volatility with decay `lam`."""
    r2 = _log_returns(close) ** 2
    return _annualise(r2.ewm(alpha=1.0 - lam, adjust=False).mean())


def _garch_mean_variance(omega, alpha, beta, h1, horizon):
    persistence = alpha + beta
    if abs(1.0 - persistence) < 1e-9:
        return h1 + 0.5 * omega * (horizon - 1.0)
    uncond = omega / (1.0 - persistence)
    decay = (1.0 - persistence ** horizon) / (horizon * (1.0 - persistence))
    return uncond + (h1 - uncond) * decay


def garch_forecast(close, horizon=21, refit_every=21, min_obs=500):
    """Walk-forward GARCH(1,1) forecast with Student-t errors of the average
    annualised volatility over the next `horizon` days."""
    returns = _log_returns(close).dropna()
    out = pd.Series(np.nan, index=close.index, dtype=float)
    values = returns.to_numpy() * 100.0
    n = values.size
    if n < min_obs:
        return out
    forecast = np.full(n, np.nan)
    start = min_obs - 1
    mu = omega = alpha = beta = 0.0
    var_t = eps_t = 0.0
    for i in range(start, n):
        if (i - start) % refit_every == 0:
            model = arch_model(
                values[: i + 1], p=1, q=1, mean="Constant", dist="t", rescale=False
            )
            fit = model.fit(disp="off", show_warning=False)
            mu = float(fit.params["mu"])
            omega = float(fit.params["omega"])
            alpha = float(fit.params["alpha[1]"])
            beta = float(fit.params["beta[1]"])
            var_t = float(fit.conditional_volatility[-1]) ** 2
        else:
            var_t = omega + alpha * eps_t * eps_t + beta * var_t
        eps_t = values[i] - mu
        h1 = omega + alpha * eps_t * eps_t + beta * var_t
        mean_var = _garch_mean_variance(omega, alpha, beta, h1, horizon)
        forecast[i] = np.sqrt(TRADING_DAYS * max(mean_var, 0.0)) / 100.0
    out.loc[returns.index] = forecast
    return out


def _trailing_mean(values, window):
    return pd.Series(values).rolling(window).mean().to_numpy()


def har_forecast(close, horizon=21, min_obs=500):
    """Walk-forward HAR-RV forecast of the average annualised volatility over the
    next `horizon` days, re-estimated by expanding-window OLS on log variances."""
    returns = _log_returns(close).dropna()
    out = pd.Series(np.nan, index=close.index, dtype=float)
    rv = returns.to_numpy() ** 2
    n = rv.size
    if n < min_obs:
        return out
    monthly = np.maximum(_trailing_mean(rv, 22), _TINY)
    floor = 0.01 * monthly
    daily = np.maximum(rv, floor)
    weekly = np.maximum(_trailing_mean(rv, 5), floor)
    design = np.column_stack(
        [np.ones(n), np.log(daily), np.log(weekly), np.log(monthly)]
    )
    target = np.log(
        np.maximum(pd.Series(rv).rolling(horizon).mean().shift(-horizon).to_numpy(), _TINY)
    )
    xtx = np.zeros((4, 4))
    xty = np.zeros(4)
    ridge = 1e-10 * np.eye(4)
    forecast = np.full(n, np.nan)
    start = min_obs - 1
    added = -1
    count = 0
    for i in range(start, n):
        while added < i - horizon:
            added += 1
            row, obs = design[added], target[added]
            if np.isfinite(row).all() and np.isfinite(obs):
                xtx += np.outer(row, row)
                xty += row * obs
                count += 1
        if count < 40 or not np.isfinite(design[i]).all():
            continue
        coef = np.linalg.solve(xtx + ridge, xty)
        pred = float(design[i] @ coef)
        forecast[i] = np.sqrt(TRADING_DAYS * np.exp(np.clip(pred, -60.0, 5.0)))
    out.loc[returns.index] = forecast
    return out


def variance_risk_premium(iv30, rv_forecast):
    """Implied minus forecast realised volatility, annualised decimals.

    Percentage-point input is detected from a median above 1.5 and rescaled."""
    iv = iv30.astype(float)
    median = iv.median()
    if np.isfinite(median) and median > 1.5:
        iv = iv / 100.0
    return iv - rv_forecast


def vrp_zscore(vrp, window=252):
    """Rolling z-score of the variance risk premium using only data up to each date."""
    mean = vrp.rolling(window).mean()
    sd = vrp.rolling(window).std(ddof=1)
    return (vrp - mean) / sd.where(sd > 0.0)
