#Black-Scholes analytique (dividende continu q) et volatilité implicite
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def d1_d2(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_call_price(S, K, T, r, q, sigma):
    if T <= 0:
        return np.maximum(S - K, 0.0)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, q, sigma):
    return bs_call_price(S, K, T, r, q, sigma) - S * np.exp(-q * T) + K * np.exp(-r * T)


def bs_vega(S, K, T, r, q, sigma):
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)


def price_gap(sigma, market_price, S, K, T, r, q):
    return bs_call_price(S, K, T, r, q, sigma) - market_price


def bs_implied_vol(market_price, S, K, T, r, q, option_type="call", tol=1e-6, max_iter=50):
    #Newton depuis sigma = 0.2, puis Brent si Newton échoue. NaN hors des bornes d'arbitrage.
    if option_type == "put":
        market_price = market_price + S * np.exp(-q * T) - K * np.exp(-r * T)

    borne_basse = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    borne_haute = S * np.exp(-q * T)
    if market_price <= borne_basse or market_price >= borne_haute:
        return np.nan

    sigma = 0.2
    for i in range(max_iter):
        step = price_gap(sigma, market_price, S, K, T, r, q) / bs_vega(S, K, T, r, q, sigma)
        sigma = sigma - step
        if abs(step) < tol:
            return sigma
        if sigma <= 0 or sigma >= 5:
            break
    return brentq(price_gap, 1e-4, 5.0, args=(market_price, S, K, T, r, q), xtol=tol)


def implied_vol_surface(prices, S, strikes, maturities, r, q, option_type="call"):
    surface = np.zeros((len(maturities), len(strikes)))
    for i in range(len(maturities)):
        for j in range(len(strikes)):
            surface[i, j] = bs_implied_vol(prices[i][j], S, strikes[j], maturities[i], r, q, option_type)
    return surface
