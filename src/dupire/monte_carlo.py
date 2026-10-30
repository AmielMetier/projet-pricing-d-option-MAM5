#Monte Carlo sous volatilité locale : Euler sur ln S, dS = (r - q) S dt + sigma_loc(S, t) S dW
import numpy as np


def simulate_paths(S0, r, sigma_loc, T, n_steps, n_paths, q=0.0):
    #Trajectoires de forme (n_paths, n_steps + 1). sigma_loc est une fonction (S, t)
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for k in range(n_steps):
        S = paths[:, k]
        sigma = sigma_loc(S, k * dt)
        Z = np.random.standard_normal(n_paths)
        paths[:, k + 1] = S * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return paths


def mc_price(payoffs, r, T):
    #Prix actualisé et intervalle de confiance à 95 % : (prix, borne basse, borne haute)
    x = np.exp(-r * T) * payoffs
    prix = x.mean()
    h = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return prix, prix - h, prix + h


def european_call_payoff(paths, K):
    return np.maximum(paths[:, -1] - K, 0.0)


def european_put_payoff(paths, K):
    return np.maximum(K - paths[:, -1], 0.0)


def barrier_up_and_out_call_payoff(paths, K, B):
    return np.where(paths.max(axis=1) < B, european_call_payoff(paths, K), 0.0)


def barrier_down_and_out_call_payoff(paths, K, B):
    return np.where(paths.min(axis=1) > B, european_call_payoff(paths, K), 0.0)
