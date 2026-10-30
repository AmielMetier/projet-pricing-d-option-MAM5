#Extraction de la volatilité locale par la formule de Dupire
import numpy as np


def price_derivatives(K, T, C):
    #dC/dT, dC/dK et d2C/dK2 par différences finies à 3 points d'ordre 2 
    dC_dT = np.full(C.shape, np.nan)
    dC_dK = np.full(C.shape, np.nan)
    d2C_dK2 = np.full(C.shape, np.nan)

    # En K, différences centrées sur les strikes intérieurs
    h_g = K[1:-1] - K[:-2]
    h_d = K[2:] - K[1:-1]
    dC_dK[:, 1:-1] = (h_g**2 * C[:, 2:] - h_d**2 * C[:, :-2] + (h_d**2 - h_g**2) * C[:, 1:-1]) / (h_g * h_d * (h_g + h_d))
    d2C_dK2[:, 1:-1] = 2 * (h_g * C[:, 2:] - (h_g + h_d) * C[:, 1:-1] + h_d * C[:, :-2]) / (h_g * h_d * (h_g + h_d))

    # En T, différences centrées sur les maturités intérieures
    for i in range(1, len(T) - 1):
        t_g = T[i] - T[i - 1]
        t_d = T[i + 1] - T[i]
        dC_dT[i] = (t_g**2 * C[i + 1] - t_d**2 * C[i - 1] + (t_d**2 - t_g**2) * C[i]) / (t_g * t_d * (t_g + t_d))

    # Dernière maturité, différence décentrée arrière
    t_1 = T[-2] - T[-3]
    t_2 = T[-1] - T[-2]
    dC_dT[-1] = t_2 * C[-3] / (t_1 * (t_1 + t_2)) - (t_1 + t_2) * C[-2] / (t_1 * t_2) + (t_1 + 2 * t_2) * C[-1] / (t_2 * (t_1 + t_2))
    return dC_dT, dC_dK, d2C_dK2


def local_vol_from_prices(K, T, C, r, min_gamma=1e-10, q=0.0):
    #Formule de Dupire. NaN là où d2C/dK2 < min_gamma ou là où la variance est négative
    dC_dT, dC_dK, d2C_dK2 = price_derivatives(K, T, C)
    num = dC_dT + (r - q) * K * dC_dK + q * C
    den = 0.5 * K**2 * d2C_dK2

    sigma_loc = np.full(C.shape, np.nan)
    ok = (d2C_dK2 >= min_gamma) & (den > 0) & (num > 0)
    sigma_loc[ok] = np.sqrt(num[ok] / den[ok])
    sigma_loc[T == 0] = np.nan   # T = 0 : payoff
    return sigma_loc


def fill_local_vol_surface(K, T, sigma_loc):
    #Remplace les NaN par interpolation linéaire en K puis en T (valeur constante aux bords)
    surface = sigma_loc.copy()
    for i in range(len(T)):
        ok = np.isfinite(surface[i])
        if np.any(ok):
            surface[i] = np.interp(K, K[ok], surface[i, ok])
    for j in range(len(K)):
        ok = np.isfinite(surface[:, j])
        surface[:, j] = np.interp(T, T[ok], surface[ok, j])
    return surface


def interpolate_local_vol(k, t, K, T, surface):
    #Vol locale en (k, t) par interpolation linéaire, constante hors de la grille. surface sans NaN
    sigma_t = np.zeros(len(K))
    for j in range(len(K)):
        sigma_t[j] = np.interp(t, T, surface[:, j])
    return np.interp(k, K, sigma_t)
