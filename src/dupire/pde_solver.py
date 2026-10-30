#EDP forward de Dupire (q = 0) : dC/dT = 1/2 sigma_loc^2 K^2 d2C/dK2 - r K dC/dK
import numpy as np
from scipy.linalg import solve_banded


def dupire_operator(K, t, sigma_loc, r):
    #Coefficients 
    dK = K[1] - K[0]
    Ki = K[1:-1]
    a = 0.5 * sigma_loc(Ki, t) ** 2 * Ki**2 / dK**2
    b = r * Ki / (2 * dK)
    return a + b, -2 * a, a - b


def theta_step(K, U, t, dt, theta, sigma_loc, r, S0):
    #Un pas de temps 
    sub, diag, sup = dupire_operator(K, t + theta * dt, sigma_loc, r)

    # Membre de droite
    LU = sub * U[:-2] + diag * U[1:-1] + sup * U[2:]
    rhs = U[1:-1] + (1 - theta) * dt * LU
    rhs[0] += theta * dt * sub[0] * S0   # bord K = 0 : C = S0

    # Matrice tridiagonale
    M = np.zeros((3, len(diag)))
    M[0, 1:] = -theta * dt * sup[:-1]
    M[1] = 1 - theta * dt * diag
    M[2, :-1] = -theta * dt * sub[1:]

    U_new = U.copy()
    U_new[1:-1] = solve_banded((1, 1), M, rhs)
    return U_new


def solve_dupire_pde(S0, r, sigma_loc, K_max, T_max, n_K, n_T, theta=0.5, n_rannacher=2):
    #Renvoie (K, T, C) avec C[i, j] = prix du call de strike K[j] et de maturité T[i]
    K = np.linspace(0.0, K_max, n_K + 1)
    T = np.linspace(0.0, T_max, n_T + 1)
    dt = T_max / n_T
    C = np.zeros((n_T + 1, n_K + 1))
    C[0] = np.maximum(S0 - K, 0.0)

    for n in range(n_T):
        if n < n_rannacher:
            demi = theta_step(K, C[n], T[n], dt / 2, 1.0, sigma_loc, r, S0)
            C[n + 1] = theta_step(K, demi, T[n] + dt / 2, dt / 2, 1.0, sigma_loc, r, S0)
        else:
            C[n + 1] = theta_step(K, C[n], T[n], dt, theta, sigma_loc, r, S0)
    return K, T, C
