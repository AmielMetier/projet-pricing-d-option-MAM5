# Modèle de volatilité locale de Dupire

Mini-projet de calcul numérique pour la finance — **Polytech Nice Sophia (MAM, parcours IMAFA)**.

## Équipe
- Amiel Metier
- Nicolas Muratore
- Tahitoa Senis

---

## Présentation du projet

Dans le modèle de Black-Scholes classique, la volatilité du sous-jacent est supposée constante. Sur les marchés réels, les prix des options révèlent pourtant un *smile* ou un *skew* de volatilité implicite selon le strike et la maturité.

Le modèle de **volatilité locale de Bruno Dupire (1994)** résout cette anomalie en introduisant une fonction de volatilité déterministe $\sigma_{loc}(S_t, t)$ :
$$dS_t = (r - q) S_t \, dt + \sigma_{loc}(S_t, t) S_t \, dW_t$$

Ce modèle est calibré pour reproduire exactement l'ensemble des prix de calls et de puts vanilles cotés sur le marché.

Le projet met en œuvre l'ensemble de la chaîne de calcul :
1. **Formules fermées et inversion de Black-Scholes** (calcul de la volatilité implicite par Newton-Raphson et Brent).
2. **Résolution de l'EDP forward de Dupire** en $(K, T)$ par différences finies ($\\theta$-schéma de Crank-Nicolson avec amortissement de Rannacher).
3. **Extraction de la volatilité locale** à partir des prix d'options par différences finies (formule de Dupire).
4. **Validation numérique** : ordres de consistance en temps et en espace, robustesse au bruit de marché.
5. **Application sur données réelles du S&P 500** et **pricing d'options barrières exotiques** par simulation Monte Carlo.

---

## Organisation du code

```
.
├── src/dupire/
│   ├── black_scholes.py   # Formules fermées, vega, vol implicite (Newton / Brent)
│   ├── pde_solver.py      # EDP forward de Dupire (theta-schéma, Crank-Nicolson, Rannacher)
│   ├── local_vol.py       # Extraction de sigma_loc (formule de Dupire) et interpolation
│   └── monte_carlo.py     # Simulateur d'Euler sous vol locale et payoffs d'options
├── notebooks/
│   ├── 01_validation_synthetique.ipynb   # Étude des schémas, convergence et bruit
│   └── 02_marche_spx.ipynb               # Données réelles S&P 500 et option barrière
├── data/
│   └── spx_options.csv                   # Données de marché S&P 500 téléchargées
├── report/figures/                       # Graphiques générés pour le rapport
├── requirements.txt                      # Dépendances Python
└── README.md
```

---

## Installation et exécution

### Prérequis
Python 3.10 ou supérieur.

```bash
# Création et activation de l'environnement virtuel
python -m venv .venv
source .venv/bin/activate       # Sous Windows : .venv\Scripts\activate

# Installation des dépendances
pip install -r requirements.txt
```

### Lancement des notebooks
```bash
jupyter lab
```

- **`notebooks/01_validation_synthetique.ipynb`** : lance la validation sur données synthétiques (comparaison des schémas numériques, étude des ordres de convergence $\mathcal{O}(\Delta t^2)$ et $\mathcal{O}(\Delta K^2)$, aller-retour avec la formule de Dupire, impact du bruit).
- **`notebooks/02_marche_spx.ipynb`** : télécharge ou charge les options du S&P 500 (`^SPX`), extrait le smile et la volatilité locale, et price un Call barrière Up-and-Out par Monte Carlo en comparant avec Black-Scholes.

---

## Synthèse des résultats

1. **Stabilité et consistance des schémas d'EDP** :
   - Le schéma **explicite** diverge dès que la condition CFL n'est pas respectée.
   - Les schémas **implicite** et **Crank-Nicolson** sont inconditionnellement stables.
   - L'ajout de 2 demi-pas de **Rannacher** au démarrage amortit la discontinuité du payoff sans dégrader l'ordre 2 en temps $\mathcal{O}(\Delta t^2)$ (pente observée en log-log : $\approx 2{,}0$).
   - Le schéma spatial centré d'ordre 2 est validé avec une pente log-log de $\approx 1{,}8$.

2. **Sensibilité au bruit et lissage** :
   - Un faible bruit de marché ($0{,}5$ centime) fait exploser la dérivée seconde $\frac{\partial^2 C}{\partial K^2}$ par différences finies brutes, produisant des variances locales négatives.
   - Sur données réelles, un **lissage régulier $C^2$** (ajustement polynomial) est indispensable avant d'appliquer la formule de Dupire.

3. **Validation sur le marché S&P 500** :
   - Le S&P 500 présente un **skew prononcé** (la volatilité baisse quand le strike augmente).
   - À la monnaie, la règle théorique de Dupire (*twice the skew*) est bien vérifiée : le ratio entre la pente de la volatilité locale et celle de la volatilité implicite est de $\approx 1{,}6$ (proche de 2).

4. **Impact sur le pricing d'un Call Barrière Up-and-Out ($K = 7700$, $B = 8200$)** :
   - Sur le **Call Vanille** ($K = 7700$), Black-Scholes et Dupire donnent le même prix que le marché ($\\approx 369\$$).
   - Sur le **Call Barrière**, Dupire donne **$26{,}7\$$** contre **$16{,}3\$$** pour Black-Scholes (**$+63\%$**).
   - **Explication financière** : Le skew réel réduit la volatilité aux strikes élevés ($\approx 11\%$ vers la barrière contre $16\%$ à la monnaie). L'indice est donc plus calme à la hausse et risque beaucoup moins d'être désactivé que sous l'hypothèse de volatilité constante de Black-Scholes. Ignorer le smile sous-évalue très lourdement ce type de contrat exotique.

---

## Références
- B. Dupire, *Pricing with a Smile*, Risk, 1994.
- J. Gatheral, *The Volatility Surface: A Practitioner's Guide*, Wiley, 2006.
