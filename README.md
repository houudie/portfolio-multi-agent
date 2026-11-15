# Assistant multi-agent pour la sélection intelligente de portefeuilles financiers


Six agents spécialisés (Profil, Données, Prévision, Risque, Allocation, Conseiller), pilotés par un orchestrateur, transforment le profil d'un investisseur en portefeuille expliqué. Un moteur de backtest compare chaque allocation à des stratégies de référence (1/N, Markowitz, indice).

## Structure du dépôt

```
portfolio-multi-agent/
├── app/                 # Application Streamlit (étape 7)
├── config.yaml          # Univers d'actifs, dates de découpage, paramètres
├── data/
│   ├── raw/             # Données brutes téléchargées (non versionnées)
│   └── processed/       # Données nettoyées et features (non versionnées)
├── notebooks/           # EDA et expériences
├── reports/figures/     # Figures pour le rapport et les slides
├── src/
│   ├── agents/          # Les 6 agents + orchestrateur (étape 5)
│   ├── backtest/        # Moteur de backtest et métriques (étape 3)
│   ├── data/            # Téléchargement et préparation des données (étape 2)
│   └── models/          # Modèles de prévision et agent RL (étape 4)
└── tests/               # Tests automatiques
```

## Installation

```bash
git clone <url-du-depot>
cd portfolio-multi-agent
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Exécution

```bash
# 1. Télécharger les données (Yahoo Finance + FRED)
python -m src.data.download

# 2. Nettoyer, découper (train / validation / test) et créer les features
python -m src.data.prepare

# 3. Analyse exploratoire : ouvrir et exécuter notebooks/01_eda.ipynb

# 4. Lancer les tests
python -m pytest

# 5. Lancer l'application (à partir de l'étape 7)
streamlit run app/app.py
```

## Reproduction des résultats

Toutes les étapes lisent `config.yaml` : univers d'actifs, dates de découpage et graine aléatoire. Pour reproduire les résultats, il suffit de relancer les scripts dans l'ordre ci-dessus.

## Données

| Source | Contenu | Accès |
| --- | --- | --- |
| Yahoo Finance (`yfinance`) | Prix journaliers ajustés des actions et ETF | Gratuit, sans clé |
| FRED (Federal Reserve Bank of St. Louis) | Taux à 10 ans (DGS10), VIX (VIXCLS), taux directeur effectif quotidien (DFF) | Gratuit, sans clé (CSV) |

### Préparation (`src/data/prepare.py`)

1. **Nettoyage :** la période commence quand tous les actifs sont cotés (META depuis mai 2012) ; les trous de cotation de 5 jours maximum sont comblés avec le dernier prix connu.
2. **Découpage chronologique :** train jusqu'au 31/12/2019, validation 2020–2021, test 2022–2025.
3. **Features causales** par actif : momentum 21/63/126 jours, volatilité 21/63 jours, RSI 14, drawdown sur 252 jours ; features de marché (SPY) ; macro décalée d'un jour.
4. **Cible :** rendement log des 21 prochains jours de bourse.
5. **Purge :** les 21 derniers jours du train et de la validation sont retirés pour que les cibles ne débordent pas sur la période suivante.

## Règles anti-fuite de données

1. Le découpage **chronologique** train / validation / test est fait avant tout prétraitement.
2. Les scalers et modèles sont ajustés sur le train uniquement.
3. Chaque feature à la date *t* n'utilise que des informations disponibles jusqu'à *t − 1*.
4. La période de test n'est évaluée qu'une seule fois, à l'étape 8.

## Équipe

| Membre | Rôle |
| --- | --- |
| Abderrahmane  | Données, EDA, backtest, évaluation finale |
| Rachid | Modèles de prévision, interprétabilité (SHAP) |
| Hatim | Agent RL (PPO), agent Allocation |
| Houda | Orchestrateur, agent Conseiller, Streamlit, rapport |
