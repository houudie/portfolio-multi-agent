# Assistant multi-agent pour la sélection intelligente de portefeuilles financiers

Projet Centrale Casablanca — rendu le 30 novembre 2026.

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

# 2. Lancer les tests
pytest

# 3. Lancer l'application (à partir de l'étape 7)
streamlit run app/app.py
```

## Reproduction des résultats

Toutes les étapes lisent `config.yaml` : univers d'actifs, dates de découpage et graine aléatoire. Pour reproduire les résultats, il suffit de relancer les scripts dans l'ordre ci-dessus.

## Données

| Source | Contenu | Accès |
| --- | --- | --- |
| Yahoo Finance (`yfinance`) | Prix journaliers ajustés des actions et ETF | Gratuit, sans clé |
| FRED (Federal Reserve Bank of St. Louis) | Taux à 10 ans, VIX, taux directeur | Gratuit, sans clé (CSV) |

## Règles anti-fuite de données

1. Le découpage **chronologique** train / validation / test est fait avant tout prétraitement.
2. Les scalers et modèles sont ajustés sur le train uniquement.
3. Chaque feature à la date *t* n'utilise que des informations disponibles jusqu'à *t − 1*.
4. La période de test n'est évaluée qu'une seule fois, à l'étape 8.

## Équipe

| Membre | Rôle |
| --- | --- |
| Membre A | Données, EDA, backtest, évaluation finale |
| Membre B | Modèles de prévision, interprétabilité (SHAP) |
| Membre C | Agent RL (PPO), agent Allocation |
| Membre D | Orchestrateur, agent Conseiller, Streamlit, rapport |
