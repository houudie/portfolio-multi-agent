# Assistant multi-agent pour la sélection intelligente de portefeuilles financiers
 
Projet Centrale Casablanca — rendu le 30 novembre 2026.
 
Six agents spécialisés (Profil, Données, Prévision, Risque, Allocation, Conseiller), pilotés par un orchestrateur, transforment le profil d'un investisseur en portefeuille expliqué. Un moteur de backtest compare chaque allocation à des stratégies de référence (1/N, Markowitz, indice).
 
## Structure du dépôt
 
```
portfolio-multi-agent/
├── app/                 # Application Streamlit
├── config.yaml          # Univers d'actifs, dates de découpage, paramètres
├── data/
│   ├── raw/             # Données brutes téléchargées (non versionnées)
│   └── processed/       # Données nettoyées, features, prévisions (non versionnées)
├── models_saved/        # Modèles entraînés (non versionnés, recréés par les scripts)
├── notebooks/           # EDA et expériences
├── reports/             # Tableaux de résultats (.csv)
│   └── figures/         # Figures pour le rapport et les slides
├── src/
│   ├── agents/          # Les 6 agents + orchestrateur
│   ├── backtest/        # Moteur de backtest, baselines, comparaison des règles
│   ├── data/            # Téléchargement et préparation des données
│   ├── evaluation/      # Évaluation finale sur le test
│   └── models/          # Modèles de prévision, interprétabilité, agent RL
└── tests/               # Tests automatiques (pytest)
```
 
## Installation
 
Python 3.10 ou plus récent est nécessaire.
 
```bash
git clone https://github.com/houudie/portfolio-multi-agent.git
cd portfolio-multi-agent
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```
 
Sur Mac, si LightGBM signale une erreur `libomp` : `brew install libomp`.
 
## Exécution
 
Les scripts se lancent depuis la racine du dépôt, dans cet ordre :
 
```bash
# 1. Données (Houda) : téléchargement, puis nettoyage, découpage et features
python -m src.data.download
python -m src.data.prepare
 
# 2. Analyse exploratoire : ouvrir et exécuter notebooks/01_eda.ipynb
 
# 3. Baselines et agent Risque (Houda)
python -m src.backtest.run_baselines
python -m src.agents.run_risk
 
# 4. Agent Prévision et interprétabilité (Rachid)
python -m src.models.train_forecast
python -m src.backtest.run_forecast_backtest
python -m src.models.interpret --model lightgbm      # SHAP, permutation, analyse des erreurs
 
# 5. Agent PPO (Abderrahmane)
python -m src.models.rl.train_ppo                    # modèle retenu (pénalité de risque 0,1)
python -m src.models.rl.train_ppo --risk 0.5 --suffix _risk05
python -m src.models.rl.train_ppo --turnover 0.02 --suffix _turnover
 
# 6. Règle d'allocation et pipeline multi-agent (Hatim)
python -m src.backtest.compare_rules                 # choix de la règle en validation
python -m src.agents.pipeline                        # démo du pipeline complet
 
# 7. Évaluation finale sur le test (Houda) - UNE SEULE FOIS, quand tout est figé
python -m src.evaluation.run_test --model lightgbm
python -m src.evaluation.run_test_ppo                # PPO, programme séparé (conflit OpenMP sur Mac)
python -m src.evaluation.robustness                 # bootstrap du Sharpe, frais, IC par année, diagnostic macro
 
# 8. Tests automatiques
python -m pytest
 
# 9. Application
streamlit run app/app.py
```
 
L'option `--suffix` de PPO évite d'écraser le modèle retenu (`models_saved/ppo_portfolio.zip`), utilisé par l'évaluation finale.
 
## Reproduction des résultats
 
Toutes les étapes lisent `config.yaml` : univers d'actifs, dates de découpage, frais de transaction et graine aléatoire (42). Relancer les scripts dans l'ordre ci-dessus reproduit les tableaux de `reports/` et les figures de `reports/figures/`.
 
## Données
 
| Source | Contenu | Accès |
| --- | --- | --- |
| Yahoo Finance (`yfinance`) | Prix journaliers ajustés de 20 actions, 4 ETF et de l'indice SPY | Gratuit, sans clé |
| FRED (Federal Reserve Bank of St. Louis) | Taux à 10 ans (DGS10), VIX (VIXCLS), taux directeur effectif quotidien (DFF) | Gratuit, sans clé (CSV) |
 
### Préparation (`src/data/prepare.py`)
 
1. **Nettoyage :** la période commence quand tous les actifs sont cotés (META depuis mai 2012) ; les trous de cotation de 5 jours maximum sont comblés avec le dernier prix connu.
2. **Découpage chronologique :** train jusqu'au 31/12/2019, validation 2020–2021, test 2022–2025.
3. **Features causales** par actif : momentum 21/63/126 jours, volatilité 21/63 jours, RSI 14, drawdown sur 252 jours ; features de marché (SPY) ; macro décalée d'un jour.
4. **Cible :** rendement log des 21 prochains jours de bourse.
5. **Purge :** les 21 derniers jours du train et de la validation sont retirés pour que les cibles ne débordent pas sur la période suivante.
## Règles anti-fuite de données
 
1. Le découpage **chronologique** train / validation / test est fait avant tout prétraitement.
2. Les scalers, modèles et seuils de régime sont ajustés sur le train uniquement. Pour l'évaluation finale, le modèle de prévision est réajusté sur train + validation, jamais sur le test.
3. Une feature à la date *t* n'utilise que des prix connus à la clôture de *t* ; les données macro sont décalées d'un jour. Les poids décidés à *t* ne s'appliquent qu'à partir de *t + 1*.
4. Les modèles et la règle d'allocation sont choisis sur la validation ; la période de test n'est évaluée qu'une seule fois, à la fin.
## Équipe
 
| Membre | Rôle |
| --- | --- |
| Houda Ait Hamou | Données, EDA, moteur de backtest, agent Risque, évaluation finale, robustesse |
| Rachid Hadi | Modèles de prévision, interprétabilité (SHAP, permutation, analyse des erreurs) |
| Abderrahmane Hasni | Agent RL (PPO), pénalité de turnover |
| Hatim Taouil | Orchestrateur, agents Profil, Allocation et Conseiller, application Streamlit |
 