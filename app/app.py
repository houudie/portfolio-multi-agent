"""Application Streamlit : assistant multi-agent de sélection de portefeuille (Hatim)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pour importer src/

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agents.base import SharedState
from src.agents.pipeline import DEFENSIVE, build_pipeline
from src.agents.profile import PROFILES, QUESTIONS, apply_constraints
from src.backtest.baselines import buy_and_hold, equal_weight
from src.backtest.engine import compare, run_backtest
from src.config import ROOT, load_config
from src.evaluation.run_test import profile_strategy

st.set_page_config(page_title="Assistant portefeuille multi-agent", layout="wide")
cfg = load_config()
UNIVERSE = cfg["universe"]["stocks"] + cfg["universe"]["etfs"]
BENCH = cfg["universe"]["benchmark"]
COST = cfg["backtest"]["transaction_cost"]
VAL_START = pd.Timestamp(cfg["split"]["train_end"]) + pd.Timedelta(days=1)
VAL_END = pd.Timestamp(cfg["split"]["val_end"])
COLUMNS = {"rendement_annuel_%": "Rendement annuel (%)", "volatilite_annuelle_%": "Volatilité (%)",
           "sharpe": "Sharpe", "sortino": "Sortino", "drawdown_max_%": "Perte max (%)",
           "calmar": "Calmar", "cvar_95_jour_%": "CVaR 95 % jour (%)", "turnover_moyen_%": "Turnover (%)"}


@st.cache_data
def load_prices() -> pd.DataFrame:
    return pd.read_csv(ROOT / cfg["paths"]["processed"] / "prices_clean.csv", index_col=0, parse_dates=True)


@st.cache_resource
def get_pipeline(as_of: str):
    return build_pipeline(cfg, as_of=as_of)


def equal_weight_profile(profile: str):
    """1/N avec les mêmes contraintes que le profil : la comparaison la plus juste."""
    c = PROFILES[profile]
    return lambda hist: apply_constraints(equal_weight(hist), DEFENSIVE, c["min_defensif"], c["poids_max"])


@st.cache_data(show_spinner="Backtest en cours...")
def backtest(profile: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assistant (profil choisi) contre 1/N et l'indice, sur la validation."""
    prices = load_prices()
    pipe = get_pipeline(str(VAL_END.date()))
    detector, preds = pipe.agents[3].detector, pipe.agents[2].predictions
    results = [
        run_backtest(prices[UNIVERSE], profile_strategy(profile, detector, prices[BENCH], preds),
                     VAL_START, VAL_END, name=f"Assistant ({profile})", cost=COST),
        run_backtest(prices[UNIVERSE], equal_weight_profile(profile),
                     VAL_START, VAL_END, name=f"1/N ({profile})", cost=COST),
        run_backtest(prices[UNIVERSE], equal_weight, VAL_START, VAL_END, name="1/N", cost=COST),
        run_backtest(prices[[BENCH]], buy_and_hold(BENCH), VAL_START, VAL_END, name=f"Indice {BENCH}", cost=COST),
    ]
    curves = pd.DataFrame({res.name: 100 * res.wealth for res in results})
    table = compare(results).rename(columns=COLUMNS)
    table.index.name = "Stratégie"
    return curves, table


st.title("Assistant multi-agent de sélection de portefeuille")
st.caption("Projet pédagogique, École Centrale Casablanca. Ce n'est pas un conseil en investissement.")

# ---------------------------------------------------------------- 1. questionnaire
with st.sidebar:
    st.header("1. Votre profil")
    answers = {q: st.radio(d["texte"], list(d["choix"]), key=q) for q, d in QUESTIONS.items()}
    st.header("2. Date de décision")
    as_of = st.date_input("Se placer au", value=pd.Timestamp("2021-06-30"),
                          min_value=VAL_START, max_value=VAL_END,
                          help="Période de validation (2020-2021) : les prévisions y sont disponibles.")
    if st.button("Construire mon portefeuille", type="primary", width="stretch"):
        st.session_state["demande"] = {"answers": answers, "as_of": str(as_of)}

if "demande" not in st.session_state:
    st.info("Répondez au questionnaire, choisissez une date, puis cliquez sur « Construire mon portefeuille ».")
    st.stop()

# ---------------------------------------------------------------- 2. pipeline des 6 agents
demande = st.session_state["demande"]
state = get_pipeline(demande["as_of"]).run(SharedState(profile={"reponses": demande["answers"]}))
p, r = state.profile, state.risk

c1, c2, c3, c4 = st.columns(4)
c1.metric("Profil", f"{p['profil']} ({p['score']}/20)")
c2.metric("Régime de marché", r["regime"])
c3.metric("VaR 95 % (jour)", f"{r['var_%']:.2f} %".replace(".", ","))
c4.metric("CVaR 95 % (jour)", f"{r['cvar_%']:.2f} %".replace(".", ","))

w = (100 * pd.Series(state.weights)).sort_values(ascending=False)
w = w[w > 0.5]
left, mid, right = st.columns([3, 2, 3])
fig_w = px.bar(x=w.index, y=w.values, labels={"x": "Actif", "y": "Poids (%)"},
               title=f"Portefeuille proposé au {demande['as_of']}")
left.plotly_chart(fig_w, width="stretch")
split = pd.Series({"Obligations et or": w.reindex(DEFENSIVE).fillna(0).sum()})
split["Actions et Nasdaq"] = w.sum() - split.iloc[0]
mid.plotly_chart(px.pie(values=split.values, names=split.index, hole=0.5, title="Répartition"),
                 width="stretch")
right.subheader("Explication de l'agent Conseiller")
right.markdown(state.explanation)
with st.expander("Journal des 6 agents"):
    st.code("\n".join(state.log))

# ---------------------------------------------------------------- 3. backtest
st.subheader("Et si vous aviez suivi ce conseil en 2020-2021 ?")
curves, table = backtest(p["profil"])
st.plotly_chart(px.line(curves, labels={"value": "Valeur (base 100)", "index": "", "variable": ""}),
                width="stretch")
st.dataframe(table, width="stretch")
st.caption("« 1/N (profil) » : poids égaux avec les mêmes contraintes que votre profil, la comparaison la plus juste. Backtest mensuel, frais de 0,1 % par transaction. Les performances passées ne préjugent pas des performances futures.")
