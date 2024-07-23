from data.WhalesOptimizer import WhalesOptimizer
import streamlit as st
import json
import time
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.markdown("# Visualisation des Nappes")

###########################################
#         CONFIGURATION SIDEBAR
###########################################

# Définir les paramètres par défaut
indice1_config = {
    "w_whale": [],
    "ppc": 0.6,
    "phwh": 0.5,
    "rhwh": 1,
    "rbwh": -1,
    "phld": 0.5,
    "rhld": 1,
    "rbld": -1,
    "N_indice": 100,
    "N_ptf": 30
}

indice2_config = {
    "w_whale": [0.1, 0.1, 0.1, 0.1, 0.1],
    "ppc": 0.6,
    "phwh": [0.5, 0.5, 0.5, 0.5, 0.5],
    "rhwh": [1, 1, 1, 1, 1],
    "rbwh": [-1, -1, -1, -1, -1],
    "phld": 0.5,
    "rhld": 1,
    "rbld": -1,
    "N_indice": 100,
    "N_ptf": 30
}

indice3_config = {
    "w_whale": [0.5],
    "ppc": 0.6,
    "phwh": [0.5],
    "rhwh": [1],
    "rbwh": [-1],
    "phld": 0.5,
    "rhld": 1,
    "rbld": -1,
    "N_indice": 100,
    "N_ptf": 30
}


parameter_for = st.sidebar.selectbox(
    "Paramètres pour",
    ['Indice 1', 'Indice 2', 'Indice 3']
)

mapping = {
    'Indice 1' : indice1_config,
    'Indice 2' : indice2_config,
    'Indice 3' : indice3_config
}

default_config = mapping[parameter_for]

# Construire la configuration JSON à partir des entrées utilisateur
st.sidebar.markdown("# Whales Optimizer 🐳")
st.sidebar.header("Paramètres généraux :")

ppc = st.sidebar.slider("$$P_{pc}$$ : Probabilité de Prédictions Correctes", min_value=0.0, max_value=1.0, step=0.01, value=default_config["ppc"])
N_indice = st.sidebar.number_input("$$N_{indice}$$ : Nombre de valeurs dans l'indice", min_value=1, max_value=1000, value=default_config["N_indice"])
N_ptf = st.sidebar.number_input("$$N_{portefeuille}$$ : Nombre de valeurs dans le portefeuille", min_value=1, max_value=1000, value=default_config["N_ptf"])

st.sidebar.header("Paramètres des Whales :")

n_whale = st.sidebar.slider("Nombre de whales", min_value=1, max_value=10, step=1, value=len(default_config['w_whale']))
total_exposure = json.loads(str(st.sidebar.slider("Exposition totale des Whales", min_value=0.00, max_value=1.00, step=0.01, value=float(sum(default_config['w_whale'])))))
w_whale = list(np.full(n_whale, total_exposure/n_whale)) if n_whale > 0 else [float(0)]

first_value_phwh = float(default_config['phwh'][0]) if isinstance(default_config['phwh'], list) and len(default_config['phwh']) > 0 else float(0)
first_value_rhwh = float(default_config['rhwh'][0]) if isinstance(default_config['rhwh'], list) and len(default_config['rhwh']) > 0 else float(0)
first_value_rbwh = float(default_config['rbwh'][0]) if isinstance(default_config['rbwh'], list) and len(default_config['rbwh']) > 0 else float(0)
phwh = json.loads(str(st.sidebar.slider("$$P_{haussier}^{lambda}$$ : Probabilité de rendements haussiers (Whale)", min_value=0.00, max_value=1.00, step=0.01, value=first_value_phwh)))
rhwh = json.loads(str(st.sidebar.slider("$$R_{haussier}^{whale}$$ : Rendements haussiers (whale)", min_value=-5.0, max_value=5.0, step=0.1, value=first_value_rhwh)))
rbwh = json.loads(str(st.sidebar.slider("$$R_{baissier}^{whale}$$ : Rendements baissiers (whale)", min_value=-5.0, max_value=5.0, step=0.1, value=first_value_rbwh)))
st.sidebar.subheader(f"Il y a actuellement {n_whale} whale(s), toutes d'un poids de {w_whale[0]*100:.2f}% dans l'indice")

st.sidebar.header("Paramètres des Lambdas :")

phld = st.sidebar.slider("$$P_{haussier}^{lambda}$$ : Probabilité de rendements haussiers (Lambda)", min_value=0.0, max_value=1.0, step=0.01, value=default_config["phld"])
rhld = st.sidebar.slider("$$R_{haussier}^{lambda}$$ : Rendements haussiers (lambda)", min_value=-5.0, max_value=5.0, step=0.1, value=float(default_config["rhld"]))
rbld = st.sidebar.slider("$$R_{baissier}^{lambda}$$ : Rendements baissiers (lambda)", min_value=-5.0, max_value=5.0, step=0.1, value=float(default_config["rbld"]))
st.sidebar.text_area("**Attention**, les données pour $$E(XL)$$ sont pré-enregistrés pour certaine combinaisons. Si on sort de ces combinaisons, il y a un délai de calcul supplémentaire. Combinaisons pré-enregistrées :", value="N_Indice = 100, N_ptf = 30\n rhld, rbld = [1, -1], [3, -1], [1, -3], [3, -3]")

config = {
    "w_whale": w_whale,
    "phwh": phwh,
    "rhwh": rhwh,
    "rbwh": rbwh,
    "ppc": ppc,
    "phld": phld,
    "rhld": rhld,
    "rbld": rbld,
    "N_indice": N_indice,
    "N_ptf": N_ptf
}

json_config = json.dumps(config)

###########################################
#         CONFIGURATION SIDEBAR
###########################################

st.markdown("Dans cette section, les poids ne peuvent pas être différents pour la cohérence de nos graphiques. On va donc se limiter à une seule valeur pour les poids, les probabilités de rendements, pour les rendements haussiers et baissiers des whales et des lambdas.")

# Initialiser l'optimizer avec la configuration utilisateur
WO = WhalesOptimizer(json_config)

# Créer deux colonnes st, et met un graph WO.generate_3d_graph(100, WO.portfolio()) dans colonne 1, et WO.generate_3d_graph(100, WO.delta()) dans colonne 2
sharpness = st.slider("Sharpness", min_value=10, max_value=100, step=1, value=50)
col1, col2 = st.columns(2)
col1.plotly_chart(WO.generate_3d_graph(sharpness, 'portfolio'), use_container_width=True)
col2.plotly_chart(WO.generate_3d_graph(sharpness, 'delta'), use_container_width=True)

optimizing_type = st.selectbox(
    "Paramètres pour",
    ['portfolio', 'delta']
)
fig1, fig2, fig3, fig4 = WO.plot_surface(sharpness, optimizing_type)

col1, col2 = st.columns(2)
col1.plotly_chart(fig1, use_container_width=True)
col2.plotly_chart(fig2, use_container_width=True)

col1, col2 = st.columns(2)
col1.plotly_chart(fig3, use_container_width=True)
col2.plotly_chart(fig4, use_container_width=True)