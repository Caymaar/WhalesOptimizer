from WhalesOptimizer import WhalesOptimizer
import streamlit as st
import json
import time
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.markdown("# Comparer Théorie et Simulation")

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
    'x': [],
    'y': [],
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
    'x': [0, 0, 0, 0, 0],
    'y': [0, 0, 0, 0, 0],
    "phld": 0.5,
    "rhld": 1,
    "rbld": -1,
    "N_indice": 100,
    "N_ptf": 30
}

indice3_config = {
    "w_whale": [0.5],
    "ppc": 0.6,
    "phwh": 0.5,
    "rhwh": 1,
    "rbwh": -1,
    'x': [0],
    'y': [0],
    "phld": 0.5,
    "rhld": 1,
    "rbld": -1,
    "N_indice": 100,
    "N_ptf": 30
}

custom_config = {
    "w_whale": [0.1, 0.05, 0.13, 0.11, 0.02, 0.04, 0.08],
    "ppc": 0.6,
    "phwh": [0.6, 0.5, 0.4, 0.55, 0.44, 0.51, 0.4],
    "rhwh": [1.2, 0.8, 1.1, 1.3, 0.5, 0.9, 0.5],
    "rbwh": [-0.8, -1, -0.8, -1.1, -1.1, -0.9, -0.5],
    'x': [0.02, 0.05, 0.04, 0.03, 0.02, 0.04, 0.01],
    'y': [0.02, 0.05, 0.04, 0.03, 0.02, 0.04, 0.01],
    "phld": 0.5,
    "rhld": 1,
    "rbld": -1,
    "N_indice": 100,
    "N_ptf": 30
}


parameter_for = st.sidebar.selectbox(
    "Paramètres pour",
    ['Custom', 'Indice 1', 'Indice 2', 'Indice 3']
)

mapping = {
    'Indice 1' : indice1_config,
    'Indice 2' : indice2_config,
    'Indice 3' : indice3_config,
    'Custom' : custom_config
}

default_config = mapping[parameter_for]

# Construire la configuration JSON à partir des entrées utilisateur
st.sidebar.markdown("# Whales Optimizer 🐳")
st.sidebar.header("Paramètres généraux :")

ppc = st.sidebar.slider("$$P_{pc}$$ : Probabilité de Prédictions Correctes", min_value=0.0, max_value=1.0, step=0.01, value=default_config["ppc"])
N_indice = st.sidebar.number_input("$$N_{indice}$$ : Nombre de valeurs dans l'indice", min_value=1, max_value=1000, value=default_config["N_indice"])
N_ptf = st.sidebar.number_input("$$N_{portefeuille}$$ : Nombre de valeurs dans le portefeuille", min_value=1, max_value=1000, value=default_config["N_ptf"])

st.sidebar.header("Paramètres des Whales :")

w_whale = json.loads(st.sidebar.text_input("$$w_{whale}$$ : Poid(s) des whales", value=str(default_config['w_whale'])))
phwh = json.loads(st.sidebar.text_input("$$P_{haussier}^{lambda}$$ : Probabilité de rendements haussiers (Lambda)", value=str(default_config['phwh'])))
rhwh = json.loads(st.sidebar.text_input("$$R_{haussier}^{whale}$$ : Rendements haussiers (whale)", value=str(default_config['rhwh'])))
rbwh = json.loads(st.sidebar.text_input("$$R_{baissier}^{whale}$$ : Rendements baissiers (whale)", value=str(default_config['rbwh'])))
x = json.loads(st.sidebar.text_input("$$x$$ : Paramètre de surpondération", value=str(default_config['x'])))
y = json.loads(st.sidebar.text_input("$$y$$ : Paramètre de sous pondération", value=str(default_config['y'])))
st.sidebar.subheader(f"Il y a actuellement {len(w_whale)} whale(s)")

st.sidebar.header("Paramètres des Lambdas :")

phld = st.sidebar.slider("$$P_{haussier}^{lambda}$$ : Probabilité de rendements haussiers (Lambda)", min_value=0.0, max_value=1.0, step=0.01, value=default_config["phld"])
rhld = st.sidebar.number_input("$$R_{haussier}^{lambda}$$ : Rendements haussiers (lambda)", min_value=-10, max_value=10, value=default_config["rhld"])
rbld = st.sidebar.number_input("$$R_{baissier}^{lambda}$$ : Rendements baissiers (lambda)", min_value=-10, max_value=10, value=default_config["rbld"])
st.sidebar.text_area("**Attention**, les données pour $$E(XL)$$ sont pré-enregistrés pour certaine combinaisons. Si on sort de ces combinaisons, il y a un délai de calcul supplémentaire. Combinaisons pré-enregistrées :", value="N_Indice = 100, N_ptf = 30\n rhld, rbld = [1, -1], [3, -1], [1, -3], [3, -3]")

config = {
    "w_whale": w_whale,
    "phwh": phwh,
    "rhwh": rhwh,
    "rbwh": rbwh,
    "x": x,
    "y": y,
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

# Initialiser l'optimizer avec la configuration utilisateur
WO = WhalesOptimizer(json_config)

# Simulation du portefeuille
st.header("Simulation des différentes distributions")
precision = st.slider("Select Precision", min_value=1000, max_value=10000, step=1000, value=5000)
df = WO.simulate_portfolio(precision)
st.dataframe(df)


# Comparaison des distributions
st.header("Comparer les distributions souhaitées")
distributions_to_compare = st.multiselect(
    "Selectionne les distributions à comparer",
    ['benchmark', 'B', 'L', 'La', 'portfolio', 'Y', 'X', 'W', 'XW', 'delta', 'WY', 'WB', 'YB', 'XL'],
    default=['benchmark', 'portfolio', 'delta']
)
decimals = st.number_input("Nombre de décimales", min_value=0, max_value=5, value=3)
compare_results = WO.compare(precision, distributions_to_compare, decimals, print_results=False)
st.write(compare_results)

hist_values = np.histogram(df['portfolio'], bins=24, range=(0,24))[0]
