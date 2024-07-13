from data.WhalesOptimizer import WhalesOptimizer
import streamlit as st
import json
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(layout="wide")

st.markdown("# Whales Optimizer 🐳")

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

x_y_for = st.selectbox(
    "Choisir les paramètres de x et y :",
    ['Initiaux', 'Optimisant le ratio de Sharpe', "Optimisant le ratio d'Information"]
)

if x_y_for == "Optimisant le ratio d'Information":
    WO.x, WO.y, sharpe = WO.optimize_parameters(WO.delta, print_results=False)
elif x_y_for == 'Optimisant le ratio de Sharpe':
    WO.x, WO.y, sharpe = WO.optimize_parameters(WO.portfolio, print_results=False) 

WO.plot_whale_values(streamlit=True)

if st.checkbox('Montrer les paramètres des whales dans un DataFrame'):
    st.dataframe(WO.get_whale_parameters_df(), use_container_width=True)
WO.simulate_and_plot(streamlit=True)


# Met l'output WO.x et WS.y dispo en sorti brut à la fin
# Convertir WO.x et WO.y en chaînes de caractères avec une virgule entre chaque valeur et le tout entre crochets
formatted_x = "[" + ", ".join(map(str, WO.x)) + "]"
formatted_y = "[" + ", ".join(map(str, WO.y)) + "]"

# Afficher les valeurs formatées dans Streamlit
st.text_input("Valeurs de x utilisées :", value=formatted_x)  # Hauteur en pixels
st.text_input("Valeurs de y utilisées :", value=formatted_y)  # Hauteur en pixels


