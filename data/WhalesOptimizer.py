import json
import numpy as np
import pandas as pd
from itertools import product
from collections import Counter
from math import factorial
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize, Bounds
from scipy.signal import savgol_filter
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
from plotly.subplots import make_subplots



class WhalesOptimizer:

    ##########################################################
    # Initialisation
    ##########################################################

    def __init__(self, json_config):
        data = np.load('data/simulated_XL_values1.npz')
        self.interpolator_sym_1 = RegularGridInterpolator((np.linspace(0, 1, 100), np.linspace(0, 1, 100)), data['sym_1'])
        self.interpolator_sym_3 = RegularGridInterpolator((np.linspace(0, 1, 100), np.linspace(0, 1, 100)), data['sym_3'])
        self.interpolator_asym_1 = RegularGridInterpolator((np.linspace(0, 1, 100), np.linspace(0, 1, 100)), data['asym_1'])
        self.interpolator_asym_3 = RegularGridInterpolator((np.linspace(0, 1, 100), np.linspace(0, 1, 100)), data['asym_3'])

        # Set to false "optimzing"
        self.optimizing = False

        # Charger le JSON et initialiser les paramètres
        config = json.loads(json_config)
        self.w_whale = np.array(config['w_whale'])
        self.ppc = config['ppc']

        if isinstance(config['phwh'], list):
            self.phwh = np.array(config['phwh'])
        else:
            self.phwh = np.full(len(self.w_whale), config['phwh'])

        if isinstance(config['rhwh'], list):
            self.rhwh = np.array(config['rhwh'])
        else:
            self.rhwh = np.full(len(self.w_whale), config['rhwh'])

        if isinstance(config['rbwh'], list):
            self.rbwh = np.array(config['rbwh'])
        else:
            self.rbwh = np.full(len(self.w_whale), config['rbwh'])
            
        self.phld = config['phld']
        self.rhld = config['rhld']
        self.rbld = config['rbld']
        self.N_indice = config['N_indice']
        self.n_whale = len(self.w_whale)
        self.N_ptf = config['N_ptf']

        if 'x' in config:
            self.x = np.array(config['x'])
        else:
            self.x = np.zeros(self.n_whale)
        if 'y' in config:
            self.y = np.array(config['y'])
        else:
            self.y = np.zeros(self.n_whale)

    #################################################################################
    #                                                                               #              
    #               FONCTIONS DE CALCUL DES ESPERANCES ET VARIANCES                 #
    #                                                                               #
    #################################################################################

    ##########################################################
    # Calcul du Benchmark
    ##########################################################

    def benchmark(self):
        E_B, Var_B = self.B_function()
        E_La, Var_La = self.La_function()
        E_R = E_B + E_La
        Var_R = Var_B + Var_La
        return E_R, Var_R

    def B_function(self):
        E_B_correct = self.phwh * self.w_whale * self.rhwh
        E_B_incorrect = (1 - self.phwh) * self.w_whale * self.rbwh
        E_B = np.sum(E_B_correct + E_B_incorrect)
        Var_B_correct = self.phwh * (self.w_whale * self.rhwh) ** 2
        Var_B_incorrect = (1 - self.phwh) * (self.w_whale * self.rbwh) ** 2
        Var_B = np.sum(Var_B_correct + Var_B_incorrect - (E_B_correct + E_B_incorrect) ** 2)
        return E_B, Var_B

    def L_function(self):
        E_S_correct = self.phld * self.rhld
        E_S_incorrect = (1 - self.phld) * self.rbld
        E_S = E_S_correct + E_S_incorrect
        Var_S_i = (self.phld * (self.rhld) ** 2 + (1 - self.phld) * (self.rbld) ** 2) - E_S ** 2
        Var_S = Var_S_i / (self.N_indice - self.n_whale)
        return E_S, Var_S

    def a_function(self):
        return 1 - np.sum(self.w_whale)

    def La_function(self):
        E_L, Var_L = self.L_function()
        a = self.a_function()
        E_La = E_L * a
        Var_La = Var_L * a ** 2
        return E_La, Var_La

    ##########################################################
    # Calcul du Portefeuille
    ##########################################################

    def portfolio(self):
        E_Y, Var_Y = self.Y_function()
        E_XW, Var_XW = self.XW_function()
        Cov_XW_Y = self.Cov_XW_Y_function()
        E_R = E_Y + E_XW
        Var_R = Var_XW + Var_Y + 2 * Cov_XW_Y
        return E_R, Var_R

    def Y_function(self):
        E_Y_correct = self.phwh * (self.ppc * (self.w_whale + self.x) + (1 - self.ppc) * (self.w_whale - self.y)) * self.rhwh
        E_Y_incorrect = (1 - self.phwh) * (self.ppc * (self.w_whale - self.y) + (1 - self.ppc) * (self.w_whale + self.x)) * self.rbwh
        E_Y = np.sum(E_Y_correct + E_Y_incorrect)
        Var_Y_correct = self.phwh * (self.rhwh ** 2) * (self.ppc * (self.w_whale + self.x) ** 2 + (1 - self.ppc) * (self.w_whale - self.y) ** 2)
        Var_Y_incorrect = (1 - self.phwh) * (self.rbwh ** 2) * (self.ppc * (self.w_whale - self.y) ** 2 + (1 - self.ppc) * (self.w_whale + self.x) ** 2)
        Var_Y = np.sum(Var_Y_correct + Var_Y_incorrect - (E_Y_correct + E_Y_incorrect) ** 2)
        return E_Y, Var_Y

    def X_function(self):
        E_X_correct = ((self.phld * self.ppc) / (self.phld * self.ppc + (1 - self.phld) * (1 - self.ppc))) * self.rhld
        E_X_incorrect = (1 - ((self.phld * self.ppc) / (self.phld * self.ppc + (1 - self.phld) * (1 - self.ppc)))) * self.rbld
        E_X = E_X_correct + E_X_incorrect
        Var_X_i = (((self.phld * self.ppc) / (self.phld * self.ppc + (1 - self.phld) * (1 - self.ppc))) * (self.rhld ** 2) + (1 - ((self.phld * self.ppc) / (self.phld * self.ppc + (1 - self.phld) * (1 - self.ppc)))) * (self.rbld ** 2) - E_X ** 2)
        Var_X = Var_X_i / (self.N_ptf - self.n_whale)
        return E_X, Var_X

    def W_function(self):
        E_Adj = (self.phwh * self.ppc + (1 - self.phwh) * (1 - self.ppc)) * self.x - (self.phwh * (1 - self.ppc) + (1 - self.phwh) * self.ppc) * self.y
        E_W = 1 - np.sum(self.w_whale + E_Adj)
        Var_W = np.sum((self.phwh * self.ppc + (1 - self.phwh) * (1 - self.ppc)) * (self.x - E_Adj) ** 2 + (self.phwh * (1 - self.ppc) + (1 - self.phwh) * self.ppc) * (-self.y - E_Adj) ** 2)
        return E_W, Var_W

    def XW_function(self):
        E_X, Var_X = self.X_function()
        E_W, Var_W = self.W_function()
        E_XW = E_X * E_W
        Var_XW = Var_X * E_W ** 2 + Var_W * E_X ** 2 + Var_W * Var_X
        return E_XW, Var_XW
        
    ##########################################################
    # Calcul du Delta
    ##########################################################

    def delta(self):
        E_R_ptf, Var_R_ptf = self.portfolio()
        E_R_benchmark, Var_R_benchmark = self.benchmark()
        Cov_Rptf_Rindice = self.Cov_Rptf_Rindice_function()
        Esperance = E_R_ptf - E_R_benchmark
        Var = Var_R_ptf + Var_R_benchmark - 2 * Cov_Rptf_Rindice
        return Esperance, Var

    ##########################################################
    # Calcul des espérances spécifiques
    ##########################################################

    def esperance_WY(self):
        n = len(self.w_whale)
        if n == 0:
            return 0
        rhx_matrix = np.array([self.rhwh * (self.w_whale + self.x), np.full(n, self.phwh * self.ppc), self.w_whale + self.x])
        rhy_matrix = np.array([self.rhwh * (self.w_whale - self.y), np.full(n, self.phwh * (1 - self.ppc)), self.w_whale - self.y])
        rby_matrix = np.array([self.rbwh * (self.w_whale - self.y), np.full(n, (1 - self.phwh) * self.ppc), self.w_whale - self.y])
        rbx_matrix = np.array([self.rbwh * (self.w_whale + self.x), np.full(n, (1 - self.phwh) * (1 - self.ppc)), self.w_whale + self.x])
        elements_matrix = np.stack([rhx_matrix, rhy_matrix, rby_matrix, rbx_matrix], axis=0)
        combinations = np.array(list(product(range(4), repeat=n)))
        prob_products = np.prod(elements_matrix[combinations, 1, np.arange(n)], axis=1)
        value_sums = np.sum(elements_matrix[combinations, 0, np.arange(n)], axis=1)
        adjust_sums = np.sum(elements_matrix[combinations, 2, np.arange(n)], axis=1)
        terms = prob_products * value_sums * (1 - adjust_sums)
        return np.sum(terms)

    def esperance_WB(self):
        n = len(self.w_whale)
        if n == 0:
            return 0
        rhx_matrix = np.array([self.rhwh * self.w_whale, np.full(n, self.phwh * self.ppc), self.w_whale + self.x])
        rhy_matrix = np.array([self.rhwh * self.w_whale, np.full(n, self.phwh * (1 - self.ppc)), self.w_whale - self.y])
        rby_matrix = np.array([self.rbwh * self.w_whale, np.full(n, (1 - self.phwh) * self.ppc), self.w_whale - self.y])
        rbx_matrix = np.array([self.rbwh * self.w_whale, np.full(n, (1 - self.phwh) * (1 - self.ppc)), self.w_whale + self.x])
        elements_matrix = np.stack([rhx_matrix, rhy_matrix, rby_matrix, rbx_matrix], axis=0)
        combinations = np.array(list(product(range(4), repeat=n)))
        prob_products = np.prod(elements_matrix[combinations, 1, np.arange(n)], axis=1)
        value_sums = np.sum(elements_matrix[combinations, 0, np.arange(n)], axis=1)
        adjust_sums = np.sum(elements_matrix[combinations, 2, np.arange(n)], axis=1)
        terms = prob_products * value_sums * (1 - adjust_sums)
        return np.sum(terms)

    def esperance_YB(self):
        n = len(self.w_whale)
        if n == 0:
            return 0
        rhx_matrix = np.array([self.rhwh * (self.w_whale + self.x), np.full(n, self.phwh * self.ppc), self.rhwh * self.w_whale])
        rhy_matrix = np.array([self.rhwh * (self.w_whale - self.y), np.full(n, self.phwh * (1 - self.ppc)), self.rhwh * self.w_whale])
        rby_matrix = np.array([self.rbwh * (self.w_whale - self.y), np.full(n, (1 - self.phwh) * self.ppc), self.rbwh * self.w_whale])
        rbx_matrix = np.array([self.rbwh * (self.w_whale + self.x), np.full(n, (1 - self.phwh) * (1 - self.ppc)), self.rbwh * self.w_whale])
        elements_matrix = np.stack([rhx_matrix, rhy_matrix, rby_matrix, rbx_matrix], axis=0)
        combinations = np.array(list(product(range(4), repeat=n)))
        prob_products = np.prod(elements_matrix[combinations, 1, np.arange(n)], axis=1)
        value_Y_sums = np.sum(elements_matrix[combinations, 0, np.arange(n)], axis=1)
        value_B_sums = np.sum(elements_matrix[combinations, 2, np.arange(n)], axis=1)
        terms = prob_products * value_Y_sums * value_B_sums
        return np.sum(terms)
        
    def esperance_XL(self):
        point = np.array([self.ppc, self.phld])
        if self.rhld == 1 and self.rbld == -1:
            smoothed_value = self.interpolator_sym_1(point)
        elif self.rhld == 3 and self.rbld == -3:
            smoothed_value = self.interpolator_sym_3(point)
        elif self.rhld == 3 and self.rbld == -1:
            smoothed_value = self.interpolator_asym_3(point)
        elif self.rhld == 1 and self.rbld == -3:
            smoothed_value = self.interpolator_asym_1(point)
        else:
            print('Invalid values for rhld and rbld, we will be using simulate_XL function')
            return self.get_XL_values(10000, 11)
        return smoothed_value[0]
    
    def get_XL_values(self, Precision, n):

        # vérifier que n est impair
        if n % 2 == 0:
            n += 1
        
        # Définissez les plages pour les deux paramètres
        ppc_range = np.linspace(self.ppc - 0.1, self.ppc + 0.1, n)
        phld_range = np.linspace(self.phld - 0.1, self.phld + 0.1, n)

        # stock ppc et phld
        ppc = self.ppc
        phld = self.phld

        # Créez une grille pour les simulations
        simulated_values_ppc = np.zeros(n)
        simulated_values_phld = np.zeros(n)

        for i, ppc_i in enumerate(ppc_range):
            self.ppc = ppc_i
            simulated_values_ppc[i] = self.simulate_XL(Precision)

        self.ppc = ppc

        for i, phld_i in enumerate(phld_range):
            self.phld = phld_i
            simulated_values_phld[i] = self.simulate_XL(Precision)

        self.phld = phld

        values_from_ppc = savgol_filter(simulated_values_ppc, window_length=n, polyorder=2)
        values_from_phld = savgol_filter(simulated_values_phld, window_length=n, polyorder=2)

        mid_point = n // 2

        mean_value = (values_from_ppc[mid_point] + values_from_phld[mid_point]) / 2
        return mean_value
    
    def simulate_XL(self, Precision, df=False):
        n_lambda_indice = self.N_indice - self.n_whale
        n_lambda_ptf = self.N_ptf - self.n_whale

        #np.random.seed(42)
        probas = np.array([self.ppc, 1 - self.ppc])

        # lambdas
        returns_ld = np.random.choice([self.rhld, self.rbld], size=(Precision, n_lambda_indice), p=[self.phld, 1 - self.phld])
        actions_momentum_ld = (returns_ld == self.rhld).astype(int)
        actions_momentum_pred_ld = np.where(actions_momentum_ld == 1, 
                                            np.random.choice([1, 0], size=(Precision, n_lambda_indice), p=probas), 
                                            np.random.choice([0, 1], size=(Precision, n_lambda_indice), p=probas))

        #Remove seed
        np.random.seed()

        # lambda Portefeuille
        weights_ld_XL = np.full((Precision, n_lambda_indice), 1 / n_lambda_indice)
        sorted_indices = np.argsort(-actions_momentum_pred_ld, axis=1)
        weights_ld_XL_ptf = np.zeros_like(weights_ld_XL)
        
        # Utiliser l'indexation avancée pour attribuer les valeurs à weights_ld_ptf
        rows = np.arange(Precision).reshape(-1, 1)  # Créer un tableau d'indices de ligne
        # Attribuer les valeurs à weights_ld_XL_ptf
        weights_ld_XL_ptf[rows, sorted_indices[:, :n_lambda_ptf]] = 1 / n_lambda_ptf

        XL = np.sum(returns_ld * weights_ld_XL_ptf, axis=1) * np.sum(returns_ld * weights_ld_XL, axis=1)

        # Mettre Y, XW, B, La dans un dataframe
        if df:
            return pd.DataFrame({'XL': XL,'X': np.sum(returns_ld * weights_ld_XL, axis=1),'L': np.sum(returns_ld * weights_ld_XL_ptf, axis=1)})
        
        else:
            return np.mean(XL)

    ##########################################################
    # Calcul des covariances
    ##########################################################

    def Cov_XW_Y_function(self):
        E_XW = self.XW_function()[0]
        E_Y = self.Y_function()[0]
        E_X = self.X_function()[0]
        E_WY = self.esperance_WY()
        return E_X * E_WY - E_XW * E_Y

    def Cov_Rptf_Rindice_function(self):
        if self.optimizing:
            E_XL = self.E_XL
        else: 
            E_XL = self.esperance_XL()
        E_W = self.W_function()[0]
        E_La = self.La_function()[0]
        E_X = self.X_function()[0]
        E_WB = self.esperance_WB()
        E_Y = self.Y_function()[0]
        E_YB = self.esperance_YB()
        a = self.a_function()
        E_R_ptf = self.portfolio()[0]
        E_R_indice = self.benchmark()[0]
        return a * E_W * E_XL + E_X * E_WB + E_Y * E_La + E_YB - E_R_ptf * E_R_indice
    
    #################################################################################
    #                                                                               #          
    #                 FONCTIONS DE PRODUCTION ET DE VISUALISATION                   #
    #                                                                               #
    #################################################################################

    def simulate_portfolio(self, Precision):

        """
        Simule les différentes distributions en utilisant les paramètres actuels
        """

        n_lambda_indice = self.N_indice - self.n_whale
        exposition_lambda_indice = 1 - np.sum(self.w_whale)
        w_lambda_indice = exposition_lambda_indice / n_lambda_indice
        n_lambda_ptf = self.N_ptf - self.n_whale

        # whales
        returns_wh = np.zeros((Precision, self.n_whale))
        actions_momentum_wh = np.zeros((Precision, self.n_whale), dtype=int)
        actions_momentum_pred_wh = np.zeros((Precision, self.n_whale), dtype=int)

        # Check if phwh, rhwh, rbwh are lists or single values
        for i in range(self.n_whale):
            returns_wh[:, i] = np.random.choice([self.rhwh[i], self.rbwh[i]], size=Precision, p=[self.phwh[i], 1 - self.phwh[i]])
            actions_momentum_wh[:, i] = (returns_wh[:, i] == self.rhwh[i]).astype(int)
            actions_momentum_pred_wh[:, i] = np.where(actions_momentum_wh[:, i] == 1, 
                                                    np.random.choice([1, 0], size=Precision, p=[self.ppc, 1 - self.ppc]), 
                                                    np.random.choice([0, 1], size=Precision, p=[self.ppc, 1 - self.ppc]))
 
        # whale Indice
        return_whale_indice = np.sum(returns_wh * self.w_whale, axis=1)

        # whale Portefeuille
        weights_wh_ptf = self.w_whale + np.where(actions_momentum_pred_wh == 1, self.x, -self.y)
        return_whale_ptf = np.sum(returns_wh * weights_wh_ptf, axis=1)

        # lambdas
        returns_ld = np.random.choice([self.rhld, self.rbld], size=(Precision, n_lambda_indice), p=[self.phld, 1 - self.phld])
        actions_momentum_ld = (returns_ld == self.rhld).astype(int)
        actions_momentum_pred_ld = np.where(actions_momentum_ld == 1, 
                                            np.random.choice([1, 0], size=(Precision, n_lambda_indice), p=[self.ppc, 1 - self.ppc]), 
                                            np.random.choice([0, 1], size=(Precision, n_lambda_indice), p=[self.ppc, 1 - self.ppc]))

        # lambda Indice
        weights_ld = np.full((Precision, n_lambda_indice), w_lambda_indice)
        weights_ld_XL = np.full((Precision, n_lambda_indice), 1 / n_lambda_indice)
        return_lambda_indice = np.sum(returns_ld * weights_ld, axis=1)

        # lambda Portefeuille
        sorted_indices = np.argsort(-actions_momentum_pred_ld, axis=1)
        weights_ld_ptf = np.zeros_like(weights_ld)
        weights_ld_XL_ptf = np.zeros_like(weights_ld_XL)
        
        # Calculer les poids pour weights_ld_ptf
        sums = np.sum(weights_wh_ptf, axis=1)  # Somme de chaque ligne de weights_wh_ptf
        weights_values = (1 - sums) / n_lambda_ptf  # Calculer les valeurs des poids

        # Utiliser l'indexation avancée pour attribuer les valeurs à weights_ld_ptf
        rows = np.arange(Precision).reshape(-1, 1)  # Créer un tableau d'indices de ligne
        weights_ld_ptf[rows, sorted_indices[:, :n_lambda_ptf]] = weights_values.reshape(-1, 1)

        # Attribuer les valeurs à weights_ld_XL_ptf
        weights_ld_XL_ptf[rows, sorted_indices[:, :n_lambda_ptf]] = 1 / n_lambda_ptf

        return_lambda_ptf = np.sum(returns_ld * weights_ld_ptf, axis=1)
        X_L = np.sum(returns_ld * weights_ld_XL_ptf, axis=1) * np.sum(returns_ld * weights_ld_XL, axis=1)
        X = np.sum(returns_ld * weights_ld_XL_ptf, axis=1)
        L = np.sum(returns_ld * weights_ld_XL, axis=1)
        W = 1 - np.sum(weights_wh_ptf, axis=1)
        # Mettre Y, XW, B, La dans un dataframe
        df = pd.DataFrame({
            "benchmark": return_whale_indice + return_lambda_indice,
            "B": return_whale_indice,
            "L": L,
            "La": return_lambda_indice,
            "portfolio": return_whale_ptf + return_lambda_ptf,
            "Y": return_whale_ptf,
            "X": X,
            "W": W,
            "XW": return_lambda_ptf,
            "delta": (return_whale_ptf + return_lambda_ptf) - (return_lambda_indice + return_whale_indice),
            "WY": W * return_whale_ptf,
            "WB": W * return_whale_indice,
            "YB": return_whale_ptf * return_whale_indice,
            "XL": X_L,
        })
        
        return df

    def optimize_parameters(self, func, print_results=False):

        """
        Optimise les paramètres x et y pour maximiser le ratio de Sharpe/Information
        """

        #Stockage de x et y originaux
        x_original = self.x
        y_original = self.y

        # Calcule de l'espérance de XL une unique fois
        self.optimizing = True
        self.E_XL = self.esperance_XL()

        def neg_sharpe_ratio(params):
            self.x = params[:self.n_whale]
            self.y = params[self.n_whale:]
            esp, var = func()
            sharpe_ratio = esp / np.sqrt(var)
            return -sharpe_ratio

        # Initialiser x et y comme des tableaux de longueur n_whale avec que des 0
        initial_guess = np.zeros(2 * self.n_whale)

        # Définir les bornes pour chaque élément de x et y
        lower_bounds = np.concatenate([-self.w_whale, -self.w_whale])
        upper_bounds = np.concatenate([self.w_whale, self.w_whale])
        bounds = Bounds(lower_bounds, upper_bounds)

        # Optimiser en utilisant SLSQP
        result = minimize(neg_sharpe_ratio, initial_guess, method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-10, 'disp': False})

        optimal_params = result.x
        self.optimal_x = optimal_params[:self.n_whale]
        self.optimal_y = optimal_params[self.n_whale:]
        self.optimal_sharpe_ratio = -result.fun

        if print_results:
            print(f'Optimal x: {self.optimal_x}')
            print(f'Optimal y: {self.optimal_y}')
            print(f'\nSharpe Ratio: {self.optimal_sharpe_ratio:.3f}')

        # Remettre les valeurs originales de x et y
        self.x = x_original
        self.y = y_original
        self.optimizing = False
        
        return self.optimal_x, self.optimal_y, self.optimal_sharpe_ratio

    def compare(self, Precision, distributions_to_compare, decimals=None, print_results=True):
        
        """
        Compare les valeurs simulées avec les valeurs attendues pour les distributions spécifiées
        """

        mapping = {
            "benchmark": self.benchmark,
            "B": self.B_function,
            "L": self.L_function,
            "La": self.La_function,
            "portfolio": self.portfolio,
            "Y": self.Y_function,
            "X": self.X_function,
            "W": self.W_function,
            "XW": self.XW_function,
            "delta": self.delta,
            "WY": self.esperance_WY,
            "WB": self.esperance_WB,
            "YB": self.esperance_YB,
            "XL": self.esperance_XL
        }

        simulated_df = self.simulate_portfolio(Precision)
        
        results = {}
        for distribution_name in distributions_to_compare:
            func = mapping[distribution_name]
            expected_value = func()  # Get the expected value from the function
            
            # Si expected_value est un tuple, on doit comparer la moyenne et la variance
            if isinstance(expected_value, tuple):
                expected_mean = expected_value[0]
                expected_variance = expected_value[1]
                simulated_mean = simulated_df[distribution_name].mean()
                simulated_variance = simulated_df[distribution_name].var()
                
                if decimals is not None:
                    expected_mean = round(expected_mean, decimals)
                    expected_variance = round(expected_variance, decimals)
                    simulated_mean = round(simulated_mean, decimals)
                    simulated_variance = round(simulated_variance, decimals)
                    mean_diff = round(expected_mean - simulated_mean, decimals)
                    variance_diff = round(expected_variance - simulated_variance, decimals)
                else:
                    mean_diff = expected_mean - simulated_mean
                    variance_diff = expected_variance - simulated_variance
                
                results[distribution_name] = {
                    "Expected Mean": expected_mean,
                    "Simulated Mean": simulated_mean,
                    "Mean Difference": mean_diff,
                    "Expected Variance": expected_variance,
                    "Simulated Variance": simulated_variance,
                    "Variance Difference": variance_diff
                }
            else:
                expected_mean = expected_value
                simulated_mean = simulated_df[distribution_name].mean()
                
                if decimals is not None:
                    expected_mean = round(expected_mean, decimals)
                    simulated_mean = round(simulated_mean, decimals)
                    mean_diff = round(expected_mean - simulated_mean, decimals)
                else:
                    mean_diff = expected_mean - simulated_mean
                
                results[distribution_name] = {
                    "Expected Mean": expected_mean,
                    "Simulated Mean": simulated_mean,
                    "Mean Difference": mean_diff
                }

        if print_results:
            for distribution_name, values in results.items():
                print(f'{distribution_name}:')
                for key, value in values.items():
                    print(f'{key}: {value}')
                print()
        
        return results
    
    def sharpe_ratio(self):

        """
        Calcule le ratio de Sharpe pour une distribution spécifiée
        """

        E, Var = self.portfolio()
        return E / np.sqrt(Var)
    
    def information_ratio(self):

        """
        Calcule le ratio d'Information pour une distribution spécifiée
        """
        
        E, Var = self.delta()
        return E / np.sqrt(Var)

    def show_parameters(self):

        print(f'w_whale: {self.w_whale}')
        print(f'n_whale: {self.n_whale}')
        print(f'x: {self.x}')
        print(f'y: {self.y}')
        print(f'ppc: {self.ppc}')
        print(f'phwh: {self.phwh}')
        print(f'rhwh: {self.rhwh}')
        print(f'rbwh: {self.rbwh}')
        print(f'phld: {self.phld}')
        print(f'rhld: {self.rhld}')
        print(f'rbld: {self.rbld}')
        print(f'N_indice: {self.N_indice}')
        print(f'N_ptf: {self.N_ptf}')

    def simulate_and_plot(self, streamlit=False):
        
        simulated_df = self.simulate_portfolio(10000)

        mu_ptf, var_ptf = self.portfolio()
        sigma_ptf = np.sqrt(var_ptf)
        x_values_ptf = np.linspace(mu_ptf - 3*sigma_ptf, mu_ptf + 3*sigma_ptf, 100)

        mu_delta, var_delta = self.delta()
        sigma_delta = np.sqrt(var_delta)
        x_values_delta = np.linspace(mu_delta - 3*sigma_delta, mu_delta + 3*sigma_delta, 100)

        proba_theorique_ptf = norm.sf(0, mu_ptf, sigma_ptf) * 100
        proba_simulee_ptf = (simulated_df['portfolio'] > 0).mean() * 100
        proba_theorique_delta = norm.sf(0, mu_delta, sigma_delta) * 100
        proba_simulee_delta = (simulated_df['delta'] > 0).mean() * 100

        # Plotly plots for 'Rendement Portefeuille'
        fig_ptf = go.Figure()

        hist_data_ptf = simulated_df['portfolio']
        kde_x_ptf = np.linspace(hist_data_ptf.min(), hist_data_ptf.max(), 100)
        kde_y_ptf = norm.pdf(kde_x_ptf, mu_ptf, sigma_ptf)

        fig_ptf.add_trace(go.Histogram(x=hist_data_ptf, nbinsx=50, name='Densité simulée', histnorm='probability density', marker_color='green'))
        fig_ptf.add_trace(go.Scatter(x=kde_x_ptf, y=kde_y_ptf, mode='lines', name='Densité théorique', line=dict(color='red')))
        fig_ptf.add_vline(x=0, line=dict(color='blue', dash='dash'))

        fig_ptf.update_layout(
            title='Portefeuille',
            title_x=0.3,
            xaxis_title='Rendement Portefeuille',
            yaxis_title='Densité',
            showlegend=True
        )
        # Plotly plots for 'Delta'
        fig_delta = go.Figure()

        hist_data_delta = simulated_df['delta']
        kde_x_delta = np.linspace(hist_data_delta.min(), hist_data_delta.max(), 100)
        kde_y_delta = norm.pdf(kde_x_delta, mu_delta, sigma_delta)

        fig_delta.add_trace(go.Histogram(x=hist_data_delta, nbinsx=50, name='Densité simulée', histnorm='probability density', marker_color='blue'))
        fig_delta.add_trace(go.Scatter(x=kde_x_delta, y=kde_y_delta, mode='lines', name='Densité théorique', line=dict(color='red')))
        fig_delta.add_vline(x=0, line=dict(color='green', dash='dash'))
        
        fig_delta.update_layout(
            title='Delta',
            title_x=0.4,
            xaxis_title='Rendement Delta',
            yaxis_title='Densité',
            showlegend=True
        )

        # Show the figures in Streamlit
        if streamlit:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_ptf, use_container_width=True)
                st.markdown(f'$$\mu$$ : **{mu_ptf:.3f}**, $$\sigma$$ : **{sigma_ptf:.3f}**')
                st.markdown('**Probabilité de performance (Portefeuille):**')
                st.markdown(f'théorique : **{proba_theorique_ptf:.2f}%**, simulée : **{proba_simulee_ptf:.2f}%**')
                st.markdown(f'Ratio de Sharpe : **{self.sharpe_ratio():.3f}**')

            with col2:
                st.plotly_chart(fig_delta, use_container_width=True)
                st.markdown(f'$$\mu$$ : **{mu_delta:.3f}**, $$\sigma$$ : **{sigma_delta:.3f}**')
                st.markdown('**Probabilité de sur-performance (Delta):**')
                st.markdown(f'théorique : **{proba_theorique_delta:.2f}%**, simulée : **{proba_simulee_delta:.2f}%**')
                st.markdown(f"Ratio d'Information : **{self.information_ratio():.3f}**")
        else:
            fig_ptf.show()
            print('Probabilité de performance (Portefeuille):')
            print(f'théorique : {proba_theorique_ptf:.2f}%, simulée : {proba_simulee_ptf:.2f}%')
            print(f'mu = {mu_ptf:.3f}, sqrt = {sigma_ptf:.3f}')
            print(f'Ratio de Sharpe : {self.sharpe_ratio():.3f}')
            fig_delta.show()
            print('Probabilité de sur-performance (Delta):')
            print(f'théorique : {proba_theorique_delta:.2f}%, simulée : {proba_simulee_delta:.2f}%')
            print(f'mu = {mu_delta:.3f}, sqrt = {sigma_delta:.3f}')
            print(f'Ratio de Sharpe : {self.information_ratio():.3f}')

    def plot_whale_values(self, streamlit=False):
        labels = [f'Whale {i+1}' for i in range(len(self.w_whale))]

        # Calculer les nouvelles valeurs de x et y
        x_values = self.w_whale + self.x
        y_values = self.w_whale - self.y

        # Créer deux colonnes
        col1, col2 = st.columns(2)

        # Placer chaque checkbox dans sa propre colonne
        with col1:
            show_probabilities = st.checkbox('Afficher les probabilités de hausse', value=True)
        with col2:
            show_returns = st.checkbox('Afficher les rendements haussiers et baissiers', value=True)
            
        # Créer le subplot avec les hauteurs relatives définies
        row_heights = [0.5]
        if show_probabilities:
            row_heights.append(0.25)
        if show_returns:
            row_heights.append(0.25)

        fig = make_subplots(rows=len(row_heights), cols=1, subplot_titles=(
            'Valeurs des Whales, x et y',
            'Probabilités de hausse' if show_probabilities else '',
            'Rendements haussiers et baissiers des Whales' if show_returns else ''),
            vertical_spacing=0.1,
            row_heights=row_heights)

        # Ajouter les traits pour w_whale (subplot 1)
        fig.add_trace(go.Scatter(
            x=labels,
            y=self.w_whale,
            mode='markers',
            name='w_whale',
            marker=dict(symbol='square', color='blue', size=8)
        ), row=1, col=1)

        # Ajouter les flèches pour x (subplot 1)
        fig.add_trace(go.Scatter(
            x=labels,
            y=x_values,
            mode='markers',
            name='x (w_whale + x)',
            marker=dict(symbol='triangle-up', color='green', size=10)
        ), row=1, col=1)

        # Ajouter les flèches pour y (subplot 1)
        fig.add_trace(go.Scatter(
            x=labels,
            y=y_values,
            mode='markers',
            name='y (w_whale - y)',
            marker=dict(symbol='triangle-down', color='red', size=10)
        ), row=1, col=1)

        current_row = 2

        # Ajouter les probabilités de hausse (subplot 2) si activé
        if show_probabilities:
            fig.add_trace(go.Bar(
                x=labels,
                y=self.phwh,
                name='Probabilités de hausse',
                width=0.3,
                marker=dict(color='purple', opacity=0.6)
            ), row=current_row, col=1)
            current_row += 1

        # Ajouter les rendements haussiers et baissiers (subplot 3) si activé
        if show_returns:
            fig.add_trace(go.Bar(
                x=labels,
                y=self.rhwh,
                name='Rendements haussiers',
                width=0.3,
                marker=dict(color='green', opacity=0.6),
                offsetgroup=0
            ), row=current_row, col=1)

            fig.add_trace(go.Bar(
                x=labels,
                y=self.rbwh,
                name='Rendements baissiers',
                width=0.3,
                marker=dict(color='red', opacity=0.6),
                offsetgroup=0
            ), row=current_row, col=1)

        # Mettre à jour la mise en page
        if show_returns and show_probabilities:
            fig.update_layout(height=800,showlegend=False)
        if (show_returns and not show_probabilities) or (not show_returns and show_probabilities):
            fig.update_layout(height=600,showlegend=False)
        if not show_returns and not show_probabilities:
            fig.update_layout(height=400,showlegend=False)

        # Afficher dans Streamlit ou dans un navigateur
        if streamlit:
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig.show()

    def get_whale_parameters_df(self):
        data = {
            'w_whale': self.w_whale,
            'x': self.x,
            'y': self.y,
            'phwh': self.phwh,
            'rhwh': self.rhwh,
            'rbwh': self.rbwh
        }
        df = pd.DataFrame(data, index=[f'Whale {i+1}' for i in range(self.n_whale)]).T
        return df
