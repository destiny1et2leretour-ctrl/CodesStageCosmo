# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:28:01 2026

@author: desti
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ==========================================
# 1. PARAMÈTRES COSMOLOGIQUES
# ==========================================
Omega_m0 = 0  #Univers d'Einstein-de Sitter
gamma = 6.0 / 11.0

# ==========================================
# 2. FONCTIONS DE LA PHYSIQUE
# ==========================================
def Omega_m(a, Om0):
    """
    Calcule l'évolution de la densité de matière au cours du temps (selon le facteur d'échelle 'a').
    """
    # Pour Om0 = 1, cette fonction renverra toujours 1 
    return (Om0 * a**-3) / (Om0 * a**-3 + (1 - Om0))

def integrand(a, Om0):
    """
    L'intégrande f/a pour le calcul de D(a). 
    On sait que f = Omega_m(a)^gamma.
    L'intégrale est int( f(a)/a ) da
    """
    f = Omega_m(a, Om0)**gamma
    return f / a

def calculate_D_num(a_target, Om0):
    """
    Calcule D(a) numériquement par intégration de scipy.quad.
    On intègre de 1 (aujourd'hui) vers a_target (le passé).
    """
    # quad renvoie la valeur de l'intégrale et l'erreur estimée (qu'on ignore avec _)
    integral_value, _ = quad(integrand, 1.0, a_target, args=(Om0,))
    
    # D(a) = exp(intégrale)
    return np.exp(integral_value)

# ==========================================
# 3. EXÉCUTION DU CALCUL NUMÉRIQUE
# ==========================================
# On crée un tableau de temps (via le facteur d'échelle a), 
# il y a très longtemps (a=10^-4) jusqu'à aujourd'hui (a=1)
a_values = np.linspace(1e-4, 1.0, 200)

# On calcule D(a) numériquement pour chaque instant
D_values = np.array([calculate_D_num(a, Omega_m0) for a in a_values])

# Calcul du terme (1 - f) qui pilote phi_point
# phi_point est proportionnel à (1 - f)
f_values = Omega_m(a_values, Omega_m0)**gamma
phi_point_prop = 1 - f_values

# ==========================================
# 4. AFFICHAGE DES RÉSULTATS POUR LE TUTEUR
# ==========================================
print("=== VÉRIFICATION ANALYTIQUE VS NUMÉRIQUE ===")
print(f"Paramètre utilisé : Omega_m0 = {Omega_m0}")
print(f"Valeur moyenne de f calculée : {np.mean(f_values):.5f}")
print(f"Terme pilotant phi_point (1 - f) : {np.mean(phi_point_prop):.5f}")
if np.allclose(phi_point_prop, 0):
    print("-> CONCLUSION : La dérivée temporelle du potentiel gravitationnel (phi_point) est bien NULLE.")

# Tracé du graphique
plt.figure(figsize=(8, 6))

# La théorie attendue : D(a) = a
plt.plot(a_values, a_values, 'r--', linewidth=2, label="Analytical theory : D(a) = a")

# Le résultat de ton algorithme
plt.plot(a_values, D_values, 'b-', alpha=0.6, linewidth=4, label="Numerical result ")

plt.xlabel("Scaling factor $a(t)$", fontsize=12)
plt.ylabel("Linear growth factor $D(a)$", fontsize=12)
plt.title(f" $\Omega_{{m0}}={Omega_m0}$", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)

# Montre le graphique
plt.show()