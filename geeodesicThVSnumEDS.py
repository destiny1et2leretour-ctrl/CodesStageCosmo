# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:27:16 2026

@author: desti
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# PARAMÈTRES COSMOLOGIQUES+AMAS (c = 1)
# ==========================================
Om_m = 1  
Om_L = 0    
H0_km_s_Mpc = 70.0
c_km_s = 299792.458 
H0 = H0_km_s_Mpc / c_km_s

R_s = 50.0

delta_c = 0
# ==========================================
# FONCTIONS ANALYTIQUES 
# 
# ==========================================
def t_analytique(lam, t0):
    return t0 * (1 - (5 * lam) / (3 * t0))**(3/5)

def r_analytique(lam, t0, r0, alpha):
    terme_crochet = (1 - (5 * lam) / (3 * t0))**(1/5) - 1
    interieur_racine = (1 + 6 * np.cos(alpha) * (t0 / r0) * terme_crochet 
                        + 9 * (t0**2 / r0**2) * terme_crochet**2)
    return r0 * np.sqrt(interieur_racine)

def theta_analytique(lam, t0, r0, alpha):
    # RUSTINE : Sécurité pour le tir tout droit (alpha = 0)
    if alpha == 0.0:
        return np.zeros_like(lam) 
        
    terme_crochet = (1 - (5 * lam) / (3 * t0))**(1/5) - 1
    # On utilise 1/tan pour la cotangente (cot) et 1/sin pour la cosécante (csc)
    partie_droite = (1 / np.tan(alpha)) + 3 * (t0 / r0) * terme_crochet * (1 / np.sin(alpha))
    
    # arctan donne l'angle, arctan(cot(alpha)) est l'angle de départ de l'article
    angle_initial = np.arctan(1 / np.tan(alpha))
    return angle_initial - np.arctan(partie_droite)

# ==========================================
# FONCTIONS ΛCDM 
# ==========================================
def age_universe_today():
    # Sécurité globale au cas où tu réactives la fonction
    if Om_L == 0.0:
        return 2.0 / (3.0 * H0)
    arg = np.sqrt(Om_L / Om_m)
    return (2.0 / (3.0 * H0 * np.sqrt(Om_L))) * np.arcsinh(arg)

def a_cosmic(t):
    # Si on est dans le test de validation (sans énergie sombre)
    if Om_L == 0.0:
        return (1.5 * H0 * t)**(2/3)
    # Sinon, on utilise ta formule d'origine pour le vrai Univers
    else:
        coeff = (Om_m / Om_L)**(1/3)
        arg = 1.5 * H0 * np.sqrt(Om_L) * t
        return coeff * (np.sinh(arg))**(2/3) # <--- La ligne manquante est de retour !

def H_cosmic(t):
    # RUSTINE : Si on est dans le test de validation
    if Om_L == 0.0:
        return 2.0 / (3.0 * t)
    # Sinon, vrai Univers
    else:
        arg = 1.5 * H0 * np.sqrt(Om_L) * t
        return H0 * np.sqrt(Om_L) / np.tanh(arg)

# ==========================================
# POTENTIEL GRAVITATIONNEL DE L'AMAS
# ==========================================
def Phi(r):
    """ Potentiel newtonien comobile Phi(r) """
    if r <= R_s:
        return 0.25 * H0**2 * Om_m * delta_c * (r**2 - 3 * R_s**2)
    else:
        return -0.5 * H0**2 * Om_m * delta_c * (R_s**3 / r)

def dPhi_dr(r):
    """ Dérivée spatiale Phi'(r) (Force d'attraction) """
    if r <= R_s:
        return 0.5 * H0**2 * Om_m * delta_c * r
    else:
        return 0.5 * H0**2 * Om_m * delta_c * (R_s**3 / r**2)

# ==========================================
# GÉODÉSIQUES PERTURBÉES
# ==========================================
def geodesic_equations(lam, Y):
    t, r, theta, kt, kr, ktheta = Y
    
    # Sécurité numérique pour éviter la division par zéro au centre exact (r=0)
    r = max(r, 1e-10)
    
    a_t = a_cosmic(t)
    H_t = H_cosmic(t)
    
    phi_r = Phi(r)
    dphi_r = dPhi_dr(r)
    
    # 1. Vitesses - défini ce quest k^nu pour nu={t,r,theta}
    dt_dl = kt
    dr_dl = kr
    dtheta_dl = ktheta
    
    # 2. Les equations des geodesics obtenus via calculs  
    dkt_dl = - H_t * a_t**2 * (1 - 4*phi_r) * (kr**2 + r**2 * ktheta**2) - 2 * dphi_r * kt * kr
    dkr_dl = - 2 * H_t * kt * kr + r * ktheta**2 - (1 / a_t**2) * dphi_r * kt**2 + dphi_r * kr**2 - r**2 * dphi_r * ktheta**2
    dktheta_dl = - 2 * H_t * kt * ktheta - (2.0 / r) * kr * ktheta + 2 * dphi_r * kr * ktheta
    
    return [dt_dl, dr_dl, dtheta_dl, dkt_dl, dkr_dl, dktheta_dl]

# Arrêt au CMB (z=1089)
def reach_cmb(lam, Y):
    t = Y[0]
    return a_cosmic(t) - (1.0 / 1100)
reach_cmb.terminal = True 

# ==========================================
# SIMULATION
# ==========================================
#t0 = age_universe_today() 
t0= 2/(3*H0) 
#t0= (2.0 / (3.0 * H0 * np.sqrt(Om_L))) * np.arcsinh(np.sqrt(Om_L / Om_m))   
r0 = 200.0                

angles_alpha = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
trajectories = []

for alpha in angles_alpha:
    # Conditions initiales pour g_uv k^u k^v = 0
    Y0 = [t0, r0, 0.0, -1.0, -np.cos(alpha), np.sin(alpha) / r0]

    sol = solve_ivp(geodesic_equations, [0, 20000], Y0, method='Radau', 
                    events=reach_cmb, max_step=10, dense_output=True)
    
    
    # Extraction des données NUMÉRIQUES de ton intégrateur (BIEN INDENTÉ)
    lam_vals = sol.t        # Le paramètre affine (axe X du graphique)
    t_num = sol.y[0]        # Temps cosmique numérique
    r_num = sol.y[1]        # Distance radiale numérique
    theta_num = sol.y[2]    # Angle de position numérique

    # Calcul des données THÉORIQUES pour les mêmes valeurs de lambda
    t_th = t_analytique(lam_vals, t0)
    r_th = r_analytique(lam_vals, t0, r0, alpha)
    theta_th = theta_analytique(lam_vals, t0, r0, alpha)

    # ==========================================
    # CONVERSION DES UNITÉS POUR L'AFFICHAGE
    # ==========================================
    # Facteur pour passer d'un temps en Mpc à un temps en Giga-années (Gyr)
    mpc_to_gyr = 3.26156377 / 1000.0

    # ==========================================
    # AFFICHAGE DE LA VALIDATION (À l'intérieur de la boucle)
    # ==========================================
    fig_val, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig_val.suptitle(f"EdS validation for alpha = {np.degrees(alpha):.1f}°", fontsize=16)

# Graph 1: Cosmic time t(lambda)
    ax1.plot(lam_vals, t_num * mpc_to_gyr, 'b-', linewidth=4, label='Numerical')
    ax1.plot(lam_vals, t_th * mpc_to_gyr, 'r--', linewidth=2, label='Analytical ')
    ax1.set_title("Cosmic time evolution t(λ)")
    ax1.set_xlabel("Affine parameter λ (Mpc)")
    ax1.set_ylabel("Time t (Gyr)")
    ax1.legend()
    ax1.grid(True)

# Graph 2: Radial distance r(lambda)
    ax2.plot(lam_vals, r_num, 'b-', linewidth=4, label='Numerical')
    ax2.plot(lam_vals, r_th, 'r--', linewidth=2, label='Analytical ')
    ax2.set_title("Radial distance evolution r(λ)")
    ax2.set_xlabel("Affine parameter λ (Mpc)")
    ax2.set_ylabel("Distance r (Mpc)")
    ax2.legend()
    ax2.grid(True)

# Graph 3: Angle theta(lambda)
    ax3.plot(lam_vals, theta_num, 'b-', linewidth=4, label='Numerical')
    ax3.plot(lam_vals, theta_th, 'r--', linewidth=2, label='Analytical ')
    ax3.set_title("Angular evolution θ(λ)")
    ax3.set_xlabel("Affine parameter λ (Mpc)")
    ax3.set_ylabel("Angle θ (rad)")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
    
    # Suite des calculs pour le graphique final de l'amas
    r_vals = sol.y[1]
    theta_vals = sol.y[2]
    r_emission = r_vals[-1]
    theta_emission = theta_vals[-1]

    print(f"Le photon observé sous l'angle {alpha} a été émis à r={r_emission:.2f} Mpc et theta={theta_emission:.4f} rad")
    
    # Conversion purement graphique pour matplotlib
    x_vals = r_vals * np.cos(theta_vals)
    y_vals = r_vals * np.sin(theta_vals)
    
    trajectories.append((alpha, x_vals, y_vals))

# ==========================================
# AFFICHAGE FINAL (En dehors de la boucle)
# ==========================================
fig_final, ax = plt.subplots(figsize=(9, 8)) 
fig_final.suptitle("Local Trajectories of Geodesics (Perturbed Metric)", fontsize=16)

# Tracé des géodésiques
for alpha, x_vals, y_vals in trajectories:
    ax.plot(x_vals, y_vals, label=f"$\\alpha = {np.degrees(alpha):.0f}^\circ$")

# Tracé des éléments physiques
ax.plot(0, 0, 'ko', markersize=6, label="Center of the cluster")
ax.plot(r0, 0, 'rx', markersize=8, markeredgewidth=2, label="Observer (Telescope)")
amas_circle = plt.Circle((0, 0), R_s, color='gray', fill=True, alpha=0.3, label=f"Cluster ($R_s = {R_s}$ Mpc)")
ax.add_patch(amas_circle)

# Esthétique et limites du graphique (Zoom)
ax.set_xlabel("Co-mobile coordinates X (Mpc)", fontsize=12)
ax.set_ylabel("Co-mobile coordinates Y (Mpc)", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)

ax.set_xlim(-100, 300)
ax.set_ylim(-50, 350)

# Placement de la légende
ax.legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.show()



'''
##Pour Our current univers
# 1. Calcul du facteur d'échelle a(t) pour le modèle Lambda-CDM
# (np.sinh est la fonction sinus hyperbolique de numpy)
terme_constant = (Om_m / Om_L)**(1/3)
terme_temporel = np.sinh(1.5 * H0 * np.sqrt(Om_L) * t_num)**(2/3)

a_num_lcdm = terme_constant * terme_temporel

# 2. Calcul du redshift
z_num_lcdm = (1.0 / a_num_lcdm) - 1

# ==========================================
# TRACÉ DU GRAPHIQUE Z en fonction de Lambda
# ==========================================

plt.figure(figsize=(8, 5))

# On trace la courbe numérique
plt.plot(lam_vals, z_num_lcdm, 'g-', linewidth=3, label=r"Model $\Lambda$CDM ($\Omega_m=0.3, \Omega_\Lambda=0.7$)")

plt.title("Evolution of Redshift $z$ in a Universe with Dark Energy")
plt.xlabel(r"Affine parameter $\lambda$ (Mpc)")
plt.ylabel("Redshift $z$")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# On garde l'échelle logarithmique car z explose toujours à la fin !
plt.yscale('log')

plt.tight_layout()
plt.show()
# Affichage de la valeur finale pour le prof
'''
'''
z_final_obtenu = z_num_lcdm[-1]
print(f"Le redshift au temps final de la simulation (sur le CMB) est : {z_final_obtenu:.2f}")
'''
## Pour EDS 
# # 1. Calcul du Redshift THEORIQUE
z_th = (1 - (5 * lam_vals) / (3 * t0))**(-2/5) - 1

# 2. Calcul du Redshift NUMERIQUE (à partir de ton t_num)
# On utilise ta formule a(t) = (t / t0)**(2/3)
a_num = (t_num / t0)**(2/3) 
z_num = (1.0 / a_num) - 1

# ==========================================
# TRACÉ DU GRAPHIQUE Z en fonction de Lambda
# ==========================================
plt.figure(figsize=(8, 5))

# Plot the numerical integration in thick blue
plt.plot(lam_vals, z_num, 'b-', linewidth=3, label="Numerical Integration")

# Plot the analytical solution in dashed red on top
plt.plot(lam_vals, z_th, 'r--', linewidth=2, label="Analytical (EdS Universe)")

plt.title("Evolution of Redshift $z$ along the photon trajectory")
plt.xlabel(r"Affine parameter $\lambda$ (Mpc)")
plt.ylabel("Redshift $z$")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.yscale('log')
plt.show()


