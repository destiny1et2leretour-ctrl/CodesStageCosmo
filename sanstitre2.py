# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:47:31 2026

@author: desti
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. PARAMÈTRES GLOBAUX (Univers LambdaCDM local)
# ==========================================
H0_km_s_Mpc = 70.0
c_km_s = 299792.458 
H0 = H0_km_s_Mpc / c_km_s

Om_m = 0.3
Om_L = 0.7

R_s = 50.0    
delta_c = 0.5 
Phi_c = -1.5 * (H0**2) * (R_s**2) * delta_c

t0 = (2.0 / (3.0 * H0 * np.sqrt(Om_L))) * np.arcsinh(np.sqrt(Om_L / Om_m))

# ==========================================
# 2. FONCTIONS DE LA PHYSIQUE
# ==========================================
def a_cosmic(t):
    coeff = (Om_m / Om_L)**(1/3)
    arg = 1.5 * H0 * np.sqrt(Om_L) * t
    return coeff * (np.sinh(arg))**(2/3)

def H_cosmic(t):
    arg = 1.5 * H0 * np.sqrt(Om_L) * t
    return H0 * np.sqrt(Om_L) / np.tanh(arg)

def Phi(r):
    if r < 1e-10:
        return Phi_c
    return Phi_c * (R_s / r) * np.arcsinh(r / R_s)

def vitesse_radiale_v(r, t):
    if r < 1e-10:
        return 0.0 
    terme_prefacteur = (2.0 * R_s * Phi_c) / (3.0 * H0 * r**2)
    # On utilise l'approximation de la croissance de l'article
    terme_temporel = (t / t0)**(1/3) 
    terme_spatial = np.arcsinh(r / R_s) - (r / np.sqrt(R_s**2 + r**2))
    return terme_prefacteur * terme_temporel * terme_spatial

# ==========================================
# 3. GÉODÉSIQUES : APPROXIMATION DE BORN
# ==========================================
def geodesic_born(lam, Y):
    t, r, theta, kt, kr, ktheta = Y
    r = max(r, 1e-10)
    
    a_t = a_cosmic(t)
    H_t = H_cosmic(t)
    phi_r = Phi(r)
    
    # Évolution de la position
    dt_dl = kt
    dr_dl = kr
    dtheta_dl = ktheta
    
    # Approximation de Born : La lumière va en ligne droite dans l'espace.
    # On supprime les dérivées spatiales du potentiel (dPhi_dr = 0)
    dkt_dl = - H_t * a_t**2 * (1 - 4*phi_r) * (kr**2 + r**2 * ktheta**2)
    
    # L'impulsion spatiale ne subit que l'expansion (Hubble drag), pas de lensing.
    dkr_dl = - 2 * H_t * kt * kr + r * ktheta**2
    dktheta_dl = - 2 * H_t * kt * ktheta - (2.0 / r) * kr * ktheta
    
    return [dt_dl, dr_dl, dtheta_dl, dkt_dl, dkr_dl, dktheta_dl]

# ==========================================
# 4. SIMULATION : LE PROFIL DU DIPÔLE
# ==========================================
r0 = 200.0  # Observateur à 200 Mpc
angles_deg = np.linspace(-90, 90, 40) # Balayage de -10° à +10°
angles_rad = np.radians(angles_deg)

delta_T_sur_T = []

print("Calcul du profil angulaire Delta T / T en cours...")

# On calcule l'énergie de base à l'observation
Phi_o = Phi(r0)
v_o = vitesse_radiale_v(r0, t0)
ut_o = 1.0 - Phi_o
ur_o = v_o / 1.0
g_tt_o = -(1.0 + 2.0*Phi_o)
g_rr_o = (1.0**2) * (1.0 - 2.0*Phi_o)

for alpha in angles_rad:
    kt_initial = -1.0
    kr_initial = -np.cos(alpha)
    Y0 = [t0, r0, 0.0, kt_initial, kr_initial, np.sin(alpha) / r0]

    E_obs = np.abs(- (g_tt_o * kt_initial * ut_o + g_rr_o * kr_initial * ur_o))

    # On intègre loin derrière l'amas pour capter tout l'effet (400 Mpc)
    sol = solve_ivp(geodesic_born, [0, 400], Y0, method='Radau', rtol=1e-9, atol=1e-9)
    
    t_f = sol.y[0][-1]
    r_f = sol.y[1][-1]
    kt_f = sol.y[3][-1]
    kr_f = sol.y[4][-1]
    
    a_f = a_cosmic(t_f)
    Phi_f = Phi(r_f)
    v_f = vitesse_radiale_v(r_f, t_f)
    
    ut_f = 1.0 - Phi_f
    ur_f = v_f / a_f
    g_tt_f = -(1.0 + 2.0*Phi_f)
    g_rr_f = (a_f**2) * (1.0 - 2.0*Phi_f)
    
    E_em = np.abs(- (g_tt_f * kt_f * ut_f + g_rr_f * kr_f * ur_f))
    
    # Redshift total
    z_total = (E_em / E_obs) - 1.0
    
    # Redshift purement cosmologique
    z_cosmo = (1.0 / a_f) - 1.0
    
    # Perturbation delta z
    delta_z = z_total - z_cosmo
    
    # Formule thermodynamique : Delta T / T = - Delta z / (1 + z_background)
    # (Signe moins car un redshift (+) refroidit le CMB (-))
    dT_T = - delta_z / (1.0 + z_cosmo)
    
    delta_T_sur_T.append(dT_T)

# ==========================================
# 5. TRACÉ DU GRAPHIQUE
# ==========================================
plt.figure(figsize=(9, 6))
plt.plot(angles_deg, delta_T_sur_T, color='purple', linewidth=3)

plt.title(r"Angular profile of the CMB perturbation : $\Delta T / T (\alpha)$", fontsize=14)
plt.xlabel(r"Viewing angle $\alpha$ (Degrees)", fontsize=12)
plt.ylabel(r"Temperature fluctuation $\Delta T / T$", fontsize=12)

# Lignes de repère
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)

plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()