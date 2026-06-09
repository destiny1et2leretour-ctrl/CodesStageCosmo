# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:49:37 2026

@author: desti
"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# PARAMÈTRES COSMOLOGIQUES+AMAS (c = 1)
# ==========================================
Om_m = 0.3        
Om_L = 0.7        
H0_km_s_Mpc = 70.0  
c_km_s = 299792.458 
H0 = H0_km_s_Mpc / c_km_s

R_s = 50.0

delta_c = 0.5 

# ==========================================
# FONCTIONS ΛCDM 
# ==========================================
def age_universe_today():
    arg = np.sqrt(Om_L / Om_m)
    return (2.0 / (3.0 * H0 * np.sqrt(Om_L))) * np.arcsinh(arg)

def a_cosmic(t):
    coeff = (Om_m / Om_L)**(1/3)
    arg = 1.5 * H0 * np.sqrt(Om_L) * t
    return coeff * (np.sinh(arg))**(2/3)

def H_cosmic(t):
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
    return a_cosmic(t) - (1.0 / 1100.0)
reach_cmb.terminal = True 

# ==========================================
# SIMULATION
# ==========================================
t0 = age_universe_today() 
r0 = 200.0                

angles_alpha = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
trajectories = []

for alpha in angles_alpha:
    # Conditions initiales pour g_uv k^u k^v = 0
    Y0 = [
        t0,                     # t(0)
        r0,                     # r(0)
        0.0,                    # theta(0)
        -1.0,                   # k^t bc on va du télescope vers le fond diffus cosmologique
        -np.cos(alpha),         # k^r
        np.sin(alpha) / r0      # k^theta where we looking at with the respct of g_uv k^u k^v = 0
    ]
    
    sol = solve_ivp(geodesic_equations, [0, 20000], Y0, method='Radau', 
                    events=reach_cmb, max_step=10, dense_output=True)
    # On récupère l'histoire complète (les listes de valeurs pour chaque pas de calcul)
    t_vals = sol.y[0]
    r_vals = sol.y[1]
    theta_vals = sol.y[2]
    kt_vals = sol.y[3]
    kr_vals = sol.y[4]
    ktheta_vals = sol.y[5]
    print(f"\n==========================================")
    print(f"RESULTATS POUR ALPHA = {np.degrees(alpha):.1f}°")
    print(f"==========================================")
    print(f"--- Position d'émission (au CMB) ---")
    print(f"t (temps cosmique)      = {t_vals[-1]:.4f}")
    print(f"r (distance comobile)   = {r_vals[-1]:.2f} Mpc")
    print(f"theta (angle comobile)  = {theta_vals[-1]:.4f} rad")

    print(f"\n--- Quadrivecteur vitesse à l'émission ---")
    print(f"k^t     = {kt_vals[-1]:.4f}")
    print(f"k^r     = {kr_vals[-1]:.4f}")
    print(f"k^theta = {ktheta_vals[-1]:.4f}")
    
    r_vals = sol.y[1]
    theta_vals = sol.y[2]
    r_emission = r_vals[-1]
    theta_emission = theta_vals[-1]

    print(f"Le photon observé sous l'angle {alpha} a été émis à r={r_emission:.2f} Mpc et theta={theta_emission:.4f} rad")
    
    # Conversion purement graphique pour matplotlib n'a acun impact physiquement parlant
    x_vals = r_vals * np.cos(theta_vals)
    y_vals = r_vals * np.sin(theta_vals)
    
    trajectories.append((alpha, x_vals, y_vals))
    

# ==========================================
# AFFICHAGE 
# ==========================================
fig, ax = plt.subplots(figsize=(9, 8)) 
fig.suptitle("Local Trajectories of Geodesics (Perturbed Metric)", fontsize=16)

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