#%% Paqueterías
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp

plt.style.use('seaborn-v0_8-dark')
plt.style.use('seaborn-v0_8-dark')
plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

#%% SISTEMA
def Modelo(y, t, f_KLK, f_release, f_production, k_prod, k_dif, k_deg):
    Dif, Bas = y
    factor = (1 + f_KLK * Dif) / (1 + f_release * Dif)

    dDdt = k_dif * factor * Bas - k_deg * Dif
    dBdt = k_prod / (1 + f_production * Dif) - k_dif * factor * Bas

    return [dDdt, dBdt]

#%% EQUILIBRIO (más robusto)
def Equilibrio_pos(f_KLK, f_release, f_production, k_prod, k_dif, k_deg):
    Dif, Bas = sp.symbols('Dif Bas')

    eq1 = sp.Eq(k_dif * ((1 + f_KLK * Dif) / (1 + f_release * Dif)) * Bas - k_deg * Dif, 0)
    eq2 = sp.Eq(k_prod / (1 + f_production * Dif) - k_dif * ((1 + f_KLK * Dif) / (1 + f_release * Dif)) * Bas, 0)

    soluciones = sp.solve((eq1, eq2), (Dif, Bas), dict=True)

    for sol in soluciones:
        Dif_val = float(sol[Dif])
        Bas_val = float(sol[Bas])

        if Dif_val > 0 and Bas_val > 0:
            return [Dif_val, Bas_val]

    return None  # si no hay solución positiva

#%% PARÁMETROS
params_best = {
    "f_{KLK}": 0.40509243993417915,
    "f_{release}": 0.02474920863730841,
    "f_{production}": 3.660472590236123,
    "k_{prod}": 1.530472558632681,
    "k_{dif}": 1.4399311031304896,
    "k_{deg}": 0.42859621128509695
}

#%%
# Variación
delta = 0.9

def rango(valor):
    return valor * (1 - delta), valor * (1 + delta)

#%% CONDICIONES INICIALES
y0 = [0.577, 0.194]
times = np.linspace(0, 35, 100)

# Solución base
solution_base = odeint(
    Modelo, y0, times,
    args=tuple(params_best.values())
)
D_base, B_base = solution_base.T
# Punto de equilibrio del modelo sano
eq_base = Equilibrio_pos(*params_best.values())
#%% FUNCIÓN GENERAL PARA GRAFICAR
def graficar_variacion(param_name):
    p_min, p_max = rango(params_best[param_name])

    # Copias de parámetros
    params_min = params_best.copy()
    params_max = params_best.copy()

    params_min[param_name] = p_min
    params_max[param_name] = p_max

    # Simulaciones
    sol_min = odeint(Modelo, y0, times, args=tuple(params_min.values()))
    sol_max = odeint(Modelo, y0, times, args=tuple(params_max.values()))

    D_min, B_min = sol_min.T
    D_max, B_max = sol_max.T

    # Calcular los puntos de equilibrio para mínimo y máximo
    eq_min = Equilibrio_pos(*params_min.values())
    eq_max = Equilibrio_pos(*params_max.values())

    # Gráfica
    fig, ax = plt.subplots(figsize=(10, 6))

    # Curvas temporales
    # Caso mínimo (línea continua con transparencia)
    ax.plot(times, D_min, color='darkseagreen', linewidth=1)
    ax.plot(times, B_min, color='cornflowerblue', linewidth=1)

    # Caso base (línea sólida, más gruesa)
    ax.plot(times, D_base, color='darkseagreen', label='D(t)', linewidth=2)
    ax.plot(times, B_base, color='cornflowerblue', label='B(t)', linewidth=2)

    # Caso máximo (línea continua con transparencia)
    ax.plot(times, D_max, color='darkseagreen', linewidth=3)
    ax.plot(times, B_max, color='cornflowerblue', linewidth=3)

    # --- Líneas horizontales de equilibrio (gris punteado) ---
    # Equilibrio del modelo sano (parámetros óptimos)
    if eq_base is not None:
        # Líneas horizontales grises
        ax.axhline(eq_base[0],  color='gray', linestyle='--', alpha=0.8, linewidth=1)
        ax.axhline(eq_base[1], color='gray', linestyle='--', alpha=0.8, linewidth=1)
        # Anotaciones en el borde izquierdo
        ax.text(0.93, eq_base[0], f'D*={eq_base[0]:.3f}', transform=ax.get_yaxis_transform(),
                color='gray', fontsize=9, va='center', ha='left', bbox=dict(boxstyle='round,pad=0.2', 
                facecolor='white', alpha=0.8))

        ax.text(0.93, eq_base[1], f'B*={eq_base[1]:.3f}',transform=ax.get_yaxis_transform(),
                color='gray', fontsize=9, va='center', ha='left', bbox=dict(boxstyle='round,pad=0.2', 
                facecolor='white', alpha=0.8))
    if eq_min is not None:
        # Líneas horizontales grises
        ax.axhline(eq_min[0], color='gray', linestyle='--', alpha=0.8, linewidth=1)
        ax.axhline(eq_min[1], color='gray', linestyle='--', alpha=0.8, linewidth=1)
        # Anotaciones en el borde izquierdo
        ax.text(0.93, eq_min[0], f'D={eq_min[0]:.3f}', transform=ax.get_yaxis_transform(),
                color='gray', fontsize=9, va='center', ha='left', bbox=dict(boxstyle='round,pad=0.2', 
                facecolor='white', alpha=0.8))
        ax.text(0.93, eq_min[1], f'B={eq_min[1]:.3f}', transform=ax.get_yaxis_transform(),
                color='gray', fontsize=9, va='center', ha='left', bbox=dict(boxstyle='round,pad=0.2', 
                facecolor='white', alpha=0.8))

    if eq_max is not None:
        ax.axhline(eq_max[0], color='gray', linestyle='--', alpha=0.8, linewidth=1)
        ax.axhline(eq_max[1], color='gray', linestyle='--', alpha=0.8, linewidth=1)
        ax.text(0.93, eq_max[0], f'D={eq_max[0]:.3f}', transform=ax.get_yaxis_transform(),
                color='gray', fontsize=9, va='center', ha='left', bbox=dict(boxstyle='round,pad=0.2', 
                facecolor='white', alpha=0.8))
        ax.text(0.93, eq_max[1], f'B={eq_max[1]:.3f}', transform=ax.get_yaxis_transform(),
                color='gray', fontsize=9, va='center', ha='left', bbox=dict(boxstyle='round,pad=0.2', 
                facecolor='white', alpha=0.8))

    ax.set_title(f'Dinámica celular variando ${param_name}$')
    ax.set_xlabel('Tiempo (días)')
    ax.set_ylabel('Concentración')
    ax.legend()
    ax.grid(True)
    # Ajustar límite x para que se vea el texto 
    #ax.set_xlim(0, 13)  # un poco más allá para que quede espacio
    print(f"\nParámetro: {param_name}")
    print(f"  Mínimo : {p_min:.4f}")
    print(f"  Base   : {params_best[param_name]:.4f}")
    print(f"  Máximo : {p_max:.4f}")

    # Mostrar la gráfica sin guardarla
    plt.show()
#%% GENERAR TODAS LAS GRÁFICAS
for param in params_best.keys():
    graficar_variacion(param)
# %%