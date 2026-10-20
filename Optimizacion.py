#%% Paqueterias necesarias
import matplotlib.pyplot as plt          # Para generar gráficas
import numpy as np                       # Para manejo de arreglos y operaciones matemáticas
import optuna                            # Biblioteca para optimización de hiperparámetros
from scipy.integrate import odeint       # Integrador de ecuaciones diferenciales ordinarias
import sympy as sp                        # Matemática simbólica para resolver ecuaciones

plt.style.use('seaborn-v0_8-dark')
plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

# Parámetros óptimos que se utilizaron en la tesis.
f_KLK = 0.40509243993417915
f_release = 0.02474920863730841
f_production = 3.660472590236123
k_prod = 1.530472558632681
k_dif = 1.4399311031304896
k_deg = 0.42859621128509695
parametros = (f_KLK, f_release, f_production, k_prod, k_dif, k_deg)

# Datos usados como indicadores indirectos del compartimento diferenciado.
time_data = np.array([0, 0.98, 1.98, 2.98, 5.98, 7.98, 9.98])
D_data = np.array([0.577, 0.711, 0.812, 0.845, 0.852, 0.864, 0.87])


def Modelo(y, t, f_KLK, f_release, f_production, k_prod, k_dif, k_deg):
    Dif, Bas = y
    diferenciacion = k_dif * ((1 + f_KLK * Dif) / (1 + f_release * Dif)) * Bas
    dDdt = diferenciacion - k_deg * Dif
    dBdt = k_prod / (1 + f_production * Dif) - diferenciacion
    return [dDdt, dBdt]


# Equilibrio positivo del sistema, calculado con las ecuaciones del modelo.
# k_prod = k_deg * D_eq * (1 + f_production * D_eq).
D_eq = (-1 + np.sqrt(1 + 4 * f_production * k_prod / k_deg)) / (2 * f_production)
B_eq = (k_deg * D_eq / k_dif) * (1 + f_release * D_eq) / (1 + f_KLK * D_eq)

# Condición inicial de D fijada con la primera observación; B inicia en su equilibrio.
y0 = [D_data[0], B_eq]
times_fine = np.linspace(0, 10, 100)
solution = odeint(Modelo, y0, times_fine, args=parametros)
D_sim, B_sim = solution.T

# Única figura que se utiliza en la tesis.
plt.figure(figsize=(10, 6))
plt.plot(times_fine, D_sim, color='darkseagreen', label='Células diferenciadas')
plt.plot(times_fine, B_sim, color='cornflowerblue', label='Células basales')
plt.scatter(time_data, D_data, color='slategrey', label='Datos de Hoffman (2015)')
plt.axhline(B_eq, color='slategrey', linestyle='--', alpha=0.5)
plt.axhline(D_eq, color='slategrey', linestyle='--', alpha=0.5)
plt.title('Modelo optimizado')
plt.xlabel('Tiempo (días)')
plt.ylabel('Concentración celular')
plt.legend()
plt.grid(True)

# Guarda la figura en la carpeta de trabajo actual.
plt.savefig('Optimizacion.png', dpi=600, bbox_inches='tight')
plt.show()
plt.close()

# %%
