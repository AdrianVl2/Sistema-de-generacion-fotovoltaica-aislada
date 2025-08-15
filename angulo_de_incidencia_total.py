import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar datos
file_path = "irradiacionriv.xlsx"
data = pd.read_excel(file_path).reset_index(drop=True)

# Calcular ángulo cenital y acimut solar en radianes
data['theta_z'] = np.arccos(np.clip(data['CZ'], -1, 1))
data['az'] = np.radians(data['GS'])

# Inclinación fija 31°
beta_deg = 31
beta_rad = np.radians(beta_deg)

# Calcular ángulo de incidencia
theta = np.arccos(
    np.cos(beta_rad) * np.cos(data['theta_z']) +
    np.sin(beta_rad) * np.sin(data['theta_z']) * np.cos(data['az'])
)

# Convertir a grados
angulo_inc_31 = np.degrees(theta)

# Reemplazar valores cercanos a 59° por NaN para no graficar esos puntos
angulo_inc_31_plot = angulo_inc_31.copy()
angulo_inc_31_plot[np.isclose(angulo_inc_31_plot, 59, atol=0.01)] = np.nan

# Graficar
plt.figure(figsize=(14, 4))
plt.plot(angulo_inc_31_plot, color='blue', linewidth=0.7, label='Angulo de Incidencia')
plt.title('Angulo de Incidencia Solar para Inclinacion Fija de 31° y Acimut 0°')
plt.axhline(90, color='red', linestyle='--', linewidth=0.8, label='90°')
plt.xlabel('Hora del año')
plt.ylabel('Angulo de incidencia (°)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
