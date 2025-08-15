import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Parámetros ---
file_path = "irradiacionriv.xlsx"
gamma_p = 0  # orientación del panel en grados (negativo = oeste, positivo = este)

# --- Cargar datos ---
data = pd.read_excel(file_path)

# --- Función: diferencia angular con signo en grados ---
def angulo_diferencia_signed(gs, gamma_p):
    diff = gs - gamma_p
    if pd.isna(diff):
        return np.nan
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff

# --- Calcular incidencia angular hora a hora ---
incidencias = data['GS'].apply(lambda gs: angulo_diferencia_signed(gs, gamma_p))

# --- Guardar resultados ---
df_resultados = pd.DataFrame({
    'Hora': range(1, len(incidencias) + 1),
    'Incidencia_Angulo': incidencias
})
df_resultados.to_csv('angulo_incidencia_acimut.csv', index=False)

# --- Graficar en función de la hora del año ---
plt.figure(figsize=(14, 6))
plt.plot(df_resultados['Hora'], df_resultados['Incidencia_Angulo'],
         color='green', linewidth=0.7, label="Ángulo de Incidencia (con signo)")

plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel("Hora del Año")
plt.ylabel("Ángulo de Incidencia (°)")
plt.title(f"Ángulo de Incidencia Solar según Acimut (γₚ = {gamma_p}°)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
