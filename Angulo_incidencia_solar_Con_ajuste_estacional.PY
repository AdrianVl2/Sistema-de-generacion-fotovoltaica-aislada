import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --------- 0. Definir carpeta de salida ---------
output_dir = r"D:\Datos_python"
os.makedirs(output_dir, exist_ok=True)

# --------- 1. Cargar datos ---------
data = pd.read_excel("irradiacionriv.xlsx")

if 'DOY' not in data.columns or 'AS' not in data.columns or 'GS' not in data.columns or 'CZ' not in data.columns:
    raise ValueError("El archivo debe contener las columnas: 'DOY', 'AS' (altura solar), 'GS' (acimut solar), 'CZ'")

# --------- 2. Función de inclinación estacional ---------
def inclinacion_estacional(doy):
    if doy < 56:
        return 10.5
    elif doy < 98:
        return 31
    elif doy < 261:
        return 54.5
    elif doy < 292:
        return 31
    else:
        return 10.5

# --------- 3. Cálculo de ángulos de incidencia ---------
theta_z_rad = np.radians(90 - data['AS'])  # Ángulo cenital
az_sol_rad = np.radians(data['GS'])        # Acimut solar en rad

# Estacional
cos_theta_est = []
inclinaciones_usadas = []

for i, row in data.iterrows():
    doy = row['DOY']
    beta_deg = inclinacion_estacional(doy)
    inclinaciones_usadas.append(beta_deg)
    beta_rad = np.radians(beta_deg)
    phi_rad = az_sol_rad.iloc[i]

    cos_t = (np.cos(beta_rad) * np.cos(theta_z_rad.iloc[i]) +
             np.sin(beta_rad) * np.sin(theta_z_rad.iloc[i]) * np.cos(phi_rad))

    cos_theta_est.append(np.nan if row['CZ'] <= 0 else np.clip(cos_t, -1, 1))

angulo_total_estacional_rad = np.arccos(cos_theta_est)
angulo_total_estacional_deg = np.degrees(angulo_total_estacional_rad)

# Fijo (inclinación constante 31°)
cos_theta_fijo = []
beta_fija_rad = np.radians(31)

for i, row in data.iterrows():
    phi_rad = az_sol_rad.iloc[i]
    cos_t_fijo = (np.cos(beta_fija_rad) * np.cos(theta_z_rad.iloc[i]) +
                  np.sin(beta_fija_rad) * np.sin(theta_z_rad.iloc[i]) * np.cos(phi_rad))

    cos_theta_fijo.append(np.nan if row['CZ'] <= 0 else np.clip(cos_t_fijo, -1, 1))

angulo_total_fijo_rad = np.arccos(cos_theta_fijo)
angulo_total_fijo_deg = np.degrees(angulo_total_fijo_rad)

# --------- 4. Remover valores específicos (ruido nocturno) ---------
valores_a_remover = [(59, 0.02), (77.45, 0.01), (35.55, 0.01)]
for valor, tol in valores_a_remover:
    angulo_total_estacional_deg = np.where(np.isclose(angulo_total_estacional_deg, valor, atol=tol), np.nan, angulo_total_estacional_deg)
    angulo_total_fijo_deg = np.where(np.isclose(angulo_total_fijo_deg, valor, atol=tol), np.nan, angulo_total_fijo_deg)

# --------- 5. Componente altura solar (AS - 90 + inclinación) ---------
angulo_altura_estacional = (data['AS'] - 90) + inclinaciones_usadas

# --------- 6. Guardar resultados ---------
df_resultado = pd.DataFrame({
    'Hora': range(1, len(data)+1),
    'DOY': data['DOY'],
    'Angulo_Incidencia_Altura_Solar_Estacional': angulo_altura_estacional,
    'Inclinacion_Aplicada': inclinaciones_usadas,
    'Angulo_Incidencia_Total_Combinado_Estacional': angulo_total_estacional_deg,
    'Angulo_Incidencia_Total_Combinado_Fijo_31': angulo_total_fijo_deg
})

df_resultado.to_csv(os.path.join(output_dir, 'Angulo_de_Incidencia_Sistema_Estacional.csv'), index=False)

# --------- 7. Cargar datos de comparación de altura solar ---------
df_fija = pd.read_csv('Angulo_Incidencia_Altura_Solar.csv')
horas = df_fija['Hora']
valores_fijos_altura = df_fija['Angulo_Incidencia_Altura_Solar']
valores_estacionales_altura = df_resultado['Angulo_Incidencia_Altura_Solar_Estacional']

# --------- 8. Gráfico: Fija vs Estacional (altura solar) ---------
plt.figure(figsize=(14, 6))
plt.plot(horas, valores_estacionales_altura, label='Inclinación Estacional (Altura)', color='blue', linewidth=1.5)
plt.plot(horas, valores_fijos_altura, label='Inclinación Fija -31° (Altura)', color='purple', linestyle='-', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', linewidth=1)

# Etiquetas de meses
horas_por_dia = 24
dias_por_mes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
         'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
posiciones_meses = []
acumulador = 0
for dias in dias_por_mes:
    posiciones_meses.append(acumulador * horas_por_dia)
    acumulador += dias

plt.xticks(posiciones_meses, meses, rotation=45)
plt.xlabel("Meses del Año")
plt.ylabel("Ángulo de Incidencia (°)")
plt.title("Comparación por Altura Solar\nSistema Fijo (31°) vs Estacional")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# --------- 9. Gráfico: Comparación Ángulo Total (altura + acimut) ---------
plt.figure(figsize=(14, 6))
plt.plot(horas, df_resultado['Angulo_Incidencia_Total_Combinado_Estacional'],
         label='Angulo de Incidencia Estacional ', color='green', linewidth=2, )
plt.plot(horas, df_resultado['Angulo_Incidencia_Total_Combinado_Fijo_31'],
         label='Angulo de Incidencia Fijo 31° ', color='red', linestyle='-', alpha=1, linewidth=0.7)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xticks(posiciones_meses, meses, rotation=45)
plt.xlabel("Meses del Año")
plt.ylabel("Ángulo de Incidencia Total (°)")
plt.title("Comparación del Ángulo Total de Incidencia\nSistema Fijo (31°) vs Estacional ")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
