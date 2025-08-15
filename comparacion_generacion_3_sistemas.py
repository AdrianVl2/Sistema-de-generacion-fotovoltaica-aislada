import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Cargar datos desde el archivo Excel ---
file_path = "irradiacionriv.xlsx"
data = pd.read_excel(file_path)

# --- Calcular ángulo cenital y acimut solar en radianes ---
data['theta_z'] = np.arccos(np.clip(data['CZ'], -1, 1))  # Ángulo cenital en radianes
data['az'] = np.radians(data['GS'])  # Acimut solar en radianes

# --- Cálculo del ángulo de incidencia para el sistema fijo ---
gamma_fijo_rad = np.radians(31)  # Inclinación fija de 31°
angulo_incidencia_fijo = np.arccos(
    np.cos(gamma_fijo_rad) * np.cos(data['theta_z']) +
    np.sin(gamma_fijo_rad) * np.sin(data['theta_z']) * np.cos(data['az'] - 0)
)

# --- Cálculo del ángulo de incidencia para el sistema estacional ---
angulo_incidencia_estacional = []

for index, row in data.iterrows():
    doy = row['DOY']
    theta_z = row['theta_z']
    az = row['az']

    # Asignar inclinación estacional según el día del año
    if doy < 56:
        beta_est_deg = 12.55
    elif doy < 111:
        beta_est_deg = 31
    elif doy < 235:
        beta_est_deg = 54.55
    elif doy < 295:
        beta_est_deg = 31
    else:
        beta_est_deg = 12.55

    beta_est_rad = np.radians(beta_est_deg)

    # Calcular el ángulo de incidencia
    cos_theta = (
        np.cos(beta_est_rad) * np.cos(theta_z) +
        np.sin(beta_est_rad) * np.sin(theta_z) * np.cos(az - 0)
    )
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    angulo_incidencia_estacional.append(theta)

theta_fijo_rad = angulo_incidencia_fijo
theta_est_rad = np.array(angulo_incidencia_estacional)

# --- Parámetros del sistema ---
E_p_fabricante = 22.5      # Eficiencia nominal del panel (%)
Area_panel = 2             # Área del panel (m²)
gamma = -0.0045            # Coef. pérdida por temperatura (%/°C)
rho = 0.2                  # Reflectividad del suelo (valor típico)

# --- Radiación ---
GHI = data['GHI']
CZ = data['CZ']
DNI = data['DNI']
DHI = GHI - DNI * CZ  #estimacion de radiacion difusa


# --- Temperatura de la celda ---
temp_data = pd.read_csv(r"D:\Datos_python\generacion_esperada.csv")
T_celda = temp_data['Temperatura_Celda_C']

# --- Corrección por ángulo ---
GHI_corr_est = DNI * np.clip(np.cos(theta_est_rad), 0, None)
GHI_corr_fijo = DNI * np.clip(np.cos(theta_fijo_rad), 0, None)

# --- Corrección por temperatura ---
eficiencia_corr = E_p_fabricante * (1 + gamma * (T_celda - 25))
eficiencia_corr = eficiencia_corr.clip(lower=0) / 100

# --- Generación hora a hora ---
gen_optima = (DNI + DHI) * Area_panel * eficiencia_corr
gen_estacional = (GHI_corr_est + DHI) * Area_panel * eficiencia_corr
gen_fijo = (GHI_corr_fijo + DHI) * Area_panel * eficiencia_corr

# --- Agrupar por día ---
dias = np.arange(len(GHI)) // 24
gen_optima_diaria = pd.Series(gen_optima).groupby(dias).sum()
gen_estacional_diaria = pd.Series(gen_estacional).groupby(dias).sum()
gen_fijo_diaria = pd.Series(gen_fijo).groupby(dias).sum()

# --- Promedios diarios ---
print(f"Promedio diario - Sistema con seguidor: {gen_optima_diaria.mean():.2f} Wh")
print(f"Promedio diario - Sistema estacional:   {gen_estacional_diaria.mean():.2f} Wh")
print(f"Promedio diario - Sistema fijo:         {gen_fijo_diaria.mean():.2f} Wh")

# --- Gráfica: Generación hora a hora ---
plt.figure(figsize=(14, 5))
plt.plot(gen_optima, label='Generación Sistema con Seguidor', color='green', linewidth=0.6)
plt.plot(gen_fijo, label='Generación Sistema Fijo', color='blue', linewidth=0.6)
plt.plot(gen_estacional, label='Generación Sistema Estacional', color='orange', linewidth=0.6)
plt.title('Generación Fotovoltaica Hora a Hora (con pérdidas por temperatura)')
plt.xlabel('Horas')
plt.ylabel('Generación (W)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# --- Gráfica: Generación diaria acumulada ---
plt.figure(figsize=(14, 5))
plt.plot(gen_optima_diaria, label='Sistema con Seguidor', color='green')
plt.plot(gen_estacional_diaria, label='Sistema Estacional', color='orange')
plt.plot(gen_fijo_diaria, label='Sistema Fijo', color='blue')
plt.title('Generación Promedio Diaria')
plt.xlabel('Día del Año')
plt.ylabel('Energía diaria (Wh)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# --- Promedios por semana ---
semanas = gen_optima_diaria.index // 7
gen_optima_semanal = gen_optima_diaria.groupby(semanas).mean()
gen_estacional_semanal = gen_estacional_diaria.groupby(semanas).mean()
gen_fijo_semanal = gen_fijo_diaria.groupby(semanas).mean()

plt.figure(figsize=(14, 5))
plt.step(gen_optima_semanal.index * 7, gen_optima_semanal, label='Seguidor (semanal)', color='green')
plt.step(gen_estacional_semanal.index * 7, gen_estacional_semanal, label='Estacional (semanal)', color='orange', alpha=0.6)
plt.step(gen_fijo_semanal.index * 7, gen_fijo_semanal, label='Fijo (semanal)', color='blue', alpha=0.6)
plt.title('Promedio Semanal de Generación')
plt.xlabel('Día del Año')
plt.ylabel('Promedio semanal (Wh/día)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# --- Ángulo de incidencia en radianes ---
plt.figure(figsize=(14, 4))
plt.plot(np.clip(np.cos(theta_est_rad), 0, None), label='cos(θ) Estacional', alpha=0.6)
plt.plot(np.clip(np.cos(theta_fijo_rad), 0, None), label='cos(θ) Fijo', alpha=0.6)
plt.title('Coseno del Ángulo de Incidencia')
plt.xlabel('Hora del año')
plt.ylabel('cos(θ)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
