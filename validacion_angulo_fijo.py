import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Cargar datos desde el archivo Excel y CSV, reiniciar índices ---
file_path = "irradiacionriv.xlsx"
data = pd.read_excel(file_path).reset_index(drop=True)
temp_data = pd.read_csv(r"D:\Datos_python\generacion_esperada.csv").reset_index(drop=True)

# --- Verificar longitudes consistentes ---
assert len(data) == len(temp_data), "Los archivos tienen diferente cantidad de filas"

# --- Calcular ángulo cenital y acimut solar en radianes ---
data['theta_z'] = np.arccos(np.clip(data['CZ'], -1, 1))  # Ángulo cenital en radianes
data['az'] = np.radians(data['GS'])  # Acimut solar en radianes

# --- Definir inclinaciones ---
inclinaciones_deg = [24, 31, 38]
inclinaciones_rad = [np.radians(beta) for beta in inclinaciones_deg]

# --- Cálculo del ángulo de incidencia para cada inclinación fija ---
theta_fijo = {}
for beta_deg, beta_rad in zip(inclinaciones_deg, inclinaciones_rad):
    theta = np.arccos(
        np.cos(beta_rad) * np.cos(data['theta_z']) +
        np.sin(beta_rad) * np.sin(data['theta_z']) * np.cos(data['az'])
    )
    theta_fijo[beta_deg] = theta

# --- Parámetros del sistema ---
E_p_fabricante = 22.5      # Eficiencia nominal del panel (%)
Area_panel = 2             # Área del panel (m²)
gamma = -0.0045            # Coef. pérdida por temperatura (%/°C)

# --- Radiación ---
GHI = data['GHI']
CZ = data['CZ']
DNI = data['DNI']
DHI = GHI - DNI * CZ
DHI = np.clip(DHI, 0, None)

# --- Temperatura de la celda ---
T_celda = temp_data['Temperatura_Celda_C']

# --- Corrección por temperatura ---
eficiencia_corr = E_p_fabricante * (1 + gamma * (T_celda - 45))
eficiencia_corr = eficiencia_corr.clip(lower=0) / 100

# --- Generación hora a hora para cada inclinación ---
gen_fijo = {}
for beta_deg in inclinaciones_deg:
    GHI_corr = DNI * np.clip(np.cos(theta_fijo[beta_deg]), 0, None)
    gen = (GHI_corr + DHI) * Area_panel * eficiencia_corr
    gen_fijo[beta_deg] = gen

# --- Agrupar por día ---
dias = np.arange(len(GHI)) // 24
gen_fijo_diaria = {beta: pd.Series(gen).groupby(dias).sum() for beta, gen in gen_fijo.items()}

# --- Promedios diarios ---
print("Promedios diarios por inclinación:")
for beta, gen in gen_fijo_diaria.items():
    print(f"  Inclinación {beta}°: {gen.mean():.2f} Wh/día")

# --- Gráfica: Generación hora a hora ---
plt.figure(figsize=(14, 5))
for beta, gen in gen_fijo.items():
    plt.plot(gen, label=f'{beta}°', linewidth=0.6)
plt.title('Generación Hora a Hora para Diferentes Inclinaciones')
plt.xlabel('Horas')
plt.ylabel('Generación (W)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# --- Gráfica: Generación diaria por inclinación ---
plt.figure(figsize=(14, 5))
for beta, gen in gen_fijo_diaria.items():
    plt.plot(gen, label=f'{beta}°')
plt.title('Generación Diaria por Inclinación')
plt.xlabel('Día del Año')
plt.ylabel('Energía diaria (Wh)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# --- Gráfica: Coseno del ángulo de incidencia ---
plt.figure(figsize=(14, 4))
for beta in inclinaciones_deg:
    plt.plot(np.clip(np.cos(theta_fijo[beta]), 0, None), label=f'{beta}°', alpha=0.6)
plt.title('Coseno del Ángulo de Incidencia para Inclinaciones Fijas')
plt.xlabel('Hora del año')
plt.ylabel('cos(θ)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# --- Gráfica del Ángulo de Incidencia (excluyendo 59°) ---
angulo_inc_31 = np.degrees(theta_fijo[31])
angulo_inc_31_plot = angulo_inc_31.copy()
angulo_inc_31_plot[np.isclose(angulo_inc_31_plot, 59, atol=0.01)] = np.nan

plt.figure(figsize=(14, 4))
plt.plot(angulo_inc_31_plot, color='blue', linewidth=0.7, label='Angulo de Incidencia')
plt.title('Ángulo de Incidencia Solar para Inclinación Fija de 31° y Acimut 0°')
plt.axhline(90, color='red', linestyle='--', linewidth=0.8, label='90°')
plt.xlabel('Hora del año')
plt.ylabel('Ángulo de incidencia (°)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- Guardar ángulo de incidencia (sistema fijo 31°) en CSV ---
output_path = r"D:\Datos_python\incidencia_fijo.csv"
df_incidencia_31 = pd.DataFrame({
    'Hora': range(1, len(angulo_inc_31) + 1),
    'Angulo_Incidencia_Fijo_31': angulo_inc_31
})
df_incidencia_31.to_csv(output_path, index=False)

# --- Agrupar generación diaria en semanas ---
gen_31_diaria = gen_fijo_diaria[31]  # Ya está en Wh/día

# Agrupación robusta en semanas (0 a 51 + semana parcial si hay)
semanas = np.floor(np.arange(len(gen_31_diaria)) / 7).astype(int)
gen_31_semanal = gen_31_diaria.groupby(semanas).sum()

# --- Crear eje X en horas (168 horas por semana) ---
horas = np.arange(len(gen_31_semanal)) * 168


