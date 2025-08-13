import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Cargar datos desde Excel y CSV ---
file_path = "irradiacionriv.xlsx"
data = pd.read_excel(file_path)

temp_data = pd.read_csv(r"D:\Datos_python\temperatura_celda.csv")
T_celda = temp_data['Temperatura_celda']

# --- Parámetros ---
E_p_fabricante = 22.5
Area_panel = 2
gamma = -0.0045
T_ref = 25

# --- Datos solares ---
GHI = data['GHI']
DNI = data['DNI']
CZ = data['CZ']
data['theta_z'] = np.arccos(np.clip(CZ, -1, 1))
data['az'] = np.radians(data['GS'])
DHI = np.clip(GHI - DNI * CZ, 0, None)

# --- Corrección eficiencia ---
eficiencia_corr = E_p_fabricante * (1 + gamma * (T_celda - T_ref)) / 100
eficiencia_corr = eficiencia_corr.clip(lower=0)

# --- Función inclinación estacional ---
def inclinacion_estacional(doy):
    if doy < 56: return 10.5
    elif doy < 98: return 31
    elif doy < 261: return 54.5
    elif doy < 282: return 31
    else: return 10.5

# --- Cálculo ángulo incidencia para sistema estacional ---
angulos_est = []
for i, row in data.iterrows():
    doy = row['DOY']
    beta_deg = inclinacion_estacional(doy)
    beta_rad = np.radians(beta_deg)
    theta_z = row['theta_z']
    az = row['az']
    cos_theta = (np.cos(beta_rad)*np.cos(theta_z) +
                 np.sin(beta_rad)*np.sin(theta_z)*np.cos(az))
    angulos_est.append(np.arccos(np.clip(cos_theta, -1, 1)))
theta_est = np.array(angulos_est)

# --- Ángulo incidencia sistema fijo 31° ---
beta_fijo_rad = np.radians(31)
theta_fijo = np.arccos(
    np.cos(beta_fijo_rad) * np.cos(data['theta_z']) +
    np.sin(beta_fijo_rad) * np.sin(data['theta_z']) * np.cos(data['az'])
)

# --- Ángulo incidencia sistema con seguidor: siempre incidencia 0 ---
# Para sistema seguidor la radiación directa incide siempre perpendicularmente
theta_seguidor = np.zeros_like(GHI)

# --- Calcular irradiancia corregida para cada sistema ---
GHI_corr_seguidor = DNI * np.clip(np.cos(theta_seguidor), 0, None) + DHI
GHI_corr_estacional = DNI * np.clip(np.cos(theta_est), 0, None) + DHI
GHI_corr_fijo = DNI * np.clip(np.cos(theta_fijo), 0, None) + DHI

# --- Generación hora a hora ---
gen_seguidor = GHI_corr_seguidor * Area_panel * eficiencia_corr
gen_estacional = GHI_corr_estacional * Area_panel * eficiencia_corr
gen_fijo = GHI_corr_fijo * Area_panel * eficiencia_corr

# --- Agrupar por día ---
dias = np.arange(len(GHI)) // 24
gen_seguidor_diaria = pd.Series(gen_seguidor).groupby(dias).sum()
gen_estacional_diaria = pd.Series(gen_estacional).groupby(dias).sum()
gen_fijo_diaria = pd.Series(gen_fijo).groupby(dias).sum()

# --- Recortar a 364 días (52 semanas completas) ---
gen_seguidor_diaria = gen_seguidor_diaria[:364]
gen_estacional_diaria = gen_estacional_diaria[:364]
gen_fijo_diaria = gen_fijo_diaria[:364]

# --- Agrupar por semana ---
semanas = np.arange(len(gen_seguidor_diaria)) // 7
gen_seguidor_semanal = gen_seguidor_diaria.groupby(semanas).mean()
gen_estacional_semanal = gen_estacional_diaria.groupby(semanas).mean()
gen_fijo_semanal = gen_fijo_diaria.groupby(semanas).mean()

# --- Crear eje X en horas (inicio de cada semana) ---
horas = np.arange(len(gen_seguidor_semanal)) * 168  # 168 horas por semana

# --- Graficar generación semanal promedio ---
plt.figure(figsize=(14, 6))
plt.step(horas / 24, gen_seguidor_semanal, where='mid', label='Sistema con Seguidor', color='green', linewidth=2)
plt.step(horas / 24, gen_estacional_semanal, where='mid', label='Sistema Estacional', color='orange', linewidth=2)
plt.step(horas / 24, gen_fijo_semanal, where='mid', label='Sistema Fijo (31°)', color='blue', linewidth=2)
plt.title('Generación Semanal Promedio de Sistemas Fotovoltaicos')
plt.xlabel('Días del Año')
plt.ylabel('Energía semanal promedio (Wh/semana)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


# --- Generación anual sumando los días (Wh) ---
generacion_anual_seguidor = gen_seguidor_diaria.sum()
generacion_anual_estacional = gen_estacional_diaria.sum()
generacion_anual_fijo = gen_fijo_diaria.sum()

# --- Mostrar resultados en consola ---
print("Generación anual total (Wh):")
print(f"  Sistema con Seguidor: {generacion_anual_seguidor:.2f} Wh")
print(f"  Sistema Estacional:   {generacion_anual_estacional:.2f} Wh")
print(f"  Sistema Fijo (31°):   {generacion_anual_fijo:.2f} Wh")



# --- Guardar GHI corregido para sistemas estacional y fijo en un archivo Excel ---
df_ghi_corr = pd.DataFrame({
    'GHI_corr_seguidor': GHI_corr_seguidor,
    'GHI_corr_estacional': GHI_corr_estacional,
    'GHI_corr_fijo': GHI_corr_fijo

})

output_path = r"C:\Users\adria\OneDrive\Escritorio\informacion proyecto final\programas_python\ghi_corregido.xlsx"
df_ghi_corr.to_excel(output_path, index=False)

print(f"Archivo guardado en: {output_path}")
