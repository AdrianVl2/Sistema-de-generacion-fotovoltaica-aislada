import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------- 1. Cargar archivos ----------
# Datos principales desde Excel
data = pd.read_excel("irradiacionriv.xlsx")
data['Hora'] = range(1, len(data) + 1)

# Generación con pérdidas por temperatura desde CSV
gen_temp_df = pd.read_csv("generacion_esperada_temperatura.csv")
gen_temp_df['Hora'] = range(1, len(gen_temp_df) + 1)

# Ángulo de incidencia absoluto desde CSV
incidencia_df = pd.read_csv("angulo_de_incidencia_total.csv")
incidencia_df['Hora'] = range(1, len(incidencia_df) + 1)  # Asegura columna Hora
# Renombrar columna para usar consistentemente
incidencia_df = incidencia_df.rename(columns={
    'Angulo_Incidencia_Total_Abs': 'angulo_de_incidencia_abs'
})

# ---------- 2. Unir DataFrames ----------
# Merge con generación por temperatura
data = pd.merge(data, gen_temp_df[['Hora', 'Generacion_esperada_W']], on='Hora', how='left')

# Merge con ángulo de incidencia
data = pd.merge(data, incidencia_df[['Hora', 'angulo_de_incidencia_abs']], on='Hora', how='left')

# ---------- 3. Parámetros ----------
E_p_fabricante = 22.5  # %
Area_panel = 2  # m²

# ---------- 4. Cálculos ----------
# Generación óptima (sin pérdidas)
data['Generacion_optima'] = Area_panel * data['GHI'] * (E_p_fabricante / 100)

# Generación con pérdidas por temperatura (ya calculada)
data['Generacion_temp'] = data['Generacion_esperada_W']

# Factor de corrección por ángulo de incidencia
def factor_incidencia(ang):
    if 0 <= ang <= 90:
        return np.cos(np.radians(ang))
    else:
        return 0

data['Factor_incidencia'] = data['angulo_de_incidencia_abs'].apply(factor_incidencia)

# Generación con pérdidas solo por ángulo
data['Generacion_incidencia'] = data['Generacion_optima'] * data['Factor_incidencia']

# Generación total con ambas pérdidas
data['Generacion_total'] = data['Generacion_temp'] * data['Factor_incidencia']

# ---------- 5. Calcular área bajo la curva ----------
horas = data['Hora']

area_optima = np.trapz(data['Generacion_optima'], x=horas)
area_temp = np.trapz(data['Generacion_temp'], x=horas)
area_incidencia = np.trapz(data['Generacion_incidencia'], x=horas)
area_total = np.trapz(data['Generacion_total'], x=horas)

print(f"Área bajo la curva Generación Óptima: {area_optima:.2f} Wh")
print(f"Área bajo la curva Con pérdidas por temperatura: {area_temp:.2f} Wh")
print(f"Área bajo la curva Con pérdidas por incidencia: {area_incidencia:.2f} Wh")
print(f"Área bajo la curva Total (temperatura + incidencia): {area_total:.2f} Wh")

# ---------- 6. Graficar ----------
plt.figure(figsize=(14, 7))

plt.plot(horas, data['Generacion_optima'], label='Óptima (sin pérdidas)', color='green', linestyle='-', alpha=0.25)
plt.plot(horas, data['Generacion_temp'], label='Con pérdidas temperatura', color='red', alpha=0.5)
plt.plot(horas, data['Generacion_incidencia'], label='Con pérdidas incidencia', color='orange', alpha=0.8)
plt.plot(horas, data['Generacion_total'], label='Total (temperatura + incidencia)', color='blue', linewidth=1.5)

plt.xlabel('Hora')
plt.ylabel('Generación (W)')
plt.title('Comparación de Generación Fotovoltaica: Óptima vs. Pérdidas')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
