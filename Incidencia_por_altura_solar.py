import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar datos desde el archivo Excel
file_path = "irradiacionriv.xlsx"
data = pd.read_excel(file_path)

# Parámetros fijos
gamma_p = -59  # Inclinación del panel (grados)

# Lista para almacenar ángulo de incidencia con signo
angulo_incidencia_altura_solar = []


# Iterar por las filas del DataFrame
for index, row in data.iterrows():
    Altura_solar = row['AS']  # Altura solar en grados

    # Validar datos: si altura solar es 0 o NaN, asignar nan
    #if pd.isna(Altura_solar) or Altura_solar == 0:
     #   angulo_incidencia_altura_solar.append(np.nan)
      #  continue

    # Calcular el ángulo de incidencia con signo directamente
    angulo_incidencia = Altura_solar + gamma_p
    angulo_incidencia_altura_solar.append(angulo_incidencia)

# Guardar los datos en un archivo Excel (.xlsx)
df_resultados = pd.DataFrame({
    'Hora': range(1, len(angulo_incidencia_altura_solar) + 1),
    'Angulo_Incidencia_Altura_Solar': angulo_incidencia_altura_solar
})
df_resultados.to_csv('Angulo_Incidencia_Altura_Solar.csv', index=False)


# Configurar eje X
horas = range(1, len(angulo_incidencia_altura_solar) + 1)
horas_por_dia = 24
dias_por_mes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
         'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

etiquetas_meses = []
posiciones_meses = []
acumulador_dias = 0
for i, dias in enumerate(dias_por_mes):
    posicion = acumulador_dias * horas_por_dia
    posiciones_meses.append(posicion)
    etiquetas_meses.append(meses[i])
    acumulador_dias += dias



# Graficar
plt.figure(figsize=(12, 6))
plt.plot(horas, angulo_incidencia_altura_solar, linestyle='-', color='purple',
         label='Ángulo de incidencia con signo (altura solar - inclinación panel)')
plt.xticks(posiciones_meses, etiquetas_meses, rotation=45)
plt.title("Ángulo de Incidencia sobre el Panel según altura solar")
plt.xlabel("Meses del Año")
plt.ylabel("Ángulo de Incidencia (°)")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
