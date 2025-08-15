import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Leer datos de irradiación para tres sistemas ---
archivo_irr = r"C:\Users\adria\OneDrive\Escritorio\informacion proyecto final\programas_python\ghi_corregido.xlsx"

irradiaciones_sistemas = {
    "Seguidor":   pd.read_excel(archivo_irr, usecols=["GHI_corr_seguidor"])["GHI_corr_seguidor"].values.flatten(),
    "Estacional": pd.read_excel(archivo_irr, usecols=["GHI_corr_estacional"])["GHI_corr_estacional"].values.flatten(),
    "Fijo":       pd.read_excel(archivo_irr, usecols=["GHI_corr_fijo"])["GHI_corr_fijo"].values.flatten()
}

# --- Temperatura de la celda  ---
ruta_temp_pv = r"D:\Datos_python\temperatura_celda.csv"
temp_pv = pd.read_csv(ruta_temp_pv)["Temperatura_celda"].values.flatten()

# --- 2. Función para curvas I-V y P-V ---
def iv_curve(Va, Suns, TaC):
    k = 1.38e-23
    q = 1.6e-19
    A = 0.6
    Vg = 0.595
    Ns = 144
    T1 = 273 + 43
    Voc_T1 = 52.9 / Ns
    Isc_T1 = 10.74
    T2 = 273 + 80
    Voc_T2 = 52.8112 / Ns
    Isc_T2 = 10.7548
    TarK = 273 + TaC

    Iph_T1 = Isc_T1 * Suns
    a = (Isc_T2 - Isc_T1) / Isc_T1 * 1 / (T2 - T1)
    Iph = Iph_T1 * (1 + a * (TarK - T1))
    Vt_T1 = k * T1 / q
    Ir_T1 = Isc_T1 / (np.exp(Voc_T1 / (A * Vt_T1)) - 1)
    b = Vg * q / (A * k)
    Ir = Ir_T1 * (TarK / T1) ** (3 / A) * np.exp(-b * (1 / TarK - 1 / T1))
    X2v = Ir_T1 / (A * Vt_T1) * np.exp(Voc_T1 / (A * Vt_T1))
    dVdI_Voc = -0.3 / Ns / 2
    Rs = -dVdI_Voc - 1 / X2v
    Vt_Ta = A * k * TarK / q
    Vc = Va / Ns

    Ia = np.zeros_like(Vc)
    for _ in range(10):
        Ia = Ia - (Iph - Ia - Ir * (np.exp((Vc + Ia * Rs) / Vt_Ta) - 1)) / \
             (-1 - Ir * (np.exp((Vc + Ia * Rs) / Vt_Ta) - 1) * Rs / Vt_Ta)

    Ia = np.maximum(Ia, 0)
    Ppv = Va * Ia
    return Ia, Ppv

# --- 3. Configuración del barrido ---
Va = np.linspace(0, 52, 500)
horas_anio = np.arange(len(temp_pv))

# Diccionarios para guardar resultados
resultados_V = {}
resultados_P = {}
resultados_I = {}

# --- 4. Calcular para cada sistema ---
for nombre_sistema, irradiacion in irradiaciones_sistemas.items():
    P_MPPT = []
    V_MPPT = []

    for G, T in zip(irradiacion, temp_pv):
        Suns = G / 1000
        Ipv, Ppv = iv_curve(Va, Suns, T)
        if np.max(Ppv) > 0:
            idx = np.argmax(Ppv)
            P_MPPT.append(Ppv[idx])
            V_MPPT.append(Va[idx])
        else:
            P_MPPT.append(np.nan)
            V_MPPT.append(np.nan)

    P_MPPT = np.array(P_MPPT)
    V_MPPT = np.array(V_MPPT)
    I_MPPT = P_MPPT / V_MPPT

    resultados_V[nombre_sistema] = V_MPPT
    resultados_P[nombre_sistema] = P_MPPT
    resultados_I[nombre_sistema] = I_MPPT

# --- 5. Definir estilos para cada sistema ---
estilos = {
    "Seguidor":   {"color": "red",    "lw": 2.5, "alpha": 1.0, "zorder": 3, "linestyle": "-"},
    "Estacional": {"color": "blue",   "lw": 2.0, "alpha": 1.0, "zorder": 2, "linestyle": "--"},
    "Fijo":       {"color": "green",  "lw": 1.5, "alpha": 1.0, "zorder": 1, "linestyle": ":"}
}


# --- 6. Graficar todas las variables ---
plots = [
    ("Tensión MPP por hora del año", resultados_V, "Tensión (V)"),
    ("Potencia MPP por hora del año", resultados_P, "Potencia (W)"),
    ("Corriente MPP por hora del año", resultados_I, "Corriente (A)")
]

for titulo, datos_dict, ylabel in plots:
    plt.figure(figsize=(15, 4))
    for sistema, datos in datos_dict.items():
        plt.plot(
            horas_anio, datos, label=sistema,
            color=estilos[sistema]["color"],
            linewidth=estilos[sistema]["lw"],
            alpha=estilos[sistema]["alpha"]
        )
    plt.title(titulo)
    plt.xlabel('Hora del año')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# --- 5. Imprimir valores máximos de cada sistema ---
print("Valores máximos alcanzados por cada sistema:")
for sistema in irradiaciones_sistemas.keys():
    V_max = np.nanmax(resultados_V[sistema])
    P_max = np.nanmax(resultados_P[sistema])
    I_max = np.nanmax(resultados_I[sistema])
    print(f"\n{sistema}:")
    print(f"  Tensión máxima (V): {V_max:.2f}")
    print(f"  Potencia máxima (W): {P_max:.2f}")
    print(f"  Corriente máxima (A): {I_max:.2f}")

