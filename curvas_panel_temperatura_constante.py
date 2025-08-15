import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Función para curvas I-V y P-V ---
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

    for _ in range(100):
        Ia = Ia - (Iph - Ia - Ir * (np.exp((Vc + Ia * Rs) / Vt_Ta) - 1)) / \
                  (-1 - Ir * (np.exp((Vc + Ia * Rs) / Vt_Ta) - 1) * Rs / Vt_Ta)
    Ia = np.maximum(Ia, 0)

    Ppv = Va * Ia
    return Ia, Ppv

# --- 2. Parámetros de barrido ---
Va = np.linspace(0, 55, 500)
irradiaciones = [200, 400, 600, 800, 1000]  # W/m²
temperatura_fija = 43  # °C

# --- 3. Graficar I-V ---
plt.figure(figsize=(10, 5))
for G in irradiaciones:
    Suns = G / 1000
    Ipv, _ = iv_curve(Va, Suns, temperatura_fija)
    # cortar cuando Ipv sea casi cero y Va > 20
    idx_cut = np.argmax((Ipv <= 0.01) & (Va > 20))
    if idx_cut == 0 and Ipv[-1] > 0.01:
        idx_cut = len(Va)
    plt.plot(Va[:idx_cut], Ipv[:idx_cut], label=f'{G} W/m²')
plt.title(f'I-V Característica (T = {temperatura_fija} °C)')
plt.xlabel('Vpv (V)')
plt.ylabel('Ipv (A)')
plt.grid(True, which='both')
plt.legend()
plt.show()

# --- 4. Graficar P-V ---
plt.figure(figsize=(10, 5))
for G in irradiaciones:
    Suns = G / 1000
    _, Ppv = iv_curve(Va, Suns, temperatura_fija)
    # cortar cuando Ppv sea casi cero y Va > 20
    idx_cut = np.argmax((Ppv <= 0.01) & (Va > 20))
    if idx_cut == 0 and Ppv[-1] > 0.01:
        idx_cut = len(Va)
    plt.plot(Va[:idx_cut], Ppv[:idx_cut], label=f'{G} W/m²')
plt.title(f'P-V Característica (T = {temperatura_fija} °C)')
plt.xlabel('Vpv (V)')
plt.ylabel('Ppv (W)')
plt.grid(True, which='both')
plt.legend()
plt.show()
