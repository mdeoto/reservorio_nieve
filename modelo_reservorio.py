import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ────────────────────────────────
# 1. CONFIGURACIÓN GENERAL
# DATOS DE LAS CONSTANTES
# ────────────────────────────────

# Parámetros físicos y operativos
R_max         = 125000.0                    # Capacidad máxima del embalse (en m3)
R_min         = 0.0                         # Nivel mínimo físico (en m3)
P_ini         = 100.0                       # Porcentaje inicial del nivel del embalse (entre 0 y 100)
R_init        = R_max * (P_ini/100.0)       # Nivel ó Condicion Inicial del embalse

# Parámetros del modelo de aporte y consumo
A1_k          = 5.00      # Coef. base de escurrimiento
A1_alpha      = 0.25      # Tasa de crecimiento de A1 con Tw
A1_max        = 10800.0   # m³/día
A1_central    = 0.0       # Tw en la que A1 ≈ A1_max / 2

N_max         = 6000.0  # m³/día
N_central     = 3.00      # Centro de la sigmoide de N (°C)
N_steep       = 1.50      # Pendiente de la sigmoide

# Parámetros de descarga ordinaria
a_salida      = 0.05      # Coeficiente de descarga (lineal con R)
A2_min        = 0.5       # Descarga mínima

# ────────────────────────────────
# 2. CARGA DE DATOS
# ────────────────────────────────

df = pd.read_csv('Tw_desde_T_y_RH.csv')  # columnas: 't', 'Tw'
t_series  = df['t'].values                  # tiempos (en dias), si es en horas ajustar constantes!
t_series_dt = pd.to_datetime(t_series)      # Para Plots: Asegurar que t_series sea datetime
Tw_series = df['Tw'].values                 # Temperatura de Bulbo Húmedo
n_steps   = len(Tw_series)                  # Cantidad de iteraciones (depende de la longitud del archivo)
dt        = 1.0                             # Paso de integracion. Dejar fijo siempre en 1

# ────────────────────────────────
# 3. FUNCIONES EMPÍRICAS
# ────────────────────────────────

# Ingreso de Agua al Reservorio (A. La Cascada)
# Esto es fundamental, pero requiere de mucho cononcimiento base
# def A1(Tw):   # Perfil exponencial
#     return A1_k * np.exp(A1_alpha * Tw)
def A1(Tw):     # Perfil sigmoide, el cual es mas limitado
    return A1_max / (1 + np.exp(-A1_alpha * (Tw - A1_central)))
# Generacion de Nieve Artificial
def N(Tw):
    return N_max / (1 + np.exp(N_steep * (Tw - (-N_central))))
# Egreso de Agua (A. La Cascada)
def A2_ord(R):
    return max(a_salida * R, A2_min)

# ────────────────────────────────
# 4. INTEGRACIÓN TEMPORAL
# ────────────────────────────────

R = np.zeros(n_steps)   # Lleno con ceros el array
R[0] = R_init           # Condicion Inicial

for n in range(n_steps - 1):
    a1_val = A1(Tw_series[n])
    n_val  = N(Tw_series[n])
    ### Condiciones ###
    # Si el Reservorio esta lleno, y el ingreso de agua es mayor que el consumo de generacion de nieve
    # Entonces el egreso de agua es la diferencia entre lo que ingresa y se consume
    if (R[n] >= R_max) and (a1_val > n_val):
        a2_val = a1_val - n_val     # Basicamente dR/dt = 0 (variacion del reservorio nula)
    # En cualquier caso distinto el egreso se calcula
    else:
        a2_val = A2_ord(R[n])
    # Euler
    dR = a1_val - n_val - a2_val
    R[n+1] = R[n] + dt * dR
    # Reestricciones del Reservorio, no puede ser menor que un R_min ni mayor que un R_max
    R[n+1] = np.clip(R[n+1], R_min, R_max)


# ────────────────────────────────
# 5. VISUALIZACIONES
# ────────────────────────────────

# Recalcular series
A1_series = np.array([A1(tw) for tw in Tw_series])
N_series  = np.array([N(tw) for tw in Tw_series])
A2_series = []

for n in range(n_steps - 1):
    if (R[n] >= R_max) and (A1_series[n] > N_series[n]):
        A2_series.append(A1_series[n] - N_series[n])
    else:
        A2_series.append(A2_ord(R[n]))
A2_series.append(A2_series[-1])
A2_series = np.array(A2_series)

# 5.1 R y Tw
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axs[0].plot(t_series_dt, 100 * R / R_max, label='R(t) [%]', color='blue', linewidth=2)
axs[0].set_ylabel('Nivel del Reservorio (%)')
axs[0].set_title('Evolución del Nivel del Reservorio')
axs[0].legend(); axs[0].grid(True)
axs[0].annotate(f'Máx: {np.max(R):.0f} m³', xy=(t_series_dt[np.argmax(R)], 100 * np.max(R)/R_max),
                xytext=(0, 10), textcoords='offset points', arrowprops=dict(arrowstyle='->'))
axs[0].annotate(f'Mín: {np.min(R):.0f} m³', xy=(t_series_dt[np.argmin(R)], 100 * np.min(R)/R_max),
                xytext=(0, -25), textcoords='offset points', arrowprops=dict(arrowstyle='->'))
axs[1].plot(t_series_dt, Tw_series, label='Tw(t)', color='red', alpha=0.8)
axs[1].set_ylabel('Tw (°C)')
axs[1].set_title('Temperatura de Bulbo Húmedo (Tw)')
axs[1].legend(); axs[1].grid(True)
axs[1].annotate(f'Máx: {np.max(Tw_series):.1f} °C', xy=(t_series_dt[np.argmax(Tw_series)], np.max(Tw_series)),
                xytext=(0, 10), textcoords='offset points', arrowprops=dict(arrowstyle='->'))
axs[1].annotate(f'Mín: {np.min(Tw_series):.1f} °C', xy=(t_series_dt[np.argmin(Tw_series)], np.min(Tw_series)),
                xytext=(0, -25), textcoords='offset points', arrowprops=dict(arrowstyle='->'))
locator_major = mdates.DayLocator(interval=5)
locator_minor = mdates.DayLocator(interval=1)
formatter     = mdates.DateFormatter('%d-%b')
axs[1].xaxis.set_major_locator(locator_major)
axs[1].xaxis.set_minor_locator(locator_minor)
axs[1].xaxis.set_major_formatter(formatter)
axs[1].tick_params(axis='x', which='major', rotation=45)
axs[1].tick_params(axis='x', which='minor', length=4, color='gray')

plt.xlabel('Tiempo (días)')
plt.tight_layout()
#plt.savefig("reservorio_tw.png", dpi=300)
plt.show()

# 5.2 A1 vs Tw
Tw_vals = np.linspace(-10, 10, 400)
plt.figure(figsize=(6,4))
plt.plot(Tw_vals, [A1(tw) for tw in Tw_vals])
plt.xlabel('Tw (°C)'); plt.ylabel('A1 (m³/día)')
plt.title('Ingreso A1 al reservorio en función de Tw')
plt.grid(True); plt.tight_layout()
#plt.savefig("A1_vs_Tw.png", dpi=300)
plt.show()

# 5.3 N vs Tw
plt.figure(figsize=(6,4))
plt.plot(Tw_vals, [N(tw) for tw in Tw_vals], color='orange')
plt.xlabel('Tw (°C)'); plt.ylabel('N (m³/día)')
plt.title('Consumo N de nieve artificial en función de Tw')
plt.grid(True); plt.tight_layout()
#plt.savefig("N_vs_Tw.png", dpi=300)
plt.show()

# 5.4 A2 real (egreso) y R con doble eje y
fig, ax1 = plt.subplots(figsize=(10, 4))
# Primer eje: A2 (descarga)
ax1.plot(t_series_dt, A2_series, label='A2(t) [m³/día]', color='green')
ax1.set_ylabel('Descarga A2 (m³/día)', color='green')
ax1.tick_params(axis='y', labelcolor='green')
ax1.grid(True)
ax1.set_title('Descarga del Arroyo La Cascada (A2) y Nivel del Reservorio')
# Anotaciones A2
ax1.annotate(f'Máx A2: {np.max(A2_series):.0f} m³/día',
             xy=(t_series_dt[np.argmax(A2_series)], np.max(A2_series)),
             xytext=(0, 10), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='green'), color='green')
ax1.annotate(f'Mín A2: {np.min(A2_series):.0f} m³/día',
             xy=(t_series_dt[np.argmin(A2_series)], np.min(A2_series)),
             xytext=(0, -25), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='green'), color='green')
# Segundo eje: R (nivel del embalse)
ax2 = ax1.twinx()
ax2.plot(t_series_dt, R, label='R(t) [m³]', color='blue', linestyle='--', alpha=0.6)
ax2.set_ylabel('Volumen del Reservorio (m³)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
# Anotaciones R
ax2.annotate(f'Máx R: {np.max(R):.0f} m³',
             xy=(t_series_dt[np.argmax(R)], np.max(R)),
             xytext=(-50, 10), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
ax2.annotate(f'Mín R: {np.min(R):.0f} m³',
             xy=(t_series_dt[np.argmin(R)], np.min(R)),
             xytext=(-50, -25), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
# Eje temporal
ax1.set_xlabel('Tiempo (días)')
ax1.xaxis.set_major_locator(locator_major)
ax1.xaxis.set_minor_locator(locator_minor)
ax1.xaxis.set_major_formatter(formatter)
ax1.tick_params(axis='x', which='major', rotation=45)
ax1.tick_params(axis='x', which='minor', length=4, color='gray')
plt.tight_layout()
#plt.savefig("A2_R_twinx.png", dpi=300)
plt.show()


# ────────────────────────────────
# 6. EXPORTACIÓN A CSV
# ────────────────────────────────

df_out = pd.DataFrame({
    't': t_series_dt,
    'Tw': Tw_series,
    'R_m3': R,
    'R_pct': 100 * R / R_max,
    'A1_m3_dia': A1_series,
    'N_m3_dia': N_series,
    'A2_m3_dia': A2_series
})
# Descomentar para guardar
# df_out.to_csv('salida_modelo_reservorio.csv', index=False)
