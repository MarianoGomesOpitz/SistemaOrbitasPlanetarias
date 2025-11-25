import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy.integrate import solve_ivp

# Ajustes para permitir la visualización en el notebook
plt.rcParams['animation.embed_limit'] = 50.0 # Límite de 50 MB

# --- 1. MÉTODOS DE INTEGRACIÓN ---

def EulerExplicito(f, t0, y0, h, n):
    y0 = np.array(y0, dtype=float)
    ts = [t0]
    ys = [y0]
    for i in range(n):
        y0 = y0 + h * np.array(f(t0, y0))
        t0 = t0 + h
        ts.append(t0)
        ys.append(y0)
    return np.array(ts), np.array(ys)

def RK2_Trapecio(f, t0, y0, h, n):
    y0 = np.array(y0, dtype=float)
    ts = [t0]
    ys = [y0]
    for i in range(n):
        # Implementación de Heun (Explícito Trapecio Predictor-Corrector)
        k1 = np.array(f(t0, y0))
        k2 = np.array(f(t0 + h, y0 + h * k1))
        y0 = y0 + 0.5 * h * (k1 + k2)
        t0 = t0 + h
        ts.append(t0)
        ys.append(y0)
    return np.array(ts), np.array(ys)

def RungeKutta4(f, t0, y0, h, n):
    y_actual = np.array(y0, dtype=float)
    t_actual = t0
    ts = [t_actual]
    ys = [y_actual.copy()]
    
    for i in range(n):
        k1 = np.array(f(t_actual, y_actual))
        k2 = np.array(f(t_actual + 0.5*h, y_actual + 0.5*h*k1))
        k3 = np.array(f(t_actual + 0.5*h, y_actual + 0.5*h*k2))
        k4 = np.array(f(t_actual + h, y_actual + h*k3))
        
        y_actual = y_actual + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t_actual = t_actual + h
        
        ts.append(t_actual)
        ys.append(y_actual.copy())
        
    return np.array(ts), np.array(ys)


# --- 2. MODELO FÍSICO (UN CUERPO - PROBLEMA DE DOS CUERPOS) ---

masa_estrella = 1.0  # Masa del Sol en unidades solares
# Contante gravitacional (Unidades AU/Años)
GM = 4 * (np.pi**2) * masa_estrella

def fuerza_gravitacional(t, estado):
    x, y, vx, vy = estado
    
    # Calculamos distancia al centro (r)
    r = np.sqrt(x**2 + y**2)
    
    # Ecuaciones de movimiento
    dxdt = vx
    dydt = vy
    dvxdt = -(GM / r**3) * x
    dvydt = -(GM / r**3) * y
    
    return [dxdt, dydt, dvxdt, dvydt]


# --- 3. CÁLCULO DE RESULTADOS ---

t_ini = 0
# Condiciones iniciales: Posición (1 AU, 0), Velocidad (0, 2π AU/año) para órbita circular
y_sistema = [1.0, 0.0, 0.0, 2*np.pi] 

# NOTA: Usamos un paso GRANDE (h=0.04) para acentuar el error de Euler
h = 0.01  
n = 5000 
print(f"Simulando {n*h} años con h={h}...")

ts_euler, ys_euler = EulerExplicito(fuerza_gravitacional, t_ini, y_sistema, h, n)
ts_trapecio, ys_trapecio = RK2_Trapecio(fuerza_gravitacional, t_ini, y_sistema, h, n)
ts_rk4, ys_rk4 = RungeKutta4(fuerza_gravitacional, t_ini, y_sistema, h, n)

t_fin = t_ini + n*h
sol = solve_ivp(
    fun=fuerza_gravitacional, 
    t_span=(t_ini, t_fin), 
    y0=y_sistema, 
    t_eval=np.linspace(t_ini, t_fin, n),
    method='RK45', 
    rtol=1e-9, 
    atol=1e-8)

ts_rk45 = sol.t
ys_rk45 = sol.y.T

t_fin = t_ini + n*h
sol = solve_ivp(
    fun=fuerza_gravitacional, 
    t_span=(t_ini, t_fin), 
    y0=y_sistema, 
    t_eval=np.linspace(t_ini, t_fin, n),
    method='Radau', 
    rtol=1e-9, 
    atol=1e-8)

ts_radau = sol.t
ys_radau = sol.y.T



print("Cálculos terminados.")

# --- 4. PREPARACIÓN DE LA ANIMACIÓN ---

fig, ax = plt.subplots(figsize=(8, 8))
# Los límites deben ser fijos y grandes, ya que Euler se dispara
limite = 5.0 


ax.set_xlim(-limite, limite)
ax.set_ylim(-limite, limite)
ax.set_aspect('equal')
ax.grid(False)
ax.set_title(f"Comparación Métodos Numéricos (h={h}, n={n}, {n*h} años terrestres)")
ax.set_facecolor('black')

# Dibujar el Sol
ax.plot(0, 0, 'yo', markersize=15, label='Sol')

# Inicialización de líneas (Rastros) y puntos (Posiciones actuales)
line_euler, = ax.plot([], [], 'r--', lw=1, alpha=0.6, label='Euler (Error ∝ h)')
line_trapecio, = ax.plot([], [], 'g--', lw=1, alpha=0.6, label='Trapecio (Error ∝ h²)')
line_rk4, = ax.plot([], [], 'y--', lw=1.5, alpha=0.6, label='RK4 (Error ∝ h⁴)')
line_rk45, = ax.plot([], [], 'b--', lw=1.5, alpha=0.6, label='RK45 (Error ∝ h⁵)')
line_radau, = ax.plot([], [], 'm--', lw=1.5, alpha=0.6, label='Radau (SciPy)')

punto_euler, = ax.plot([], [], 'ro', markersize=6)
punto_trapecio, = ax.plot([], [], 'go', markersize=6)
punto_rk4, = ax.plot([], [], 'yo', markersize=6)
punto_rk45, = ax.plot([], [], 'bo', markersize=6)
punto_radau, = ax.plot([], [], 'mo', markersize=6)

texto_tiempo = ax.text(0.05, 0.95, '', transform=ax.transAxes, color="white", fontsize=12)

ax.legend(loc="lower right", fontsize='small')

# Lista de las trayectorias para fácil manejo en la actualización

trayectorias = [ys_euler, ys_trapecio, ys_rk4, ys_rk45, ys_radau]
lineas = [line_euler, line_trapecio, line_rk4, line_rk45, line_radau]
puntos = [punto_euler, punto_trapecio, punto_rk4, punto_rk45, punto_radau]

# --- 5. FUNCIÓN DE ACTUALIZACIÓN ---

def update(frame):
    
    for i in range(4): # Iteramos sobre los 4 métodos
        ys = trayectorias[i]
        linea = lineas[i]
        punto = puntos[i]
        
        # 1. ACTUALIZAR EL RASTRO (Rastro Infinito)
        # El slice comienza en 0 y termina en el frame actual (exclusivo)
        x_hist = ys[:frame, 0] 
        y_hist = ys[:frame, 1]
        linea.set_data(x_hist, y_hist)
        
        # 2. Actualizar la Posición actual (El punto)
        x_actual = ys[frame, 0]
        y_actual = ys[frame, 1]
        punto.set_data([x_actual], [y_actual])
        
    t_actual = ts_rk45[frame]
    texto_tiempo.set_text(f'Tiempo: {t_actual:.2f} años')

    return lineas + puntos + [texto_tiempo]

# --- 6. EJECUCIÓN Y RENDERIZADO ---

# Saltamos frames para que la animación no sea demasiado lenta
salto = 1 
indices_frames = range(0, len(ts_radau), salto)

print("Generando animación...")
ani = FuncAnimation(fig, update, frames=indices_frames, interval=20, blit=True)

plt.show()



# Guardar como GIF
#print("Guardando como MP4 (esto puede tardar un poco)...")
#ani.save('orbita_solar.gif', writer='pillow', fps=30)
#print("Animación guardada como 'orbita_solar.mp4'.")

#Guardar como MP4 (Mejor calidad)
#print("Guardando como MP4 (esto puede tardar un poco)...")
#ani.save('orbita_solar.mp4', writer='ffmpeg', fps=30)
#print("Animación guardada como 'orbita_solar.mp4'.")