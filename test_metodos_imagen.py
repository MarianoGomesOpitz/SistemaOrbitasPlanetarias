import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
        y0 = y0 + 0.5*h*( np.array(f(t0, y0)) + np.array(f(t0 + h, y0 + h*np.array(f(t0, y0)))) )
        t0 = t0 + h
        ts.append(t0)
        ys.append(y0)
    return np.array(ts), np.array(ys)

    
def RungeKutta4(f, t0, y0, h, n):
    y_actual = np.array(y0, dtype=float)
    t_actual = t0
    
    ts = [t_actual]
    ys = [y_actual.copy()] # Guardamos copia inicial
    
    for i in range(n):
        # Los 4 pasos de RK4
        k1 = np.array(f(t_actual, y_actual))
        k2 = np.array(f(t_actual + 0.5*h, y_actual + 0.5*h*k1))
        k3 = np.array(f(t_actual + 0.5*h, y_actual + 0.5*h*k2))
        k4 = np.array(f(t_actual + h, y_actual + h*k3))
        
        # Promedio ponderado
        y_actual = y_actual + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t_actual = t_actual + h
        
        ts.append(t_actual)
        ys.append(y_actual.copy())
        
    return np.array(ts), np.array(ys)


# Contante gravitacional, la medición es en base a la masa del Sol (1)
GM = 4 * (np.pi**2) 

def fuerza_gravitacional(t, estado):
    # Desempaquetamos el vector de estado
    x = estado[0]
    y = estado[1]
    vx = estado[2]
    vy = estado[3]
    
    # Calculamos distancia al centro (r)
    r = np.sqrt(x**2 + y**2)
    
    # Ecuaciones de movimiento
    dxdt = vx
    dydt = vy
    dvxdt = -(GM / r**3) * x
    dvydt = -(GM / r**3) * y
    
    return [dxdt, dydt, dvxdt, dvydt]


# --- CONDICIONES INICIALES (Tierra) ---
# Usando Unidades Astronómicas (AU) y Años
t_ini = 0
# Posición: 1 AU de distancia en X
# Velocidad: 2*pi AU/año en Y (velocidad necesaria para órbita circular con GM=4pi^2)
y_sistema = [1.0, 0.0, 0.0, 2*np.pi] 

h = 0.01   # Paso de tiempo (en años)
n = 200    # Cantidad de pasos (2 años aprox)

# 1. Calcular con Euler 
ts_euler, ys_euler = EulerExplicito(fuerza_gravitacional, t_ini, y_sistema, h, n)

# 2. Calcular con Trapecio
ts_trapecio, ys_trapecio = RK2_Trapecio(fuerza_gravitacional, t_ini, y_sistema, h, n)

# 3. Calcular con RK4 
ts_rk4, ys_rk4 = RungeKutta4(fuerza_gravitacional, t_ini, y_sistema, h, n)

# 4. Calcular con solve_ivp (RK45)
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

# 5. Calcular con solve_ivp (Radau)
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

# --- VISUALIZACIÓN ---
plt.figure(figsize=(8, 8))

# Dibujar el Sol en el centro
plt.plot(0, 0, 'yo', markersize=15, label='Sol')

# Dibujar Orbita Euler
plt.plot(ys_euler[:, 0], ys_euler[:, 1], 'r--', label='Euler (No estable)')
plt.plot(ys_euler[-1, 0], ys_euler[-1, 1], 'ro') # Punto final Euler

# Dibujar Orbita Trapecio
plt.plot(ys_trapecio[:, 0], ys_trapecio[:, 1], 'g--', label='Trapecio (Estable para h chico)')
plt.plot(ys_trapecio[-1, 0], ys_trapecio[-1, 1], 'go') # Punto final Euler

# Dibujar Orbita RK4
plt.plot(ys_rk4[:, 0], ys_rk4[:, 1], 'b--', label='Runge-Kutta 4 (Estable)')
plt.plot(ys_rk4[-1, 0], ys_rk4[-1, 1], 'bo') # Punto final RK4

# Dibujar Orbita RK45
plt.plot(ys_rk45[:, 0], ys_rk45[:, 1], 'y--', label='RK45 (SciPy)')
plt.plot(ys_rk45[-1, 0], ys_rk45[-1, 1], 'yo') # Punto final RK45

# Dibujar Orbita Radau
plt.plot(ys_radau[:, 0], ys_radau[:, 1], 'm--', label='Radau (SciPy)')
plt.plot(ys_radau[-1, 0], ys_radau[-1, 1], 'mo') # Punto final Radau

plt.title(f"Simulación Orbital: Euler vs Tarpecio vs RK4 vs RK45 vs Radau (h={h})")
plt.xlabel("Posición X (AU)")
plt.ylabel("Posición Y (AU)")
plt.legend()
plt.grid(True)
plt.legend(loc='lower right')
plt.axis('equal') # Importante para que se vea circular y no ovalado
plt.show()