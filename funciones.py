import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from IPython.display import HTML
#from scipy.integrate import solve_ivp


# Función de Euler Explícito
def EulerExplicito(f, t0, y0, h, n, masas):
    y0 = np.array(y0, dtype=float)
    ts = [t0]
    ys = [y0]
    for i in range(n):
        y0 = y0 + h * np.array(f(t0, y0, masas))
        t0 = t0 + h
        ts.append(t0)
        ys.append(y0)
    return np.array(ts), np.array(ys)

# Función de Trapecio (RK2)
def RK2_Trapecio(f, t0, y0, h, n, masas):
    y0 = np.array(y0, dtype=float)
    ts = [t0]
    ys = [y0]
    for i in range(n):
        # Implementación de Heun (Explícito Trapecio Predictor-Corrector)
        k1 = np.array(f(t0, y0, masas))
        k2 = np.array(f(t0 + h, y0 + h * k1, masas))
        y0 = y0 + 0.5 * h * (k1 + k2)
        t0 = t0 + h
        ts.append(t0)
        ys.append(y0)
    return np.array(ts), np.array(ys)

# Función de Runge-Kutta 4, el mejor método para este análisis
def RungeKutta4(f, t0, y0, h, n, masas):
    y_actual = np.array(y0, dtype=float)
    t_actual = t0
    
    ts = [t_actual]
    ys = [y_actual.copy()] # Guardo copia inicial
    
    for i in range(n):
        # Los 4 pasos de RK4
        k1 = np.array(f(t_actual, y_actual, masas))
        k2 = np.array(f(t_actual + 0.5*h, y_actual + 0.5*h*k1, masas))
        k3 = np.array(f(t_actual + 0.5*h, y_actual + 0.5*h*k2, masas))
        k4 = np.array(f(t_actual + h, y_actual + h*k3, masas))
        
        # Promedio ponderado
        y_actual = y_actual + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t_actual = t_actual + h
        
        ts.append(t_actual)
        ys.append(y_actual.copy())
        
    return np.array(ts), np.array(ys)



# Constante Gravitacional (es igual a 4*pi^2 en relacion con la masa del Son y las distancias son en AU)
G = 4 * (np.pi**2)

def Generar_Sistema(datos_planetas):
    masas = []          # Lista de masas de los cuerpos
    nombres = []        # Lista de nombres de los cuerpos
    colores = []        # Lista de colores de los cuerpos
    tamano_puntos = []  # Lista de tamaños de los puntos de los cuerpos
    estado_inicial_lista = []  # Lista de estados iniciales de los cuerpos

    for nombre, datos in datos_planetas.items():
        m = datos['m']
        r = datos['r']
        color = datos['c']
        tamanio = datos['t']
        
        # Guardo masas, nombre, colores, y tamaños
        masas.append(m)
        nombres.append(nombre)
        colores.append(color)
        tamano_puntos.append(tamanio)
        
        # Calculo de las condiciones iniciales de los cuerpos
        if r == 0.0: # Si la distancia del cuerpo con respecto al Sol es cero, es el Sol
            vec = [0.0, 0.0, 0.0, 0.0]
        elif r < 0.0: # Si la distancia es negativa, el cuerpo inicia en el eje X negativo
            x = r
            y = 0.0
            # Velocidad: Hacia abajo (vy) para órbita anti-horaria
            # Formula de velocidad circular: v = sqrt(GM / r)
            # Como usamos GM = 4*pi^2, entonces v = 2*pi / sqrt(r)
            vx = 0.0
            vy = -np.sqrt(G/abs(r)) # M = 1, pues respecto al Sol
            
            vec = [x, y, vx, vy] # Vector de uno de los cuerpos
        else:
            x = r
            y = 0.0
            # Velocidad: Hacia arriba (vy) para órbita anti-horaria
            # Formula de velocidad circular: v = sqrt(GM / r)
            # Como usamos GM = 4*pi^2, entonces v = 2*pi / sqrt(r)
            vx = 0.0
            vy = np.sqrt(G/r) # M = 1, pues respecto al Sol
            
            vec = [x, y, vx, vy] # Vector de uno de los cuerpos
        
        estado_inicial_lista.extend(vec) # Agrego el vector del cuerpo al vector de estados del sistema
    
    return estado_inicial_lista, masas, nombres, colores, tamano_puntos


# Función auxiliar para convertir vector plano a matriz (N, 4)
def state_to_matrix(estado, n):
    return np.reshape(estado, (n, 4))

# Ecuaciones diferenciales del sistema de N cuerpos
def EDO_Sistema(t, estado, masas):
    #Convierto el vector de estados del sistema en una matriz para facilitar cálculos
    n_cuerpos = len(estado) // 4 
    cuerpos = state_to_matrix(estado, n_cuerpos)
    
    # Matriz de las derivadas que devuelve el sistema
    derivadas = np.zeros_like(cuerpos)
    
    # Bucle para cada cuerpo i
    for i in range(n_cuerpos):
        xi, yi, vxi, vyi = cuerpos[i] # Obtengo la posicion y la velocidad del cuerpo
        
        # Las derivadas de posición son simplemente las velocidades
        derivadas[i, 0] = vxi # dx/dt
        derivadas[i, 1] = vyi # dy/dt
        
        # Para la aceleración, hay que sumar las fuerzas de los cuerpos de todo el sistema, excluyendo el mismo cuerpo
        ax, ay = 0.0, 0.0
        for j in range(n_cuerpos):
            if i == j: continue # No calcular fuerza sobre sí mismo
            
            xj, yj, _, _ = cuerpos[j] # Posicion del cuerpo j
            mj = masas[j] # Masa del cuerpo j
            
            # Distancia entre cuerpo i y cuerpo j
            rx = xj - xi
            ry = yj - yi
            r = np.sqrt(rx**2 + ry**2)
            
            # Ley de Gravitación: F = G * m1 * m2 / r^3 * vector_r
            # Aceleración (F/m1) = G * m2 / r^3 * vector_r
            factor = G * mj / (r**3)
            ax += factor * rx
            ay += factor * ry

        # Aceleraciones
        derivadas[i, 2] = ax # dvx/dt
        derivadas[i, 3] = ay # dvy/dt
        
    # Transformo la matriz de las derivadas en un vector simple
    return derivadas.flatten()

def Generar_Orbita(ax, nombres, colores, tamano_puntos):
    n_cuerpos = len(nombres)
    lineas = []  # Lista de líneas (rastro de cada cuerpo)
    puntos = []  # Lista de puntos (posición actual de cada cuerpo)
    for i in range(n_cuerpos):
        # Rastro (línea)
        ln, = ax.plot([], [], '--', color=colores[i], lw=1, alpha=0.5)
        lineas.append(ln)
        # Cuerpo actual (punto)
        pt, = ax.plot([], [], 'o', color=colores[i], markersize=tamano_puntos[i], label=nombres[i])
        puntos.append(pt)
    
    return lineas, puntos


# Función de generación de los frames del video del sistema
def update(frame, n_cuerpos, lineas, puntos, ys_radau, ts_radau, texto_tiempo):
    # 'frame' es el índice del paso de tiempo actual (0, 1, 2... n)
    for i in range(n_cuerpos):
        # Extraemos los datos de este cuerpo
        # La columna x es i*4, la columna y es i*4 + 1
        idx_x = i * 4
        idx_y = i * 4 + 1
        
        # 1. Actualizar el rastro (Toda la historia hasta el frame actual)
        # Si quieres que el rastro sea infinito:
        x_hist = ys_radau[:frame, idx_x]
        y_hist = ys_radau[:frame, idx_y]
        
        # Opcional: Si quieres un rastro corto (tipo "cola de cometa"), usa:
        # inicio = max(0, frame - 50)
        # x_hist = ys_radau[inicio:frame, idx_x]
        # y_hist = ys_radau[inicio:frame, idx_y]

        lineas[i].set_data(x_hist, y_hist)
        
        # 2. Actualizar la posición del planeta (Punto actual)
        x_actual = ys_radau[frame, idx_x]
        y_actual = ys_radau[frame, idx_y]
        puntos[i].set_data([x_actual], [y_actual]) # set_data espera listas/arrays
    
    t_actual = ts_radau[frame]
    texto_tiempo.set_text(f'Tiempo: {t_actual:.2f} años')

    return lineas + puntos + [texto_tiempo]


def Calcular_Energia_Sistema(ys, masas):
    n_pasos = ys.shape[0]
    n_cuerpos = len(masas)
    
    energia_total = []
    
    # Iteramos sobre cada instante de tiempo (frame)
    for paso in range(n_pasos):
        estado_actual = ys[paso]
        
        # 1. Energía Cinética (T)
        T = 0
        for i in range(n_cuerpos):
            idx = i * 4
            vx = estado_actual[idx + 2]
            vy = estado_actual[idx + 3]
            v_sq = vx**2 + vy**2
            T += 0.5 * masas[i] * v_sq
            
        # 2. Energía Potencial (U)
        U = 0
        for i in range(n_cuerpos):
            for j in range(i + 1, n_cuerpos): # j > i para no repetir pares ni auto-interacción
                # Posición cuerpo i
                xi, yi = estado_actual[i*4], estado_actual[i*4+1]
                # Posición cuerpo j
                xj, yj = estado_actual[j*4], estado_actual[j*4+1]
                
                # Distancia
                dist = np.sqrt((xj - xi)**2 + (yj - yi)**2)
                
                # U = -G * m1 * m2 / r
                U -= G * masas[i] * masas[j] / dist
                
        energia_total.append(T + U)
        
    return np.array(energia_total)