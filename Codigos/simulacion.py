import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from IPython.display import HTML
from scipy.integrate import solve_ivp
import funciones as fn

# Limite de tamaño para embeber animaciones en Jupyter Notebooks
# plt.rcParams['animation.embed_limit'] = 500.0


# Base de datos del sistema
#    m: Masa del cuerpo en relacion al Sol
#    r: Distancia del Sol al cuerpo, en AU
#    c: Color con el que aparecera el cuerpo y su orbita en el sistema, no afecta los calculos
#    t: Tamaño con el que aparecerá en el sistema, no afecta los calculos
datos_planetas = {
    'Sol':      {'m': 1.0,       'r': 0.0,    'c': 'gold',    't': 10},
    'Mercurio': {'m': 1.66e-7,   'r': 0.39,   'c': 'grey',    't': 5},
    'Venus':    {'m': 2.45e-6,   'r': 0.72,   'c': 'orange',  't': 6},
    'Tierra':   {'m': 3.00e-6,   'r': 1.00,   'c': 'blue',    't': 8},
    'Marte':    {'m': 3.23e-7,   'r': 1.52,   'c': 'red',     't': 7},
    #'Jupiter':  {'m': 9.54e-4,   'r': 5.20,   'c': 'purple',  't': 7},
    #'Saturno':  {'m': 2.86e-4,   'r': 9.58,   'c': 'olive',   't': 6},
    #'Urano':    {'m': 4.36e-5,   'r': 19.22,  'c': 'blue',    't': 5},
    #'Neptuno':  {'m': 5.15e-5,   'r': 30.05,  'c': 'cyan',    't': 4}
}

# Número de cuerpo involucrados en el sistema
n_cuerpos = len(datos_planetas)

# Listas de datos para llenar del sistema a medida que se realizan los cálculos
masas = []
nombres = []
colores = []
tamano_puntos = []
# Vector de estados de los cuerpos: Posiciones y Valocidades de todos los cuerpos
estado_inicial_lista = [] 

# Generacion de los vectores del sistema
estado_inicial_lista, masas, nombres, colores, tamano_puntos = fn.Generar_Sistema(datos_planetas)

# Convierto a array de numpy para la simulacion
y0_sistema = np.array(estado_inicial_lista)
masas = np.array(masas) # Convierto masas a array también






#########################################################################################################

anios = float(input("Ingrese la cantidad de años terrestres a simular (ej: 100): "))

metodos_fijo = ['Euler', 'Trapecio', 'RK4']
metodos_variable = ['RK45', 'Radau']
metodo_elegido = ""
while metodo_elegido not in metodos_fijo + metodos_variable:
    metodo_elegido = input(f"\nIngrese el método de paso fijo a utilizar ({metodos_fijo + metodos_variable}): ")

# Parámetros de simulacion
t_ini = 0.0 

h = 0.001   # Valor por defecto, si se usa paso fijo
tolerancia = 1e-6  # Valor por defecto, si se usa paso variable

if metodo_elegido in metodos_fijo: # Si se selecciona paso fijo, se elige un h
    h = float(input("Ingrese el valor del paso de tiempo h (en años terrestres, ej: 0.0001): "))
else: # Si se selecciona paso variable, se elige una tolerancia
    tolerancia = float(input("Ingrese la tolerancia relativa deseada (formato: 1e-3, 1e-9): "))

n_pasos = int(anios/h) # Cantidad de pasos de la simulacion
# Para saber cuantos años terrestres se simulan, hacer h * n_pasos
t_fin = t_ini + n_pasos * h
t_eval = np.linspace(t_ini, t_fin, n_pasos + 1)

###########################################################################################################

print(f"Sistema cargado con {len(nombres)} cuerpos.")
print(f"\nTiempo de simulacion: {n_pasos*h} años terrestres\n")
for i in range(n_cuerpos):
    print(f"Cuerpo {i+1}: {nombres[i]}, Masa: {masas[i]:.2e} M_sol")
    print(f"Posicion inicial: ({y0_sistema[i*4]:.2f}, {y0_sistema[i*4+1]:.2f}) AU")
    print(f"Velocidad orbital inicial: ({y0_sistema[i*4+2]:.2f}, {y0_sistema[i*4+3]:.2f}) AU/anio")




# Calculo de la simulacion
print("Calculando simulacion...")
inicio = time.time()

if metodo_elegido in metodos_fijo:
    match metodo_elegido:
        case 'Euler':
            ts, ys = fn.EulerExplicito(fn.EDO_Sistema, t_ini, y0_sistema, h, n_pasos, masas)
        case 'Trapecio':
            ts, ys = fn.RK2_Trapecio(fn.EDO_Sistema, t_ini, y0_sistema, h, n_pasos, masas)
        case 'RK4':
            ts, ys = fn.RungeKutta4(fn.EDO_Sistema, t_ini, y0_sistema, h, n_pasos, masas)
else:
    sol = solve_ivp(
    fun=fn.EDO_Sistema,       # Tu funcion de fisica
    t_span=(t_ini, t_fin),       # Intervalo de tiempo (inicio, fin)
    y0=y0_sistema,         # Estado inicial
    method=metodo_elegido,               # El método (Runge-Kutta-Fehlberg 4(5))
    t_eval=t_eval,               # Puntos donde guardar la solucion
    rtol=tolerancia,                   # Tolerancia relativa (Precision)
    atol=1e-9,                   # Tolerancia absoluta
    args=(masas,)                   # Argumentos adicionales para la funcion
    )
    ts = sol.t
    ys = sol.y.T


fin = time.time()
print(f"\nTiempo de calculo de {metodo_elegido}: {fin - inicio:.2f} segundos\n")


print("Simulacion calculada.")

# Creacion de la visualizacion de resultados
fig, ax = plt.subplots(figsize=(8, 8))

# Limite del sistema, es la distancia desde el Sol hasta el borde del cuadro, en AU
# Toma el ultimo cuerpo en el diccionario de datos_planetas y le suma 1 AU de margen
limite = list(datos_planetas.values())[-1]['r'] + 1.0

#######################################################################################

# Definicion de los bordes del cuadro
ax.set_xlim(-limite, limite)
ax.set_ylim(-limite, limite)

ax.set_aspect('equal') # Para que las orbitas sean circulares, no ovaladas
ax.grid(True)
if metodo_elegido in metodos_fijo:
    ax.set_title(f"Simulacion N-Cuerpos ({metodo_elegido}) con h={h}, pasos={n_pasos}, tiempo total={n_pasos*h} años terrestres")
else:
    ax.set_title(f"Simulacion N-Cuerpos ({metodo_elegido}), rtol={tolerancia}, pasos={len(ts)}, tiempo total={n_pasos*h} años terrestres")
ax.set_facecolor("white")

lineas = []
puntos = []

# Generacion del cuerpo y la orbita que deja el mismo al moverse
lineas, puntos = fn.Generar_Orbita(ax, nombres, colores, tamano_puntos)


texto_tiempo = ax.text(0.05, 0.95, '', transform=ax.transAxes, color="black", fontsize=12)

ax.legend(loc="lower right", fontsize='small')




# frames: cuántos cuadros totales (usamos el largo de ts)
# interval: milisegundos entre cuadros (20ms = 50 fps)
# blit=True: optimizacion gráfica (solo redibuja lo que cambia)
print("Generando grafica (esto tomara un tiempo)...")
salto = int(1/(10*h))
indices_frames = range(0, len(ts), salto)

#Creacion del "video"
ani = FuncAnimation(fig, fn.update, frames=indices_frames, interval=20, blit=True, fargs=(n_cuerpos, lineas, puntos, ys, ts, texto_tiempo))

plt.show()


# --- VALIDACION DE LA SIMULACION: CONSERVACION DE LA ENERGIA ---

# --- EJECUTAR EL CÁLCULO ---
print("Verificando conservacion de energia...")
Energias = fn.Calcular_Energia_Sistema(ys, masas)

# Calculamos el error relativo: (E_actual - E_inicial) / E_inicial
# Esto nos dice qué porcentaje de energia se "perdio" o "creo" falsamente
E0 = Energias[0]
error_relativo = (Energias - E0) / np.abs(E0)

# --- GRAFICAR ---
plt.figure(figsize=(10, 5))
plt.plot(ts, error_relativo, label=f'Error Relativo de Energia ({metodo_elegido})')
if metodo_elegido in metodos_fijo:
    plt.title(f"Validacion: Conservacion de la Energia Total\nh={h}, pasos={n_pasos}, tiempo total={n_pasos*h} años terrestres")
else:
    plt.title(f"Validacion: Conservacion de la Energia Total\nrtol={tolerancia}, tiempo total={n_pasos*h} años terrestres")
plt.xlabel("Tiempo (Años)")
plt.ylabel("Error Relativo de Energia")
plt.grid(True)
plt.legend()


# Forzamos notacion cientifica para ver la magnitud real
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

plt.show()

# Si se esta en un Jupyter Notebook, se puede embeber la animacion directamente
#plt.close() # Evita que se muestre una imagen estática extra vacia
#HTML(ani.to_jshtml())


####### Grafica del error en escala logaritmica

plt.figure(figsize=(10, 6))
# Usamos semilogy: Eje Y logarítmico, Eje X lineal (tiempo)
plt.semilogy(ts, np.abs(error_relativo), label='f{metodo_elegido}')

# Opcional: Si comparas con {metodo_elegido}
# plt.semilogy(ts_euler, error_euler, label='{metodo_elegido} (Explícito)')

if metodo_elegido in metodos_fijo:
    plt.title(f"Evolución del Error de Energía de {metodo_elegido} \ncon un h={h} (Escala Logarítmica)")
else:
    plt.title(f"Evolución del Error de Energía de {metodo_elegido} \ncon una tolerancia relativa de {tolerancia} (Escala Logarítmica)")
plt.xlabel("Tiempo (Años)")
plt.ylabel("Log(Error Relativo)")
plt.grid(True, which="both", ls="-") # Grid para escala log
plt.legend()
plt.show()

# Resultado numérico
max_error = np.abs(error_relativo)[-1]
print(f"Error maximo relativo: {max_error:.2e}")

# Guardar como GIF
#print("Guardando animacion como GIF...")
#ani.save('D:/Videos/Euler_0.1.gif', writer='pillow', fps=30)
#print("Animacion guardada.")

# O Guardar como MP4 (Mejor calidad)
#print("Guardando animacion como video MP4...")
#ani.save('D:/Videos/Euler_0.0001.mp4', writer='ffmpeg', fps=30)