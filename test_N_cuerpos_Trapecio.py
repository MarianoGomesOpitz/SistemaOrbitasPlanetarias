import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy.integrate import solve_ivp
import funciones as fn

# Aumentar el limite a 100 MB (o lo que necesites)
plt.rcParams['animation.embed_limit'] = 500.0


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



###################################################################################


# Parámetros de simulacion
t_ini = 0.0 
h = 0.0001   # Paso de tiempos (en años terrestres)
n_pasos = 100000 # Cantidad de pasos de la simulacion
# Para saber cuantos años terrestres se simulan, hacer h * n_pasos
t_fin = t_ini + n_pasos * h
t_eval = np.linspace(t_ini, t_fin, n_pasos + 1)

####################################################################################

print(f"Sistema cargado con {len(nombres)} cuerpos.\n")
print(f"\nTiempo de simulacion: {n_pasos*h} años terrestres\n")
for i in range(n_cuerpos):
    print(f"Cuerpo {i+1}: {nombres[i]}, Masa: {masas[i]:.2e} M_sol")
    print(f"Posicion inicial: ({y0_sistema[i*4]:.2f}, {y0_sistema[i*4+1]:.2f}) AU")
    print(f"Velocidad orbital inicial: ({y0_sistema[i*4+2]:.2f}, {y0_sistema[i*4+3]:.2f}) AU/anio\n")

# Obtengo la aproximacion de RK4 del sistema
print("Calculando simulacion...")
ts_rk4, ys_rk4 = fn.RK2_Trapecio(fn.EDO_Sistema, t_ini, y0_sistema, h, n_pasos, masas)


print("Simulacion calculada.")

# Creacion de la visualizacion de resultados
fig, ax = plt.subplots(figsize=(8, 8))

#######################################################################################

# Limite del sistema, es la distancia desde el Sol hasta el borde del cuadro, en AU
limite = list(datos_planetas.values())[-1]['r'] + 1.0

#######################################################################################

# Definicion de los bordes del cuadro
ax.set_xlim(-limite, limite)
ax.set_ylim(-limite, limite)

ax.set_aspect('equal') # Para que las orbitas sean circulares, no ovaladas
ax.grid(True)
ax.set_title(f"Simulacion N-Cuerpos (RK4) con h={h}, pasos={n_pasos}, tiempo total={n_pasos*h} años terrestres")
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
salto = 10 
indices_frames = range(0, len(ts_rk4), salto)

#Creacion del "video"
ani = FuncAnimation(fig, fn.update, frames=indices_frames, interval=20, blit=True, fargs=(n_cuerpos, lineas, puntos, ys_rk4, ts_rk4, texto_tiempo))

plt.show()


# --- VALIDACION DE LA SIMULACION: CONSERVACION DE LA ENERGIA ---



# --- EJECUTAR EL CÁLCULO ---
print("Verificando conservacion de energia...")
Energias = fn.Calcular_Energia_Sistema(ys_rk4, masas)

# Calculamos el error relativo: (E_actual - E_inicial) / E_inicial
# Esto nos dice qué porcentaje de energia se "perdio" o "creo" falsamente
E0 = Energias[0]
error_relativo = (Energias - E0) / np.abs(E0)

# --- GRAFICAR ---
plt.figure(figsize=(10, 5))
plt.plot(ts_rk4, error_relativo, label='Error Relativo de Energia (RK4)')
plt.title("Validacion: Conservacion de la Energia Total")
plt.xlabel("Tiempo (Años)")
plt.ylabel("Error Relativo de Energia")
plt.grid(True)
plt.legend()

# Forzamos notacion cientifica para ver la magnitud real
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

plt.show()

# Resultado numérico
max_error = np.max(np.abs(error_relativo))
print(f"Error maximo relativo: {max_error:.2e}")
if max_error < 1e-4:
    print("RESULTADO: La simulacion es FiSICAMENTE ROBUSTA.")
else:
    print("RESULTADO: Hay una deriva de energia significativa.")

# Usar en lugar del "show" para Jupyter Notebooks
#plt.close() # Evita que se muestre una imagen estática extra vacia
#HTML(ani.to_jshtml())

# Guardar como GIF
#ani.save('orbita_solar.gif', writer='pillow', fps=30)

# O Guardar como MP4 (Mejor calidad)
#ani.save('orbita_solar.mp4', writer='ffmpeg', fps=30)