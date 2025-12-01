import numpy as np
import matplotlib.pyplot as plt

pasos = np.array([0.1, 0.01, 0.001, 0.0001])
tolerancias = np.array([1e-3, 1e-6, 1e-9, 1e-12])

tiempo_Euler = np.array([0.18, 1.94, 18.07, 179.77])
tiempo_Trapecio = np.array([0.37, 3.74, 35.31, 353.55])
tiempo_RK4 = np.array([0.75, 7.65, 75.88, 784.64])
tiempo_RK45 = np.array([1.48, 14.83, 50.41, 53.61])
tiempo_Radau = np.array([39.57, 84.68, 288.68, 312.11])

fig = plt.figure(figsize=(25, 20))

gs = fig.add_gridspec(2, 6)

ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[0, 2:4])
ax3 = fig.add_subplot(gs[0, 4:6])

ax4 = fig.add_subplot(gs[1, 0:3])
ax5 = fig.add_subplot(gs[1, 3:6])


#plt.subplot(2, 3, 1)
ax1.semilogx(pasos, tiempo_Euler, marker='o', label='Euler', color='orange')
ax1.grid(True)
ax1.set_title('Tiempos de ejecución de Euler')
ax1.set_xlabel('Tamaño del paso')
ax1.set_ylabel('Tiempo (s)')

#plt.subplot(2, 3, 2)
ax2.semilogx(pasos, tiempo_Trapecio, marker='o', label='Trapecio', color='green')
ax2.grid(True)
ax2.set_title('Tiempos de ejecución de Trapecio')
ax2.set_xlabel('Tamaño del paso')
ax2.set_ylabel('Tiempo (s)')

#plt.subplot(2, 3, 3)
ax3.semilogx(pasos, tiempo_RK4, marker='o', label='RK4', color='blue')
ax3.grid(True)
ax3.set_title('Tiempos de ejecución de RK4')
ax3.set_xlabel('Tamaño del paso')
ax3.set_ylabel('Tiempo (s)')

#plt.subplot(2, 3, 4)
ax4.semilogx(tolerancias, tiempo_RK45, marker='o', label='RK45', color='purple')
ax4.grid(True)
ax4.set_title('Tiempos de ejecución de RK45')
ax4.set_xlabel('Tamaño del paso')
ax4.set_ylabel('Tiempo (s)')

#plt.subplot(2, 3, 5)
ax5.semilogx(tolerancias, tiempo_Radau, marker='o', label='Radau', color='red')
ax5.grid(True)
ax5.set_title('Tiempos de ejecución de Radau')
ax5.set_xlabel('Tamaño del paso')
ax5.set_ylabel('Tiempo (s)')

fig.subplots_adjust(hspace=0.4, wspace=0.5)

plt.show()