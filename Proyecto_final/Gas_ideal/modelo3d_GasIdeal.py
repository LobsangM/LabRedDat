import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del sistema
N = 8  # Número de partículas (número de partículas por lado del cubo)
V = 1.0  # Volumen del contenedor
L = np.cbrt(V)  # Longitud del lado del contenedor cúbico
T = 1.0  # Temperatura inicial
K = 0.0  # Energía cinética inicial (todas las partículas en reposo)

# Inicialización de las posiciones de las partículas en un cubo
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
z = np.linspace(0, L, N)
positions_initial = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

# Inicialización de las velocidades de las partículas (todas en reposo)
velocities_initial = np.zeros((N**3, 3))

# Función para calcular la energía total del sistema
def total_energy(velocities):
    return 0.5 * np.sum(velocities**2)

# Función para calcular la temperatura del sistema
def temperature(velocities):
    return 2 * total_energy(velocities) / (3 * N**3)

# Función para calcular la presión del sistema
def pressure(temperature):
    return N**3 / V * temperature

# Función para calcular la entropía del sistema
def entropy(velocities):
    return np.log(2 * np.pi * np.exp(1) * total_energy(velocities) / (3 * N**3))

# Función para graficar las posiciones de las partículas en 3D
def plot_positions_3d(positions, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Energía cinética inicial
initial_kinetic_energy = total_energy(velocities_initial)

# Algoritmo de Monte Carlo
num_steps = 5000  # Número de pasos de Monte Carlo
for step in range(num_steps):
    # Seleccionar una partícula aleatoria
    particle_index = np.random.randint(N**3)
    
    # Generar un movimiento aleatorio para la partícula seleccionada
    delta_position = np.random.uniform(-0.1, 0.1, size=(3,))
    new_position = positions_initial[particle_index] + delta_position
    
    # Verificar si el movimiento propuesto está dentro del contenedor
    if np.all(new_position >= 0) and np.all(new_position <= L):
        # Calcular el cambio en la energía cinética
        old_velocity = velocities_initial[particle_index]
        new_velocity = np.random.normal(0, np.sqrt(T), size=(3,))
        delta_kinetic_energy = 0.5 * np.sum(new_velocity**2) - 0.5 * np.sum(old_velocity**2)
        
        # Aceptar o rechazar el movimiento basado en el cambio en la energía
        if delta_kinetic_energy <= 0 or np.random.rand() < np.exp(-delta_kinetic_energy / T):
            positions_initial[particle_index] = new_position
            velocities_initial[particle_index] = new_velocity

# Energía cinética final
final_kinetic_energy = total_energy(velocities_initial)

# Entropía inicial
initial_entropy = entropy(velocities_initial)

# Calcular propiedades macroscópicas finales del sistema
final_temperature = temperature(velocities_initial)
final_pressure = pressure(final_temperature)

# Imprimir resultados
print("Temperatura final:", final_temperature)
print("Presión final:", final_pressure)
print("Energía cinética inicial:", initial_kinetic_energy)
print("Energía cinética final:", final_kinetic_energy)
print("Entropía inicial:", initial_entropy)

# Graficar las posiciones inicial y final de las partículas en 3D
plot_positions_3d(positions_initial, "Initial Position")

# Algoritmo de Monte Carlo para las posiciones finales
for step in range(num_steps):
    # Seleccionar una partícula aleatoria
    particle_index = np.random.randint(N**3)
    
    # Generar un movimiento aleatorio para la partícula seleccionada
    delta_position = np.random.uniform(-0.1, 0.1, size=(3,))
    new_position = positions_initial[particle_index] + delta_position
    
    # Verificar si el movimiento propuesto está dentro del contenedor
    if np.all(new_position >= 0) and np.all(new_position <= L):
        # Calcular el cambio en la energía cinética
        old_velocity = velocities_initial[particle_index]
        new_velocity = np.random.normal(0, np.sqrt(T), size=(3,))
        delta_kinetic_energy = 0.5 * np.sum(new_velocity**2) - 0.5 * np.sum(old_velocity**2)
        
        # Aceptar o rechazar el movimiento basado en el cambio en la energía
        if delta_kinetic_energy <= 0 or np.random.rand() < np.exp(-delta_kinetic_energy / T):
            positions_initial[particle_index] = new_position
            velocities_initial[particle_index] = new_velocity

# Graficar las posiciones finales de las partículas en 3D
plot_positions_3d(positions_initial, "Final Position")
