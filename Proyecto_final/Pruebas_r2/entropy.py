import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
    total_E = total_energy(velocities)
    if total_E > 0:
        return np.log(2 * np.pi * np.exp(1) * total_E / (3 * N**3))
    else:
        return -np.inf  # Valor de marcador que filtraremos más adelante

# Almacenar la entropía en cada paso de Monte Carlo
entropies = []

# Algoritmo de Monte Carlo
num_steps = 5000  # Número de pasos de Monte Carlo
intervalo_entropia = 10  # Intervalo para calcular la entropía
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
    
    # Calcular la entropía en intervalos específicos
    if step % intervalo_entropia == 0:
        entropies.append(entropy(velocities_initial))

# Filtrar los valores inválidos (inf o -inf)
x_data = np.arange(1, len(entropies) + 1) * intervalo_entropia
y_data = np.array(entropies)

valid_indices = np.isfinite(y_data)
x_data = x_data[valid_indices]
y_data = y_data[valid_indices]

# Definir la función de ajuste con dos términos exponenciales
def double_exponential_func(x, S0, A1, k1, A2, k2):
    return S0 + A1 * (1 - np.exp(-k1 * x)) + A2 * (1 - np.exp(-k2 * x))

# Realizar el ajuste
popt, pcov = curve_fit(double_exponential_func, x_data, y_data)

# Calcular el R^2
residuals = y_data - double_exponential_func(x_data, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r_squared = 1 - (ss_res / ss_tot)

# Graficar la entropía en función del tiempo y el ajuste
plt.plot(x_data, y_data, label='Entropía simulada')
plt.plot(x_data, double_exponential_func(x_data, *popt), 'r--', label='Ajuste $S_0 + A_1(1 - e^{-k_1t}) + A_2(1 - e^{-k_2t})$')
plt.xlabel('Tiempo de simulación')
plt.ylabel('Entropía')
plt.title(f'Entropía vs. Tiempo (R² = {r_squared:.4f})')
plt.legend()
plt.show()

# Imprimir el valor de R^2
print(f"R² del ajuste: {r_squared:.4f}")
