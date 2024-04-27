# File: p3.py
# Date: 23/04/2024

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import poisson
from scipy.optimize import curve_fit
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import norm

##### Aire_poisson = listo

# Cargar los datos desde el archivo CSV
data = pd.read_csv('radiacion.csv')

# Ajustar la distribución de Poisson a los datos
def poisson_fit(x, mu):
    return poisson.pmf(x, mu)

# Estimar el parámetro lambda (mu) para la distribución de Poisson
mu_estimado = data['aire'].mean()

# Generar valores de x para la distribución de Poisson
x = np.arange(0, max(data['aire']) + 1)

# Realizar el ajuste de la distribución de Poisson a los datos
popt, pcov = curve_fit(poisson_fit, x, np.histogram(data['aire'], bins=np.arange(0, max(data['aire']) + 2))[0], p0=[mu_estimado])

# Crear la figura con Plotly
fig_aire_poisson = go.Figure()

# Agregar histograma de datos
fig_aire_poisson.add_trace(go.Histogram(x=data['aire'], histnorm='probability density', name='Datos'))

# Agregar línea de ajuste de Poisson
fig_aire_poisson.add_trace(go.Scatter(x=x, y=poisson_fit(x, *popt), mode='lines', name=f'Ajuste de Poisson ($\mu$={popt[0]:.2f})'))

# Configurar diseño de la figura
fig_aire_poisson.update_layout(title='Histograma y ajuste de distribución de Poisson',
                  xaxis_title='Valores',
                  yaxis_title='Frecuencia',
                  legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                  bargap=0.1,
                  bargroupgap=0.05,
                  template='plotly_white')



# Mostrar la figura en Streamlit
st.title("Gráfica decaimiento del Aire con distribución de Poisson")
st.plotly_chart(fig_aire_poisson)







########### PRUEBA DE CHI^2 ###########



# Leer el archivo CSV
df = pd.read_csv('radiacion.csv')

# Obtener los datos de la columna 'Aire'
datos_aire = df['aire']

# Definir los límites de los intervalos del histograma
bin_edges = np.histogram_bin_edges(datos_aire, bins=20)

# Calcular las frecuencias observadas
frec_obs = np.histogram(datos_aire, bins=bin_edges)[0]

# Calcular la media de los datos
mu = datos_aire.mean()

# Calcular las frecuencias esperadas usando la distribución de Poisson
frec_esp = len(datos_aire) * np.diff(poisson.cdf(bin_edges, mu))

# Realizar la prueba de chi-cuadrado
chi_cuadrado, p_valor, grados_libertad = chi2_contingency([frec_obs, frec_esp])[:3]

# Streamlit: Agregar entrada de texto para el umbral
umbral_input = 0.95
umbral = chi2.ppf(float(umbral_input), grados_libertad)

st.subheader("Prueba del chi-cuadrado")


# Imprimir los resultados
st.write("Valor chi-cuadrado:", chi_cuadrado)
st.write("P-valor:", p_valor)
st.write("Grados de libertad:", grados_libertad)

# Determinar si el ajuste es adecuado
if chi_cuadrado < umbral:
    st.write("El ajuste parece ser adecuado.")
else:
    st.write("El ajuste no parece ser adecuado.")





####### AIRE - GAUSS #####

# Calcular los parámetros de la distribución gaussiana
mu_2, std = norm.fit(datos_aire)

# Calcular la distribución de la función de densidad de probabilidad gaussiana
xmin, xmax = min(datos_aire), max(datos_aire)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_2, std)

# Crear el histograma usando Plotly
histograma = go.Histogram(x=datos_aire, nbinsx=20, histnorm='probability density', opacity=0.7, marker=dict(color='blue', line=dict(color='black', width=1)), name='Datos de radiación')

# Crear la distribución de Gauss ajustada
linea_gaussiana = go.Scatter(x=x, y=p, mode='lines', line=dict(color='black', width=2), name='Ajuste Gaussiano')

# Crear la figura y añadir los trazos
fig_aire_gauss = go.Figure()
fig_aire_gauss.add_trace(histograma)
fig_aire_gauss.add_trace(linea_gaussiana)

# Configurar el diseño del gráfico
fig_aire_gauss.update_layout(title='Histograma de datos de radiación y ajuste de Gauss', xaxis=dict(title='Aire'), yaxis=dict(title='Densidad de probabilidad'), legend=dict(x=0.7, y=1))

# Mostrar el gráfico en Streamlit

st.title("Gráfica decaimiento del Aire con distribución de Guass ")
st.plotly_chart(fig_aire_gauss)


######## PRUEBA DE CHI-CUADRADO AIRE-GAUSS #######


def prueba_chi_cuadrado():
    # Leer el archivo CSV
    df = pd.read_csv('radiacion.csv')

    # Obtener los datos de la columna 'Aire'
    datos_aire = df['aire']

    # Definir los límites de los intervalos del histograma
    bin_edges = np.histogram_bin_edges(datos_aire, bins=20)

    # Calcular las frecuencias observadas
    frec_obs = np.histogram(datos_aire, bins=bin_edges)[0]

    # Calcular los parámetros de la distribución gaussiana
    mu, std = norm.fit(datos_aire)

    # Definir el rango límite basado en la distribución gaussiana
    limite_inferior = mu - 3 * std
    limite_superior = mu + 3 * std

    # Filtrar los bin_edges dentro del rango límite
    bin_edges_filtrados = bin_edges[(bin_edges >= limite_inferior) & (bin_edges <= limite_superior)]

    # Calcular las frecuencias esperadas usando la distribución gaussiana
    frec_esp = len(datos_aire) * np.diff(norm.cdf(bin_edges_filtrados, mu, std))

    # Realizar la prueba de chi-cuadrado
    chi_cuadrado, p_valor = chi2_contingency([frec_obs[:len(bin_edges_filtrados)-1], frec_esp])[:2]


    st.subheader("Prueba de Chi-cuadrado")

    # Mostrar los resultados en Streamlit
    st.write("Valor chi-cuadrado:", chi_cuadrado)
    st.write("P-valor:", p_valor)

    # Determinar si el ajuste es adecuado
    umbral_aire_gauss = chi2.ppf(0.95, len(bin_edges_filtrados) - 1)
    if chi_cuadrado < umbral_aire_gauss:
        st.write("El ajuste parece ser adecuado.")
    else:
        st.write("El ajuste no parece ser adecuado.")

if __name__ == '__main__':
    prueba_chi_cuadrado()






































