import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st

# Leer el archivo CSV
df = pd.read_csv('radiacion.csv')

# Obtener los datos de la columna 'Aire'
datos_aire = df['aire']

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
st.plotly_chart(fig_aire_gauss)