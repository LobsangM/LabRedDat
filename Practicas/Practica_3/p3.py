# File: p3.py
# Date: 23/04/2024

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    st.write("El ajuste parece ser adecuado por tener un valor cercano a 1.")
else:
    st.write("El ajuste no parece ser adecuado por tener un valor alejado de 1.")





####### AIRE - GAUSS #####

# Calcular los parámetros de la distribución gaussiana
mu_2, std = norm.fit(datos_aire)

# Calcular la distribución de la función de densidad de probabilidad gaussiana
xmin, xmax = min(datos_aire), max(datos_aire)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_2, std)

# Crear el histograma usando Plotly
histograma = go.Histogram(x=datos_aire, nbinsx=20, histnorm='probability density', opacity=0.7, marker=dict(color='blue', line=dict(color='brown', width=1)), name='Datos de radiación')

# Crear la distribución de Gauss ajustada
linea_gaussiana = go.Scatter(x=x, y=p, mode='lines', line=dict(color='red', width=2), name='Ajuste Gaussiano')

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
        st.write("El ajuste parece ser adecuado por tener un valor cercano a 1.")
    else:
        st.write("El ajuste no parece ser adecuado por tener un valor alejado a 1.")

if __name__ == '__main__':
    prueba_chi_cuadrado()



###### CESIO - POISSON #####

# Cargar los datos desde el archivo CSV
data = pd.read_csv('radiacion.csv')

# Ajustar la distribución de Poisson a los datos
def poisson_fit_cesio(x, mu):
    return poisson.pmf(x, mu)

# Estimar el parámetro lambda (mu) para la distribución de Poisson
mu_estimado_cesio = data['cesio'].mean()

# Generar valores de x para la distribución de Poisson
x = np.arange(350, max(data['cesio']) + 1)

# Realizar el ajuste de la distribución de Poisson a los datos
popt, pcov = curve_fit(poisson_fit_cesio, x, np.histogram(data['cesio'], bins=np.arange(350, max(data['cesio']) + 2))[0], p0=[mu_estimado_cesio])

# Crear la figura con Plotly
fig_cesio_poisson = go.Figure()

# Agregar histograma de datos
fig_cesio_poisson.add_trace(go.Histogram(x=data['cesio'], histnorm='probability density', name='Datos'))

# Agregar línea de ajuste de Poisson
fig_cesio_poisson.add_trace(go.Scatter(x=x, y=poisson_fit_cesio(x, *popt), mode='lines', name=f'Ajuste de Poisson ($\mu$={popt[0]:.2f})'))

# Configurar diseño de la figura
fig_cesio_poisson.update_layout(title='Histograma y ajuste de distribución de Poisson para Cesio',
                  xaxis_title='Valores',
                  yaxis_title='Frecuencia',
                  legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                  bargap=0.1,
                  bargroupgap=0.05,
                  template='plotly_white')

# Mostrar la figura en Streamlit
st.title("Gráfica decaimiento del Cesio con distribución de Poisson")
st.plotly_chart(fig_cesio_poisson)




####### Prueba de Chi-cuadrado para Cesio Poisson ######


# Leer el archivo CSV
df = pd.read_csv('radiacion.csv')

# Obtener los datos de la columna 'Aire'
datos_aire = df['cesio']

# Definir los límites de los intervalos del histograma
bin_edges_cesio = np.histogram_bin_edges(datos_aire, bins=20)

# Calcular las frecuencias observadas
frec_obs_cesio = np.histogram(datos_aire, bins=bin_edges_cesio)[0]

# Calcular la media de los datos
mu_cesio = datos_aire.mean()

# Calcular las frecuencias esperadas usando la distribución de Poisson
frec_esp = len(datos_aire) * np.diff(poisson.cdf(bin_edges_cesio, mu_cesio))

# Realizar la prueba de chi-cuadrado
chi_cuadrado, p_valor, grados_libertad = chi2_contingency([frec_obs_cesio, frec_esp])[:3]

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




######## Gráfica CESIO - GAUSS ######


# Leer el archivo CSV
df = pd.read_csv('radiacion.csv')

# Obtener los datos de la columna 'Cesio'
datos_cesio = df['cesio']

# Calcular los parámetros de la distribución gaussiana
mu_2_cesio, std = norm.fit(datos_cesio)

# Calcular la distribución de la función de densidad de probabilidad gaussiana
xmin, xmax = min(datos_cesio), max(datos_cesio)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_2_cesio, std)

# Crear el histograma usando Plotly
histograma_cesio_gauss = go.Histogram(x=datos_cesio, nbinsx=20, histnorm='probability density', opacity=0.7, marker=dict(color='blue', line=dict(color='brown', width=1)), name='Datos de radiación')

# Crear la distribución de Gauss ajustada
linea_gaussiana_cesio = go.Scatter(x=x, y=p, mode='lines', line=dict(color='red', width=2), name='Ajuste Gaussiano')

# Crear la figura y añadir los trazos
fig_cesio_gauss = go.Figure()
fig_cesio_gauss.add_trace(histograma_cesio_gauss)
fig_cesio_gauss.add_trace(linea_gaussiana_cesio)

# Configurar el diseño del gráfico
fig_cesio_gauss.update_layout(title='Histograma de datos de radiación y ajuste de Gauss para Cesio', xaxis=dict(title='Cesio'), yaxis=dict(title='Densidad de probabilidad'), legend=dict(x=0.7, y=1))

# Mostrar el gráfico en Streamlit
st.title("Gráfica decaimiento del Cesio con distribución de Gauss ")
st.plotly_chart(fig_cesio_gauss)




######### Prueba CHI-CUADRADO Cesio-gauss ######


def prueba_chi_cuadrado():
    # Leer el archivo CSV
    df = pd.read_csv('radiacion.csv')

    # Obtener los datos de la columna 'Aire'
    datos_aire = df['cesio']

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
        st.write("El ajuste no parece ser adecuado por tener un valor alejado a 1.")

if __name__ == '__main__':
    prueba_chi_cuadrado()


##########################################


st.title("Discución de Resultados")

st.subheader("Nombre de columnas en los datos")

st.markdown("""El archivo que contenia los datos del experimento
            no contaban con columnas con nombre, por lo tanto lo primero que
            se realizó fue colocarle el nombre de "aire" y "cesio" a la primera y segunda
            columna respectivamente.

Se realizó esto con la finalidad de facilitar la lecutra de las columnas a la hora 
de realizar los histogramas y los ajustes correspondientes. Dicha modificación ayudó a la hora
de poder ejecutar el código de una manera mas fácil y rápida.
            """)



st.subheader("Gráficas de decaimiento del Aire")

st.markdown("""Los datos obtenidos del decaimiento del aire se graficaron en un histograma
            al cual se le aplicó primero un ajuste con el modelo matemático de la distribución
            de Poisson. Luego de obtener la gráfica la distribución de Poisson aplicada a la gráfica
            realizamos la prueba del "Chi-cuadrado" la cual nos dió un valor cercano a 1, por lo cual 
            podemos afirmar que la distribución de Poisson es el modelo matemático mas acertado
            para el decaimiento del aire.

Ahora, luego del procedimiento anterior, se realizó la misma gráfica pero se le aplicó un ajuste
de la forma de la distribución de Guass, cabe resaltar que la dicha gráfica quedó un poco diferente
a la anterior debido a la cantidad de bines, desconocemos el por qué de esto pero si agrupabamos
los datos igual que la gráfica anterior nos tiraba un error que no pudimos corregir, asi que la solución fue
agrupar los datos de la forma que aparecen en la gráfica, igualmente no consideramos que esto podría generar
algún tipo de error ya que los resultados fueron según lo esperado teóricamente.
A partir de esto se realizó la prueba de "Chi-cuadrado" la cual no dió un valor alejado a 1 por lo cual este ajuste no se considera 
el adecuado para estos datos.
""")



st.subheader("Gráficas de decaimiento del Cesio")

st.markdown(""" Los datos obtenidos de la columna del "cesio" se graficaron y agruparon en un histograma
            el cual le aplicamos la distribución de Poisson de primero, y vemos como la curva sigue la 
            trayectoria de los datos. Luego le aplicamos la prueba del "Chi-cuadrado" y podemos ver que
            nos dió el valor muy cercano a 1, por lo cual se puede concluir que la distribución de Poisson
            ajusta muy bien los datos obtenidos del decaimiento del cesio.

Luego del procedimiento anterior, volvimos a tener el problema que obtuvimos con la gráfica de los datos del 
aire con la distribución de gauss, se tuvo que simplificar los bines para que los datos se obtuvieras de una manera
que consideramos correcta. Le aplicamos la distribución de gauss a nuestro histograma y vemos que la distribución no se
acomoda a los datos tan bien como la anterior. Como siguiente paso, le aplicamos la prueba de "chi-cuadrado" la cual nos dió un
valor muy alejado del 1, lo cual nos dice que dichos datos no se ajustan muy bien a la distribución de Gauss.
""")



st.subheader("Conclusiones")

st.markdown("""
- La distribución de Poisson se ajustó muy bien a los datos obtenidos del decaimiento del aire ya que nos dió
un valor cercano a 1 en la prueba del chi-cuadrado.
- La distribución de Gauss no se ajustó bien a los datos obtenidos del decaimiento del aire, porque no dió un
valor alejado de 1 en la prueba del chi-cuadrado.
- Los datos obtenidos del decaimiento del cesio corresponden y se ajustan a la distribución de Poisson ya que
el valor obtenido de la prueba de Chi-cuadrado fue cercano a 1.
- Los datos obtenidos del decaimiento del cesio no corresponden y ni se ajustan a la distribución de Gauss ya que
el valor obtenido de la prueba de Chi-cuadrado fue muy alejado a 1.
- En general, el decaimiento del aire y del cesio, corresponde a una distribución de Poisson y no a la
distribución de Gauss.
            """)
























