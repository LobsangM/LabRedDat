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

 
st.title('Decaimiento de Cesio-137')

# Fondo temático
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://i.pinimg.com/564x/c1/97/7d/c1977de3ed571607a33c3c3e59691e6b.jpg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)


input_style = """
<style>
input[type="text"] {
    background-color: transparent;
    color: #a19eae;  // This changes the text color inside the input box
}
div[data-baseweb="base-input"] {
    background-color: transparent !important;
}
[data-testid="stAppViewContainer"] {
    background-color: transparent !important;
}
</style>
"""
st.markdown(input_style, unsafe_allow_html=True)

st.subheader("¿Qué es la distribución de Poisson?")
st.markdown("""
La distribución de Poisson es una distribución de probabilidad discreta que describe la probabilidad de que ocurra un determinado número de eventos en un intervalo de tiempo o espacio, dado un valor promedio de ocurrencias. Es útil para modelar situaciones donde los eventos ocurren de manera independiente a una tasa constante en un intervalo de tiempo o espacio específico, como el número de llamadas telefónicas que recibe un centro de atención al cliente en una hora, el número de accidentes de tráfico en un tramo de carretera en un día o el número de partículas radiactivas emitidas por un material en un intervalo de tiempo.
            
La distribución de Poisson se caracteriza por un solo parámetro, denotado como lambda (λ), que representa la tasa promedio de ocurrencia de eventos en el intervalo considerado. La fórmula para la probabilidad de que ocurran exactamente k eventos en ese intervalo es:
""")

#Notación matemática usando latex
st.latex(r"P(x,n)=\frac{e^{-\mu}\mu^k}{k!}")
st.markdown("""
Donde:
            
- $P(X=k)$ es la probabilidad de que ocurran k eventos.

- e es la base del logaritmo natural
            
- $\mu$ es la tasa promedio de ocurrencia de eventos en el intervalo
            
- k  es el número de eventos que estamos interesados en calcular.
            
 La distribución de Poisson es especialmente útil cuando el número de eventos es grande y la probabilidad de que ocurra un evento en un intervalo de tiempo muy pequeño es muy pequeña. Además, la distribución de Poisson se relaciona con la distribución binomial cuando el número de ensayos es grande y la probabilidad de éxito es pequeña.           
""") 
#Distribucion Gaussiana
st.subheader("¿Qué es la distribución Gaussiana?")
st.markdown("""
La distribución Gaussiana, también conocida como distribución normal, es una de las distribuciones de probabilidad más importantes y ampliamente utilizadas en estadística y teoría de la probabilidad. Es especialmente útil debido a sus propiedades matemáticas y su capacidad para modelar una amplia variedad de fenómenos en la naturaleza y en diferentes campos de estudio.

La distribución Gaussiana se caracteriza por su forma de campana simétrica y su curva suave. La función de densidad de probabilidad de una distribución normal está dada por la fórmula:
""")

#Notación matemática usando latex
st.latex(r"f(x|\mu,\sigma )=\frac{1}{\sigma\sqrt{2\pi}}e^{\frac{-(x-\mu)^2}{2\sigma^2}}")
st.markdown("""
Donde:
            
- $x$ es la variable aleatoria.

- $\mu$ es la media de la distribución, que indica el valor central o la tendencia central de los datos
            
- $\sigma$ es la desviación estándar, que indica la dispersión o la variabilidad de los datos.
            
La distribución Gaussiana tiene las siguientes propiedades:

- Simetría: La distribución es simétrica en torno a su media.

- Media, Mediana y Moda: La media, la mediana y la moda de la distribución son iguales y están ubicadas en $\mu$

- Campana: La distribución tiene una forma de campana, con la mayoría de los valores concentrados cerca de la media y una dispersión decreciente a medida que nos alejamos de ella.

- Regla 68-95-99.7: Aproximadamente el 68% de los datos están dentro de una desviación estándar de la media, el 95% dentro de dos desviaciones estándar y el 99.7% dentro de tres desviaciones estándar.
""") 
#Chi-cuadrado

st.subheader("¿Qué Chi-cuadrado?")
st.markdown("""
La prueba de chi-cuadrado, también conocida como prueba de bondad de ajuste de chi-cuadrado, es una técnica estadística utilizada para determinar si existe una diferencia significativa entre las frecuencias observadas y las frecuencias esperadas en un conjunto de datos categóricos. Es una prueba no paramétrica, lo que significa que no requiere ninguna suposición sobre la distribución subyacente de los datos.
            
La prueba de chi-cuadrado se utiliza en una variedad de aplicaciones, como pruebas de independencia en tablas de contingencia, pruebas de bondad de ajuste para comparar una distribución observada con una teórica, y en análisis de regresión para evaluar la bondad de ajuste del modelo. Es una herramienta útil para analizar datos categóricos y determinar si hay una relación significativa entre las variables categóricas.
""")

#Contadorde Geiger
st.subheader("¿Qué es el contador de Geiger?")
st.markdown("""
El contador Geiger es un dispositivo utilizado para detectar la radiación ionizante, como los rayos alfa, beta y gamma. Consiste en un tubo lleno de gas (generalmente argón o helio) a baja presión, con un electrodo central y un electrodo externo que actúan como un condensador. Cuando una partícula cargada ioniza el gas dentro del tubo, se genera un pulso eléctrico detectable por el contador.

Cuando una partícula ionizante pasa a través del tubo, ioniza los átomos del gas, liberando electrones. Estos electrones, al acelerarse bajo la acción de un campo eléctrico, generan más ionizaciones en cascada, lo que produce una avalancha de electrones y iones positivos. Esto provoca un aumento rápido en la corriente eléctrica que puede ser detectado por el contador. El número de pulsos eléctricos registrados por el contador está relacionado con la intensidad de la radiación ionizante presente en el entorno.
            
Los contadores Geiger son ampliamente utilizados en aplicaciones de monitoreo de radiación, como en la industria nuclear, la medicina, la investigación científica y la protección radiológica. Son dispositivos portátiles, sensibles y relativamente económicos, lo que los hace ideales para la detección rápida y el monitoreo de la radiación en diferentes entornos.
""")




# Crear las pestañas
titulos_pestañas = ['Resumen del proyecto', 'Visualización de Datos', 'Discusión de Resuultados', 'Conclusiones', 'Referencias']
pestañas = st.tabs(titulos_pestañas)
 
# Agregar contenido a cada pestaña
with pestañas[0]:
    st.header('¿Qué se hizo en el Proyecto?')
    st.markdown("""
    Este reporte presenta un análisis detallado de las mediciones realizadas en clase, con el objetivo de implementar ajustes de las distribuciones de Poisson y Gaussiana. Además, se llevó a cabo la prueba de "chi-cuadrado" para cada ajuste, con el fin de determinar cuál de las distribuciones se adapta mejor al caso experimental.
    
   Procedimiento Experimental:
    - Se utilizaron mediciones realizadas en clase como datos experimentales.
                
    - Se implementaron ajustes de las distribuciones de Poisson y Gaussiana a los datos.
                
    - Se realizó la prueba de "chi-cuadrado" para cada ajuste.
                
    - Se compararon los resultados de las pruebas de "chi-cuadrado" para determinar la distribución que mejor se ajusta a los datos experimentales.

    """)
 
with pestañas[1]:
    st.header('Visualización de Datos')
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
    st.subheader("Gráfica decaimiento del Aire con distribución de Poisson")
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

    st.subheader("Gráfica decaimiento del Aire con distribución de Guass ")
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
    st.subheader("Gráfica decaimiento del Cesio con distribución de Poisson")
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
    histograma_cesio_gauss = go.Histogram(x=datos_cesio, nbinsx=20, histnorm='probability density', opacity=0.7, marker=dict(color='blue', line=dict(color='black', width=1)), name='Datos de radiación')

    # Crear la distribución de Gauss ajustada
    linea_gaussiana_cesio = go.Scatter(x=x, y=p, mode='lines', line=dict(color='black', width=2), name='Ajuste Gaussiano')

    # Crear la figura y añadir los trazos
    fig_cesio_gauss = go.Figure()
    fig_cesio_gauss.add_trace(histograma_cesio_gauss)
    fig_cesio_gauss.add_trace(linea_gaussiana_cesio)

    # Configurar el diseño del gráfico
    fig_cesio_gauss.update_layout(title='Histograma de datos de radiación y ajuste de Gauss para Cesio', xaxis=dict(title='Cesio'), yaxis=dict(title='Densidad de probabilidad'), legend=dict(x=0.7, y=1))

    # Mostrar el gráfico en Streamlit
    st.subheader("Gráfica decaimiento del Cesio con distribución de Gauss ")
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
            st.write("El ajuste no parece ser adecuado.")

    if __name__ == '__main__':
        prueba_chi_cuadrado()


    ##########################################
with pestañas[2]:
    st.title("Discusión de Resultados")

    st.subheader("Nombre de columnas en los datos")

    st.markdown("""
    Los datos obtenidos de la columna del "cesio" se graficaron y agruparon en un histograma
    el cual le aplicamos la distribución de Poisson de primero, y vemos como la curva sigue la 
    trayectoria de los datos. Luego le aplicamos la prueba del "Chi-cuadrado" y podemos ver que
    nos dió el valor muy cercano a 1, por lo cual se puede concluir que la distribución de Poisson
    ajusta muy bien los datos obtenidos del decaimiento del cesio.

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

    

with pestañas[3]:
    st.header('Conclusiones')
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

with pestañas[4]:
    st.header('Referencias')
    st.markdown("""
    -DISTRIBUCI�N DE POISSON. (s/f). Www.uv.es. Recuperado el 28 de abril de 2024, de https://www.uv.es/ceaces/base/modelos%20de%20probabilidad/poisson.htm
                
    -Física Nuclear y de Partículas y Estructura Nuclear. (s/f). Caracterización de un contador Geiger. Absorción de radiación por materiales. Ucm.es. Recuperado el 28 de abril de 2024, de http://nuclear.fis.ucm.es/LABORATORIO/guiones/Caracterizaci%F3ndeGeiger.pdf

    -Narvaez, M. (2022, mayo 4). Prueba de chi-cuadrado: ¿Qué es y cómo se realiza? QuestionPro. https://www.questionpro.com/blog/es/prueba-de-chi-cuadrado-de-pearson/

    -Ortega, C. (2023, agosto 3). Distribución gaussiana: Qué es y cuál es su importancia. QuestionPro. https://www.questionpro.com/blog/es/distribucion-gaussiana/

    -PRUEBA CHI-CUADRADO. (s/f). Www.ub.edu. Recuperado el 28 de abril de 2024, de http://www.ub.edu/aplica_infor/spss/cap5-2.htm
    """)