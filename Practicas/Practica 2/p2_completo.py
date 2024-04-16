#Práctica 2
import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats as sst
from scipy import optimize as sco
import math
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#Path
#path = "practicas/practica2/"
# Fondo temático
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://i.pinimg.com/564x/8b/2c/4e/8b2c4e50f131160938153449e8b8dad2.jpg");
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

# Agrega el enlace de la fuente cursiva de Google Fonts a tu aplicación
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Homemade Apple&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True
)

# Aplica la fuente cursiva a un título y texto en tu aplicación
st.markdown("""
    <h1 style='font-family: "Homemade Apple", cursive; text-align: center;'>Predicción de Covid 2019</h1>
    """,
    unsafe_allow_html=True
)


# Parrafos justificados
st.markdown("""
    <div style="text-align: justify">
        ¿Qué es el COVID?    
    </div>
    <br>
    <div style="text-align: justify">
        Los coronavirus son una extensa familia de virus que pueden provocar una variedad de enfermedades, desde el resfriado común hasta afecciones más graves, como el síndrome respiratorio de Oriente Medio (MERS-CoV) y el síndrome respiratorio agudo severo (SRAS-CoV).    
    </div>
    <br>
    <div style="text-align: justify">
        Un nuevo coronavirus es una cepa que no se había visto previamente en humanos, como el 2019nCoV o COVID-19, identificado por primera vez durante el brote en Wuhan, China, en diciembre de 2019.    
    </div>
    <br>
    <div style="text-align: justify">
        La transmisión de los coronavirus puede ocurrir de animales a personas (transmisión zoonótica). Se sabe que el SRAS-CoV se transmitió de la civeta al ser humano y que el MERS-CoV se originó en dromedarios. Aunque existen otros coronavirus que circulan entre animales, no han infectado a los humanos.        
    </div>    
    <br>
    <div style="text-align: justify">
        Los síntomas comunes incluyen fiebre y problemas respiratorios como tos y dificultad para respirar. En casos graves, pueden causar neumonía, síndrome respiratorio agudo severo, insuficiencia renal e incluso la muerte.            
    </div> 
   <br>
    <div style="text-align: justify">
        Las medidas de prevención habituales incluyen una buena higiene de manos y respiratoria, como cubrirse la boca al toser o estornudar, y asegurarse de cocinar completamente la carne y los huevos. También se debe evitar el contacto cercano con personas que presenten síntomas respiratorios como tos o estornudos.            
    </div>         
""", unsafe_allow_html=True)
#Espacio en blanco
st.write("")  # o st.text("")


# Texto
st.markdown("""
¿Qué es la distribución binomial?
            
La distribución binomial es un modelo matemático que describe la probabilidad de obtener un número específico de éxitos en una serie de ensayos independientes, donde cada ensayo tiene solo dos posibles resultados: éxito o fracaso. Cada ensayo se considera independiente, lo que significa que el resultado de uno no afecta al resultado de otro.

Las características principales de la distribución binomial son:

1. **Número fijo de ensayos**: Se realiza un número fijo de ensayos, denotado como \( n \).

2. **Resultados mutuamente excluyentes**: Cada ensayo tiene solo dos resultados posibles, que son mutuamente excluyentes, como éxito o fracaso.

3. **Probabilidad de éxito constante**: La probabilidad de éxito en cada ensayo, denotada como \( p \), permanece constante en todos los ensayos.

4. **Independencia**: Los ensayos son independientes entre sí.

La fórmula para calcular la probabilidad de obtener exactamente \( k \) éxitos en \( n \) ensayos bajo una distribución binomial es:
""")

#Notación matemática usando latex
st.latex(r"P(x,n)=\binom{n}{x}p^k(1-p)^{n-k}")

# Texto
st.markdown("""
Donde:

- \( P(X = k) \) es la probabilidad de obtener exactamente \( k \) éxitos.
- $\( \binom{n}{k} \)$ es el coeficiente binomial, que representa el número de formas de elegir \( k \) éxitos de \( n \) ensayos.
- \( p \) es la probabilidad de éxito en un solo ensayo.
- \( (1-p) \) es la probabilidad de fracaso en un solo ensayo.
- \( n \) es el número total de ensayos.
- \( k \) es el número de éxitos que se están considerando.

Aplicaciones de la distribución binomial:

1. **Pruebas de éxito o fracaso**: Se utiliza para modelar situaciones donde solo hay dos resultados posibles en cada ensayo, como lanzar una moneda, tirar un dado, etc.

2. **Procesos de Bernoulli**: Modela eventos que tienen solo dos resultados posibles y se repiten un número fijo de veces.

3. **Control de calidad**: Se utiliza para determinar la probabilidad de que un cierto número de productos de una línea de producción cumplan con ciertas especificaciones.

4. **Estadísticas y encuestas**: Se puede utilizar para modelar la probabilidad de obtener cierto número de respuestas afirmativas en una encuesta de opción múltiple.

5. **Biología y medicina**: Se aplica en estudios clínicos para evaluar la eficacia de un tratamiento o medicamento, así como en la genética para estudiar la herencia de ciertos rasgos.

""")

# Espacio en blanco
st.write("")  # o st.text("")

# Texto principal con notación LaTeX
st.write("""
La distribución de Poisson es un modelo matemático que describe la probabilidad de que ocurra un número específico de eventos en un intervalo de tiempo o espacio, dado un ritmo promedio de ocurrencia y bajo la suposición de que los eventos ocurren de forma independiente y a una tasa constante.

Las características principales de la distribución de Poisson son:

1. **Eventos discretos e independientes**: Los eventos que se están contando son discretos (como llegadas, llamadas telefónicas, accidentes, etc.) y son independientes entre sí, lo que significa que la ocurrencia de uno no afecta la ocurrencia de otro.

2. **Tasa promedio de ocurrencia constante**: La tasa promedio de ocurrencia de eventos, denotada como \( \lambda \) (lambda), es constante durante el intervalo de tiempo o espacio considerado.

La fórmula para calcular la probabilidad de que ocurran exactamente \( k \) eventos en un intervalo bajo una distribución de Poisson es:
""")

#Notación matemática usando latex
st.latex(r"P(X = k) = \frac{e^{-\lambda} \cdot \lambda^k}{k!} ")

# Texto
st.write("""
Donde:

- \( P(X = k) \) es la probabilidad de que ocurran \( k \) eventos.
- \( e \) es la base del logaritmo natural (aproximadamente 2.71828).
- \( \lambda \) es la tasa promedio de ocurrencia de eventos en el intervalo.
- \( k \) es el número de eventos que se están considerando.
- \( k! \) es el factorial de \( k \).

Aplicaciones de la distribución de Poisson:

1. **Modelado de llegadas**: Se utiliza para modelar el número de llegadas de clientes a un banco, llamadas a un centro de atención telefónica, llegadas de vehículos a un peaje, etc.

2. **Procesos de conteo**: Se aplica en situaciones donde se cuenta el número de eventos en un intervalo de tiempo o espacio, como el número de accidentes de tráfico en una carretera en un día.

3. **Procesos de producción**: Se utiliza en la planificación de la producción para estimar la cantidad de productos defectuosos que se producirán en un cierto período de tiempo.

4. **Estimación de probabilidades raras**: La distribución de Poisson se utiliza para modelar eventos raros pero importantes, como fallas de equipos críticos, incidentes de seguridad, etc.

5. **Biología y medicina**: Se aplica en biología para modelar la tasa de mutaciones genéticas y en medicina para analizar la frecuencia de eventos médicos raros, como casos de enfermedades raras.
""")


# Crear las pestañas
titulos_pestañas = ['Resumen de la Práctica', 'Procedimiento Experimental', 'Discución de Resultados']
pestañas = st.tabs(titulos_pestañas)
 
# Agregar contenido a cada pestaña
with pestañas[0]:
    st.header('Resumen de la práctica')
    # Texto principal del resumen
    st.write("""
    En el reporte, se aborda el desafío de predecir la evolución de los contagios por COVID-19 en Guatemala utilizando datos de los registros del Ministerio de Salud. Como epidemiólogos, nos enfrentamos a la tarea de realizar una predicción basada en datos disponibles hasta el 15 de marzo de 2021.

    Para lograr esta predicción, se optó por ajustar los datos a una distribución binomial. La selección de los datos y el enfoque metodológico se realizaron cuidadosamente, considerando varios aspectos, como la fiabilidad de los datos, la representatividad de las muestras y la idoneidad del modelo estadístico.

    Se establecieron dos fechas clave para el análisis: el 1 de junio de 2020 y el 15 de marzo de 2021. Estas fechas delimitan el período de estudio y permiten utilizar datos recopilados hasta ese momento para realizar la predicción.

    En la sección de discusión del reporte, se detallan las decisiones tomadas durante el proceso de análisis y modelado. Se justifica la elección de los datos utilizados, así como los métodos empleados para el ajuste a la distribución binomial. Además, se presentan las limitaciones y posibles sesgos del modelo, así como recomendaciones para futuras investigaciones.

    El objetivo final del reporte es proporcionar una visión clara y fundamentada sobre la predicción de la evolución de los contagios por COVID-19 en Guatemala, con el fin de contribuir al diseño de estrategias de prevención y control de la enfermedad.
    """)
    st.header('Referencias')
    st.write('Software DELSOL. (2019, junio 20). Distribución binomial. Sdelsol.com. https://www.sdelsol.com/glosario/distribucion-binomial/')
    st.write("Coronavirus. (s/f). Paho.org. Recuperado el 16 de abril de 2024, de https://www.paho.org/es/temas/coronavirus")

    #st.bar_chart(datos)
 
with pestañas[1]:
    st.header('Procedimiento Experimental para el Análisis de Datos de COVID-19')

# Introducción
    st.write("""
    El objetivo de esta práctica fué analizar los datos de COVID-19 del Ministerio de Salud para comprender mejor la evolución de la enfermedad en Guatemala y realizar predicciones sobre su propagación futura.

    Para llevar a cabo este análisis, se siguieron los siguientes pasos:
    """)

# Paso 1: Obtención de datos
    st.write("Paso 1: Obtención de Datos")
    st.write("""
    Se accedió a los registros de casos de COVID-19 proporcionados por el Ministerio de Salud. Estos datos estaban disponibles en un archivo CSV o mediante una API. Se utilizó Python para cargar y procesar los datos.
    """)

# Paso 2: Exploración de datos
    st.write("Paso 2: Exploración de Datos")
    st.write("""
    Se realizó una exploración inicial de los datos para comprender su estructura y contenido. Se analizaron las variables disponibles, como la fecha de reporte, el número de casos confirmados, las tasas de mortalidad, etc.
    """)

# Paso 3: Preprocesamiento de datos
    st.write("Paso 3: Preprocesamiento de Datos")
    st.write("""
    Se llevó a cabo el preprocesamiento de los datos para limpiarlos y prepararlos para el análisis. Esto incluyó la eliminación de valores nulos, la corrección de errores, la normalización de datos, etc.
    """)

# Paso 4: Análisis estadístico
    st.write("Paso 4: Análisis Estadístico")
    st.write("""
    Se realizó un análisis estadístico de los datos para identificar tendencias, patrones y relaciones significativas. Se calcularon estadísticas descriptivas, se visualizaron los datos mediante gráficos y se aplicaron pruebas estadísticas según fue necesario.
    """)

# Paso 5: Modelado y predicción
    st.write("Paso 5: Modelado y Predicción")
    st.write("""
    Se utilizaron técnicas de modelado estadístico, como ajustes a distribuciones binomiales o de Poisson, para realizar predicciones sobre la evolución futura de la enfermedad. Se evaluó la calidad del modelo y se realizaron ajustes según fue necesario.
    """)

# Paso 6: Conclusiones y recomendaciones
    st.write("Paso 6: Conclusiones y Recomendaciones")
    st.write("""
    Se resumieron los hallazgos del análisis y se formularon conclusiones basadas en los resultados obtenidos.
    """)


 
with pestañas[2]:
####Titulo de Streamlit
    st.header("Gráficas datos de Covid 19")


#Cargar el archivo csv y hacer que Pandas lea el archivo
    data = pd.read_csv('confirmados_fecha_1.csv')

#datos para la primera grafica, al 1 de julio

    A=1000
    u=135
    r=35.6

    x_fit = np.linspace(0, 127, 100)
    x=data['index']
    y=data['sintomas']

#definimos la funcion que vamos a usar para el fit

    def fun_gauss(x, A, u, r):
        return A * np.exp(-((x - u) / r) ** 2 / 2)

    y_fit = fun_gauss(x, A, u, r)

#definimos la figura que será nuestra gráfica y le agregamos el fit

    fig = go.Figure()

    fig.add_trace(go.Bar(x=x, y=y, name='Sítnomas por día',marker=dict(
            colorscale='Picnic',  # color
            color=data['sintomas']
        )))

    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, name='Fit', mode='lines'))


    fig.update_layout(
        title='Gráfico 1: Días vs Síntomas hasta 1 de Junio del 2020',
        xaxis_title='Numero de días',
        yaxis_title='Cantidad de casos por síntomas',
        barmode='group'
    )

    st.plotly_chart(fig)




#Grafica de datos hasta el 15 de marzo del 2021

#recordatorio: hacer enfasis en ek recorte de datos para que se ajuste el plot
# ponerlo en discusion de resultados


    data2 = pd.read_csv('confirmados_fecha_3.csv')
    A2= 900
    u2= -32.5
    r2= 43.5

    x_fit2= np.linspace(0,274,100)
    x2= data2['index']

#definimos la nueva función


    def funcion2_gaussiana(x2, A2, u2, r2):
        return A2 * np.exp(-((x2 - u2) / r2) ** 2 / 2)  


    y_fit2 = funcion2_gaussiana(x2, A2, u2, r2)
    fig2= go.Figure()

    fig2.add_trace(go.Bar(x=x2, y=data2['sintomas'],name='Síntomas por día',marker=dict(
            colorscale='Plasma',  # Aquí puedes cambiar el esquema de colores según tu preferencia
            color=data2['sintomas']
        )))

    fig2.add_trace(go.Trace(x=x_fit2, y=y_fit2, name='Fit'))
    fig2.update_layout(
        title='Gráfico 2: Días vs Síntomas hasta el 15 de Marzo del 2021 ',
        xaxis_title='Número de días',
        yaxis_title='Cantidad de casos por síntomas',
        barmode='group'
    )

    st.plotly_chart(fig2)
 
    st.subheader("Criterio de Selección de datos")

    st.write("Primero, se discutirá la elección de los datos y la columna que se graficó.\nSe utilizaron 2 archivos diferentes para realizar las gráficos y los ajustes correspondientes, en la primera gráfica se utilizó un archivo csv que contiene los datos desde el primer caso de síntomas (13 de Febrero del 2020) hasta el ultimo caso registrado por síntoma del dia 1 de Junio del 2020. \n\n  ")
    st.write("En la segunda gráfica se utilizó un archivo csv que contiene los casos desde el 15 de junio del 2020  hasta datos del 15 de Marzo del 2021, la razón de utlizar el archivo de esta forma es para que no generará conflicto a la hora de que el fit jalara los datos del csv, por lo tanto son datos que estan recortados pero siguen siendo los originales.")
    st.write("Con respecto a la columna que se utilizó como referencia para realizar las gráficas y el fit se debe decir que se escogió la columna llamada \"sintomas\" debido a la razón principal de que era la que contenia más datos desde el primer dia de registro que es el 13 de Febrero del 2020, ya que las demas columnas empezaban a tener datos desde marzo y se quería realizar una grafica con la mayor cantidad de datos posibles.   ")


    st.subheader("Criterio de Gráficas y Fit realizado")

    st.write("Con respecto a la primera gráfica (con datos de hasta el 1 de Junio del 2020) se realizó un histograma donde se ve claramente la subida de casos de manera exponencial. Se realizó un fit de la forma de una función gaussiana que cuadra perfectamente con los datos obtenidos. Con respecto a la predicción de futuros casos a partir de esa fecha con el modelo matemático propuesto, se debe aclarar que no se cumple la predicción realizada, ya que segun nuestra predicción, los datos despues del 1 de Junio del 2020, empezarian a bajar hasta llegar a cero en cuestión de unos pocos meses, pero se sabe que en realidad no fue de esa manera. Consideremos que esto se debe a que contamos con muy pocos datos y es probable que el modelo matemático propuesto no sea el indicado. ")
    st.write("Ahora bien, con respecto a la segunda gráfica (con datos desde el 15 de Junio del 2020 hasta el 15 de Marzo del 2021) también se realizó un histograma donde se ve como los datos decienden un poco pero nunca llegan a cero, se mantienen formando pequeños \"picos\" a lo largo del tiempo. El fit se realizó con una ecuación gaussiana al igual que en la primera gráfica, y a pesar que los primeros datos coinciden con la bajada, a lo largo del eje x los datos dejan de cumplir ese fit, esta discrepancia en la medida se puede deber a que no propusimos un fit adecuado para los datos. La predicción que genera nuestro modelo matemático propuesto no se ajusta muy bien a los datos obtenidos, ya que predice que los datos tienden a cero, pero en realidad a lo largo del tiempo se obtienen diferentes picos de datos los cuales el fit no predice")