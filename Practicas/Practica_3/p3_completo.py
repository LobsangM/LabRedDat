import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats as sst
from scipy import optimize as sco
import math
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
 
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
 
# Crear las pestañas
titulos_pestañas = ['Entradas del Usuario', 'Visualización de Datos', 'Análisis de Datos']
pestañas = st.tabs(titulos_pestañas)
 
# Agregar contenido a cada pestaña
with pestañas[0]:
    st.header('Entradas del Usuario')
    st.text_input('Ingrese algún texto')
    st.number_input('Ingrese un número')
 
with pestañas[1]:
    st.header('Visualización de Datos')
    st.table({'columna1': [1, 2, 3], 'columna2': [4, 5, 6]})
 
with pestañas[2]:
    st.header('Análisis de Datos')
    st.line_chart([1, 2, 3, 4, 5])