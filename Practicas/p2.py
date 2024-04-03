#File: p2.py
#Date: 02 / 04 / 2024

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats as sst
from scipy import optimize as sco
import math
import plotly.express as px
import matplotlib.pyplot as plt

#Cargar el archivo csv y hacer que Pandas lea el archivo
data = pd.read_csv('confirmados_fecha.csv')
#print(f'data:\n{data}')


#definimos una lista que será que lea scypi
lista_fechas = []

#definimos una variable indice que será hasta que fila vamos a leer
fila_max = 98

#convertimos el archivo csv en una lista 
for index, row in data.iterrows():
    if index <= fila_max:

        lista_fechas.extend([row['fecha']] * row['Casos por fecha de inicio de síntomas'])

#print(lista_fechas)    


def grafica(lista_fechas):

    #definimos los ejes del histigrama y tomamos su frecuencia 
    date, counts = np.unique(lista_fechas, return_counts=True)

    #Configurar la figura y el histograma
    plt.figure(figsize=(10, 6))
    plt.bar(date, counts, color='blue')
    plt.title('Histograma de Fechas Casos Covid 19')
    plt.xlabel('Fechas')
    plt.ylabel('Casos')
    plt.xticks(rotation=90)

    return plt


#Streamlit

st.title('Histograma de Casos Covid')

#Histograma

st.pyplot(grafica(lista_fechas))