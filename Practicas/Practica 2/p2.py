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
data_graf = pd.read_csv('confirmados_fecha.csv', usecols=[0,1,2], nrows=200, index_col=0)
#print(f'data:\n{data}')

#definimos una lista que será que lea scypi
lista_fechas = []

#definimos una variable indice que será hasta que fila vamos a leer
fila_max = 98

#convertimos el archivo csv en una lista 
for index, row in data.iterrows():
    if index <= fila_max:

        lista_fechas.extend([row['fecha']] * row['Casos por fecha de inicio de síntomas'])

#valores de columna
        
print(lista_fechas)     

#Metemos la lista en un Data frame y realizamos la gráfica

lista_fechas_df =pd.DataFrame({'fecha': lista_fechas})
fig = px.histogram(lista_fechas_df, x='fecha', title='Histograma de Casos covid')
fig.update_layout(xaxis_title='Fechas', yaxis_title='Casos')

#fit en GNUPLOT







#Streamlit

st.title('Histograma de Casos Covid')

#Histograma

st.plotly_chart(fig)





