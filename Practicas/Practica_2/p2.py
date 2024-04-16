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
import plotly.graph_objects as go

####Titulo de Streamlit

st.title("Gráficas datos de Covid 19")


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

## Empieza Discución de resultados:

st.title("Discución de Resultados")

st.subheader("Criterio de Selección de datos")

st.write("Primero, se discutirá la elección de los datos y la columna que se graficó.\nSe utilizaron 2 archivos diferentes para realizar las gráficos y los ajustes correspondientes, en la primera gráfica se utilizó un archivo csv que contiene los datos desde el primer caso de síntomas (13 de Febrero del 2020) hasta el ultimo caso registrado por síntoma del dia 1 de Junio del 2020. \n\n  ")
st.write("En la segunda gráfica se utilizó un archivo csv que contiene los casos desde el 15 de junio del 2020  hasta datos del 15 de Marzo del 2021, la razón de utlizar el archivo de esta forma es para que no generará conflicto a la hora de que el fit jalara los datos del csv, por lo tanto son datos que estan recortados pero siguen siendo los originales.")
st.write("Con respecto a la columna que se utilizó como referencia para realizar las gráficas y el fit se debe decir que se escogió la columna llamada \"sintomas\" debido a la razón principal de que era la que contenia más datos desde el primer dia de registro que es el 13 de Febrero del 2020, ya que las demas columnas empezaban a tener datos desde marzo y se quería realizar una grafica con la mayor cantidad de datos posibles.   ")


st.subheader("Criterio de Gráficas y Fit realizado")

st.write("Con respecto a la primera gráfica (con datos de hasta el 1 de Junio del 2020) se realizó un histograma donde se ve claramente la subida de casos de manera exponencial. Se realizó un fit de la forma de una función gaussiana que cuadra perfectamente con los datos obtenidos. Con respecto a la predicción de futuros casos a partir de esa fecha con el modelo matemático propuesto, se debe aclarar que no se cumple la predicción realizada, ya que segun nuestra predicción, los datos despues del 1 de Junio del 2020, empezarian a bajar hasta llegar a cero en cuestión de unos pocos meses, pero se sabe que en realidad no fue de esa manera. Consideremos que esto se debe a que contamos con muy pocos datos y es probable que el modelo matemático propuesto no sea el indicado. ")
st.write("Ahora bien, con respecto a la segunda gráfica (con datos desde el 15 de Junio del 2020 hasta el 15 de Marzo del 2021) también se realizó un histograma donde se ve como los datos decienden un poco pero nunca llegan a cero, se mantienen formando pequeños \"picos\" a lo largo del tiempo. El fit se realizó con una ecuación gaussiana al igual que en la primera gráfica, y a pesar que los primeros datos coinciden con la bajada, a lo largo del eje x los datos dejan de cumplir ese fit, esta discrepancia en la medida se puede deber a que no propusimos un fit adecuado para los datos. La predicción que genera nuestro modelo matemático propuesto no se ajusta muy bien a los datos obtenidos, ya que predice que los datos tienden a cero, pero en realidad a lo largo del tiempo se obtienen diferentes picos de datos los cuales el fit no predice")

