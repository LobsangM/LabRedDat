#File: p1.py
#date: 26/02/2024

import numpy as np
import pandas as pd
from scipy import optimize as sco
import math
import plotly.express as px
import streamlit as st

############# PRIMER PARTE, SOLO DATOS DE LA COLUMNA: LOBSANG - REBECA ########################################

st.title('Gráficas de datos: Lobsang - Rebeca')

#el usuario define el valor para m utilizando un slider

m_1 = st.slider('Escoge el valor para m', 1, 50, 100,)

#calculamos la binomial

def binom(x,n,p):
    # print('binom(',x,n,p,')')
    
    x = int(x)
    n = int(n)
        
    comb = math.comb(n,x)
    p_x = p**x
    q_nx = (1-p)**(n-x)

    return comb*p_x*q_nx
    # return A * scs.binom.pmf(x,n,p)

#vectorizamos la funcion
binom = np.vectorize(binom)

#Utilizamos pandas para leer nuestro archivo csv
data = pd.read_csv('datos.csv')
print(f'data:\n{data}')

data = data.loc[:m_1]

#esto nos da la frecuencia de cada valor de la columna

counts_non_sort = data['Lobsang - Rebeca'].value_counts()
counts = pd.DataFrame(np.zeros(11))
# print(counts)

#valores de columna

for row, value in counts_non_sort.items():
    counts.loc[row,0] = value

print(f'counts:\n{counts}')
print(f'index: {counts.index.values}')
print(f'normalized counts: {list(counts[0]/m_1)}')

#realizamos un fit

fit, cov_mat = sco.curve_fit(binom,counts.index.values,counts[0]/m_1,[10,0.5],bounds=[(0,0),(np.inf,1)])

print(f'Fit:\n{fit}\ncov_mat\n{cov_mat}')

n = fit[0]
p = fit[1]

print(f'Este es el valor de n: {n}\nEste es el valor de p: {p}')



binomial_plot = px.line(x=counts.index.values, y=binom(counts.index.values,n,p), title="Lanzamiento de fichas")

binomial_plot.add_bar(x=counts.index.values, y=counts[0]/m_1, name='Lanzamientos experimentales')

#binomial_plot.show()

####################### CALCULO DE LA DESVIACION ESTANDAR ####################

desviacion_estandar = data['Lobsang - Rebeca'].std()

#print("La desviación estándar es:", desviacion_estandar)



############# STREAMLIT ##################

#plotiamos la funciones en streamlit
st.plotly_chart(binomial_plot)

#valor del fit
st.subheader('Valor del Fit')
st.write('El fit personalizado es:',fit )

#valor de la desviacion estandar

st.subheader('Desviación Estándar')
st.write('La desviación estandar es:', desviacion_estandar)



############################ SEGUNDA PARTE - DATOS GENERALES DE CLASE ###############################3

st.title('Gráficas de Datos Generales')

#Nuevamente el usuario define el k a utilizar

k = st.slider('Escoge el valor de k a utilizar',1, 10, 100,)

#calculamos la funcional binomial 

def binom_general(a,b,c):
    #podemos relacionar a=x , b=n , c=p

    a = int(a)
    b = int(n)

    comb_general = math.comb(b,a)
    c_a = c**a
    d_ba = (1-c)**(b-a)

    return comb_general*c_a*d_ba

#vectorizamos la funcion
binom_general = np.vectorize(binom_general)

#utilizamos pandas para leer el archivo csv
data_general = pd.read_csv('datos_generales.csv')

data_general = data_general.loc[:k]



#esto nos da la frecuencia de cada valor de la columna

#counts_non_sort_general = data_general[['Lobsang - Rebeca', 'Guillermo - Shawn', 'Diego - Saul', 'Giovanna - Mario', 'Dessiré - Fabricio', 'Jacobo - Cesar']].value_counts()
counts_non_sort_general = data_general['data'].value_counts()
counts_general = pd.DataFrame(np.zeros(11))

#valores de columna

for row, value in counts_non_sort_general.items():
    counts_general.loc[row,0] = value

print(f'counts:\n{counts_general}')
print(f'index: {counts_general.index.values}')
print(f'normalized counts: {list(counts_general[0]/k)}')

#realizamos el fit

fit_general, cov_mat_general = sco.curve_fit(binom_general,counts_general.index.values,counts_general[0]/k,[10,0.5],bounds=[(0,0),(np.inf,1)])

b = fit_general[0]
c = fit_general[1]

# Definimos la grafica

binomial_plot_general = px.line(x=counts_general.index.values, y=binom_general(counts_general.index.values,b,c), title="Lanzamiento de fichas")

binomial_plot_general.add_bar(x=counts_general.index.values, y=counts_general[0]/k, name='Lanzamiento experimentales')

# Calculo de la desviacion estandar

#desviacion_estandar_general = data_general[['Lobsang - Rebeca', 'Guillermo - Shawn', 'Diego - Saul', 'Giovanna - Mario', 'Dessiré - Fabricio', 'Jacobo - Cesar']].std()
desviacion_estandar_general = data_general['data'].std()

# Streamlit

st.plotly_chart(binomial_plot_general)

st.subheader('Valor del fit de los datos generales')
st.write('El fit personalizado es:',fit_general)

#valor de la desviacion estándar

st.subheader('Desviación estándar de datos generales')
st.write('La desviación estándar es:', desviacion_estandar_general)