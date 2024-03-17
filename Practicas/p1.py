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

m = st.slider('Escoge el valor para m', 1, 50, 100,)

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

data = data.loc[:m]

#esto nos da la frecuencia de cada valor de la columna

counts_non_sort = data['Lobsang - Rebeca'].value_counts()
counts = pd.DataFrame(np.zeros(11))
# print(counts)

#valores de columna

for row, value in counts_non_sort.items():
    counts.loc[row,0] = value

print(f'counts:\n{counts}')
print(f'index: {counts.index.values}')
print(f'normalized counts: {list(counts[0]/m)}')

#realizamos un fit

fit, cov_mat = sco.curve_fit(binom,counts.index.values,counts[0]/m,[10,0.5],bounds=[(0,0),(np.inf,1)])

print(f'Fit:\n{fit}\ncov_mat\n{cov_mat}')

n = fit[0]
p = fit[1]

print(f'Este es el valor de n: {n}\nEste es el valor de p: {p}')



binomial_plot = px.line(x=counts.index.values, y=binom(counts.index.values,n,p), title="Lanzamiento de fichas")

binomial_plot.add_bar(x=counts.index.values, y=counts[0]/m, name='Lanzamientos experimentales')

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
