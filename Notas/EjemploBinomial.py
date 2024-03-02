#File: EjemploBinomial.py
#Date: 2024-02-22

#Importar librerias

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math 
import streamlit as st

#Funcion de distribucion binomial 

def binomial(x,n,p,q):
    #todo lo que este aqui adentro es parte de lo que se ejecuta en la funcion binomial
    
    comb = math.comb(n,x)
    p_x = p**x
    q_nx = q**(n-x)

    return comb*p_x*q_nx


eval_ = binomial(100,600,1/6,5/6)
#eval_ = binomial(n=100,x=600,p=1/6,q=5/6) otra forma de escribirlo

print(eval_)

n = 50
p = 1/2
q = 1/2

lista = np.arange(n+1)
print(lista)

#Esto es una lista
#[3,4,5,6]

# Esto es un diccionario
#a = {'pc1':67, 'pc2':80}
#print(a)
#print(a['pc1'])

data_table = pd.DataFrame({'x':lista})
data_table['Nueva'] = data_table['x'] - 50

# lambda
#f = lambda x: x+1
#print('Lambda:',f(7))

data_table['Pb'] = data_table.apply(lambda row: binomial(row['x'],n,p,q), axis=1)


print(data_table)


#Declarando una Fijgura

binomial_plot, axis = plt.subplots()

axis.bar(data_table['x'],data_table['Pb'])

axis.plot(data_table['x'],data_table['Pb'],color='C1')

binomial_plot.show()

#############################################################################
############################ STREAMLIT ######################################
#############################################################################

st.title('Graficos binomiales')

st.pyplot(binomial_plot)


