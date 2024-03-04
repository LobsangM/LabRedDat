# File: parcial_1.py
# Date: 01/03/2024

#importar Librerias

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math 
import streamlit as st

#Ponemos el título y nombre de la Práctica

st.title('Examen Parcial 1 - Parte Práctica')
st.caption('por Lobsang Méndez - 202112428')

#Descripcion del parcial 

multi1= '''En la siguiente página de Streamlit podrás calcular y graficar la distribución binomial para algún n y p dados.

Pero antes hablemos un poco acerca del origen de la función binomial y su explicación: '''

st.markdown(multi1)

# Deducción de Fórmula 

st.subheader('Fórmula de la distribución binomial')

multi2= '''La Fórmula de la Distribución binomial se deriva del binomio de Newton. 
El binomio de Newton es una fórmula algebraica para expandir potencias de binomios.

Recordemos que se expresa de la siguiente manera:'''

st.markdown(multi2)

# Binomio de Newton (en Latex)

st.latex(r'''
    \left( a+b \right)^n = \displaystyle \sum^n_{x=0} \binom{n}{x} \cdot a^{n-x} \cdot b^{x}
''')

# Coeficiente Binomial

st.markdown(r'''
    donde $$\binom{n}{x}$$ es el coeficiente binomial, que representa el número de formas de elegir $$x$$
    elementos de un conjunto de $$n$$ elementos y se expresa de la siguiente forma:
''')

st.latex(r'''
    \binom{n}{x} = \frac{n!}{x!\left( n-x \right)!}    
''')

# Fórmula

st.markdown(r'''
    Ahora, si consideramos un ensayo donde tengamos 2 posibles resultados (éxito o fracaso, o cara y cruz en el caso de una moneda,
     o una cara en particular en el caso de un dado) podemos expresar la posibilidad de tener $$x$$ en $$n$$ ensayos como:
''')

st.latex(r'''
    P_B(x,n) = \binom{n}{x} \cdot p^x \cdot (1-p)^{n-x} 
''')
st.latex(r'''
    P_B(x,n) = \frac{n!}{x!\left( n-x \right)!} \cdot p^x \cdot (1-p)^{n-x}
''')

st.markdown(r'''
    Finalmente esa sería la fórmula para las posibilidades en una distribución binomial.

    Notemos que el término $$p^x \cdot (1-p)^{n-x}$$ corresponde a un principio básico donde la suma de las probabilidades es
    siempre igual a $$1$$ o $$100\%$$, por lo tanto, la fórmula para $$P_b$$ nos dice las formas que podemos combinar un resultado
    multiplicado por el total de sus probabilidades.    
''')

################################################### st.slider #####################################

st.subheader('st.slider')

st.markdown(r'''
    Hablemos un poco acerca del $$input$$ que se utilizará en la presente página de streamlit, el cual es st.slider. Es un widget en
    Streamlit que permite a los usuarios seleccionar un valor numérico dentro de un rango especificado, utilizando una barra deslizante.
    es útil para permitir ajustar valores numéricos de manera interactiva en tus aplicaciones de Streamlit, como
    configuraciones de parámetros, selección de rangos, etc. 
            
    Aqui tienes un ejemplo de como se ve st.slider para que lo puedas probar   
''')

####### Prueba de st.slider

prueba = st.slider('Pruébalo tu mismo', 0, 100, 25)
st.write("Tu valor seleccionado es", prueba,)

st.markdown(r'''
    Se elijió este input ya que es bastante intuitivo y fácil de usar, además que permite establecer valores predefinidos y
    así evitar escoger valores que no sean adecuados en las ecuaciones
''')

######################################################################################################
############################### DISTRIBUCIÓN BINOMIAL ##############################################
####################################################################################################


st.header('Distribución Binomial')

st.markdown(r'''
    Ahora, con todo lo anterior explicado, prueba por ti mismo como funciona la ecuación para una distribución binomial para cualquier
    valor de $$n$$ y $$p$$.

    Porfavor ingresa los valores que gustes y se hará el cálculo correspondiente y una gráfica. 
''')

############ Valores para p y n

p_valor = st.slider('Escoge el valor para p', 0.0, 1.0, 1.0, step=0.01)
st.write("Tu valor de p es:",p_valor,)

n_valor = st.slider('Escoge el valor para n', 1, 100, 50,)
st.write("Tu valor de n es:",n_valor,)


#### Definimos la funcion llamada "binomial"


def binomial(x,n,p,q):
    #todo lo que este aqui adentro es parte de lo que se ejecuta en la funcion binomial
    
    comb = math.comb(n,x)
    p_x = p**x
    q_nx = q**(n-x)

    return comb*p_x*q_nx



eval_ = binomial(100,600,1/6,5/6)
#eval_ = binomial(n=100,x=600,p=1/6,q=5/6) otra forma de escribirlo


print(eval_)

# Asignamos los valores del usuario a las variables

n = n_valor
p = p_valor
q = (1-p_valor)

#lista

lista = np.arange(n+1)
print(lista)

#empezamos a armar la tabla

data_table = pd.DataFrame({'x':lista})
data_table['Nueva'] = data_table['x'] - 50

data_table['Pb'] = data_table.apply(lambda row: binomial(row['x'],n,p,q), axis=1)


print(data_table)


#Declarando una Fijgura

binomial_plot, axis = plt.subplots()

axis.bar(data_table['x'],data_table['Pb'])

axis.plot(data_table['x'],data_table['Pb'],color='C1')

binomial_plot.show()

#mostrar la gráfica

st.subheader('Gráfica distribución binomial')

st.pyplot(binomial_plot)



#NOTA PARA JORGE: escogí utilizar st.slider para simplificar el codigo, ya que pude haber utilizado el comando "if", para hacer que
#                 n fuera siempre un valor no muy elevado y lo mismo para p, pero me pareció una idea mas facil poner ese limite
#                 al usuario sin tanta complicación.











#ejecutamos nuestra página de Streamlit

