# File: parcial_1
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




#ejecutamos nuestra página de Streamlit

