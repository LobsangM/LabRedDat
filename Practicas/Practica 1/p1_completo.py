import streamlit as st
import numpy as np
import pandas as pd
from scipy import optimize as sco
import math
import plotly.express as px

#Fondo temático
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://i.pinimg.com/564x/39/c7/1e/39c71e43cd06601a698edc75859dd674.jpg");
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

#Título
st.title('Explorando la Distribución Binomial en Lanzamientos de Monedas')

# Párrafos justificados
st.markdown("""
    <div style="text-align: justify">
        La distribución binomial es un modelo de probabilidad discreto que indica la probabilidad de obtener uno de dos resultados posibles al realizar una serie de pruebas independientes.
    </div>
    <br>
    <div style="text-align: justify">
        Cada resultado posible tiene una probabilidad que va de 0 a 1, y en estas pruebas, solo pueden ocurrir dos resultados, como en el lanzamiento de una moneda (cara o cruz) o en la ruleta francesa (rojo o negro).
    </div>
    <br>
    <div style="text-align: justify">
        Cada prueba es independiente de las demás, lo que significa que el resultado de una no afecta la probabilidad de las siguientes. La probabilidad de cada resultado sigue siendo constante en cada prueba.
    </div>    
    <br>
     <div style="text-align: justify">
        En la distribución binomial tenemos tres variables:
    </div> 
""", unsafe_allow_html=True)

# Lista
st.markdown("""
- n: Es el número de veces que repetimos el experimento
- p: es uno de los dos resultados al que llamaremos éxito.
- q: es el otro resultado posible al que llamaremos fracaso.
""")

# Más parrafos justificados
st.markdown("""
    <div style="text-align: justify">
        Como p y q son los dos únicos resultados posibles, entre los dos su porcentaje debe sumar uno por lo que p=1-q. De esto obtenemos la siguiente fórmula.
    </div>
""", unsafe_allow_html=True)
#Notación matemática usando latex
st.latex(r"P(x,n)=\binom{n}{x}p^x(1-p)^{n-x}")
# Más parrafos justificados
st.markdown("""
    <div style="text-align: justify">
        La desviación estándar se define como una métrica que evalúa la amplitud o variabilidad en el contexto de la estadística descriptiva. Se emplea para cuantificar la extensión o dispersión en la que los datos individuales se apartan de la media de un conjunto de datos.
    </div>
    <br>
    <div style="text-align: justify">
       Cuando la desviación estándar es baja, sugiere que los datos están cercanos a la media, mientras que una desviación alta indica que los datos están más dispersos y abarcan un rango más amplio de valores.
    </div>        
""", unsafe_allow_html=True)
#Más notación de latex
st.latex(r"s=\sqrt{\dfrac{1}{n-1}\sum\limits_{n=1}^\infty (x_i-\overline{x})}")

# Crear las pestañas
titulos_pestañas = ['Resumen del Proyecto', 'Procedimiento Experimental', 'Visualización de Resultados', 'Referencias']
pestañas = st.tabs(titulos_pestañas)
 
# Agregar contenido a cada pestaña
with pestañas[0]:
    st.header('Resumen del Proyecto')

# Párrafo en la interfaz usando st.write()
    st.write("El informe incluyó dos partes:")
    st.write("")
    st.write("1. **Análisis de datos individuales:**")
    st.write("   Se generó un histograma que exhibió la distribución del conteo de caras de los primeros $m$ lanzamientos de 10 monedas, donde el lector tuvo la capacidad de variar $m$ en el rango de $0 \leq m \leq 100$. Además, se realizó un ajuste de una función binomial a estos datos y se graficó sobre el histograma. Se proporcionaron los valores resultantes del ajuste, así como el conteo medio de caras y su desviación estándar, medidos experimentalmente y obtenidos del ajuste.")
    st.write("")
    st.write("2. **Análisis de datos colectivos:**")
    st.write("   Se produjo un segundo histograma que representó la distribución del conteo de caras para los 500 datos recopilados por toda la clase. Asimismo, se llevó a cabo un ajuste de una función binomial a estos datos y se mostró sobre el histograma. Se incluyeron los valores derivados del ajuste, así como el conteo medio de caras y su desviación estándar.")
    st.write("")
    st.write("Cada sección del informe proporcionó una comprensión tanto de la distribución individual de los datos como de la distribución colectiva, permitiendo una evaluación completa de la variabilidad de los resultados y su concordancia con el modelo teórico de la distribución binomial.")

 
with pestañas[1]:
    
# Procedimiento Experimental
    st.write("## Procedimiento Experimental")

# 1. Preparación de materiales
    st.write("**1. Preparación de materiales:**")
    st.write("- Se reunieron 10 monedas idénticas.")
    st.write("- Se registraron los datos en una hoja de cálculo.")

# 2. Configuración del experimento
    st.write("**2. Configuración del experimento:**")
    st.write("- Se estableció un área de trabajo limpia y plana para realizar los lanzamientos de las monedas.")
    st.write("- Se decidió sobre el número máximo de lanzamientos que se realizarían para cada moneda, denotado como $m$.")
    st.write("- Se configuró el entorno de programación preferido con las bibliotecas necesarias, como `numpy`, `matplotlib` y `pandas`.")

# 3. Realización de los lanzamientos
    st.write("**3. Realización de los lanzamientos:**")
    st.write("- Se lanzarón las 10 monedas $m$ veces.")
    st.write("- Se registró el número de caras obtenidas en cada lanzamiento.")

# 4. Análisis de datos individuales
    st.write("**4. Análisis de datos individuales:**")
    st.write("- Para cada valor de $m$ en el rango de $0 \leq m \leq 100$:")
    st.write("  - Se calculó la distribución del conteo de caras de los primeros $m$ lanzamientos de las 10 monedas.")
    st.write("  - Se realizó un ajuste de una función binomial a los datos obtenidos.")
    st.write("  - Se graficó la distribución junto con la función binomial ajustada.")
    st.write("  - Se extrajeron los valores resultantes del ajuste, así como el conteo medio de caras y su desviación estándar.")

# 5. Análisis de datos colectivos
    st.write("**5. Análisis de datos colectivos:**")
    st.write("- Se reunieron los datos de todos los participantes de la clase (500 datos en total).")
    st.write("- Se calculó la distribución del conteo de caras para los 500 datos recopilados.")
    st.write("- Se realizó un ajuste de una función binomial a estos datos.")
    st.write("- Se graficó la distribución junto con la función binomial ajustada.")
    st.write("- Se extrajeron los valores resultantes del ajuste, así como el conteo medio de caras y su desviación estándar.")

 
with pestañas[2]:
    st.header('Histogramas')


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


with pestañas[3]:
    st.header('Referencias')
    st.write('Software DELSOL. (2019, junio 20). Distribución binomial. Sdelsol.com. https://www.sdelsol.com/glosario/distribucion-binomial/')
    st.write("Ortega, C. (2022, diciembre 21). Desviación estándar: Qué es, usos y cómo obtenerla. QuestionPro. https://www.questionpro.com/blog/es/desviacion-estandar/")