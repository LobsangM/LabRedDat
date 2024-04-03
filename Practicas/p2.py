#File: p2.py
#Date: 02 / 04 / 2024

import streamlit as st
import numpy as np
import pandas as pd
import scipy as scy
import math
import plotly.express as px

data = pd.read_csv('confirmados_fecha.csv')
print(f'data:\n{data}')


lista_fechas = []

fila_max = 100

for index, row in data.iterrows():
    if index <= fila_max:

        lista_fechas.extend([row['fecha']] * row['Casos por fecha de inicio de síntomas'])

print(lista_fechas)    