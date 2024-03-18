import numpy as np
import pandas as pd
from scipy import optimize as sco
import math
import plotly.express as px
import streamlit as st

data_general = pd.read_csv('datos_generales.csv')

print(data_general['Datos - Generales'])