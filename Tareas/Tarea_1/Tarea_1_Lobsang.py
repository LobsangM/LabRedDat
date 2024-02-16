import numpy as np
import pandas as pd 

#lista

ts=[6.29,6.37,6.35,6.62,6.23,6.39,6.40,6.29,]
ds=[10.06,10.02,10.09,10.05,9.78,9.99,9.69,9.85,]

#listas de Python

ts= np.array(ts)
ds= np.array(ds)

#promedios

t_mean = np.mean(ts)
d_mean = np.mean(ds)

print(t_mean)
print(d_mean)

#Desviación Estándar

t_std = np.std(ts,ddof=1)
d_std = np.std(ds,ddof=1)

print(t_std)
print(d_std)

#velocidad

vs = ds/ts

print(vs)

#promedio velocidad

v_mean = np.mean(vs)

print(v_mean)

#calculo de varianza de V, primero calculamos los terminos

ter1 = (1/t_mean)*d_std

print(ter1)

ter2 = -(d_mean/t_mean**2)*t_std

print(ter2)

ter3_1 = (2/(len(ts)-1))*(1/t_mean)*(-d_mean/t_mean**2)

print(ter3_1)

ter3_2 = ((ds-d_mean)*(ts-t_mean)).sum()

print(ter3_2)

ter3 = ter3_1*ter3_2

print(ter3)

v_std = (ter1**2+ter2**2+ter3)**(1/2)

print(v_std)

#cálculo de la varianza de una forma más fácil y exacta------ Valor correcto

v_std2 = np.std(vs,ddof=1)

print(v_std2)



