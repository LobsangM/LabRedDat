import numpy as np
import pandas as pd

ds=[1.1,3.2,4.9,0.4,3.2,]
ts=[2.05,5.9,10.2,1.1,6.2,]

ds=np.array(ds)
ts=np.array(ts)

#Promedios
d_mean=np.mean(ds)
t_mean=np.mean(ts)

print(d_mean)
print(t_mean)


#Desviaciones estándar
d_std=np.std(ds,ddof=1)
t_std=np.std(ts,ddof=1)

print(d_std)
print(t_std)


#lista de Velocidad
vs = ds/ts

print(vs)


#Promedio de Velocidad
v_mean=np.mean(vs)

print(v_mean)


#Desviación Estándar de la Velocidad
v_std=np.std(vs,ddof=1)

print(v_std)

