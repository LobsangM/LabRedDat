#File: EjemploBinomial.py
#Date: 2024-02-22

from matplotlib import pyplot as plt
import math 

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