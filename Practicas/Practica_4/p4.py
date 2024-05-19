# file: classification.py

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_y_prediction(W, phi):
    score = np.dot(W, phi)
    if score > 0:
        y_predict = 1
    elif score == 0:
        y_predict = np.nan
    elif score < 0:
        y_predict = -1

    return y_predict

# def get_y_prediction(W, phi):
#     score = np.dot(W, phi)

#     return score

def loss_hinge(W, phi, y):

    score = np.dot(W, phi)
    pepi2 = score * y

    if pepi2 <= 1:
        r = -pepi2 + 1
    else:
        r = 0

    return r


def grad_loss_hinge(W, phi, y):
    # Se realiza el calculo algebraico del gradiente para llegar a esta funcion:
    score = np.dot(W, phi)
    pepi2 = score * y
    if pepi2 <= 0:
        r = -phi*y
    else:
        r = phi*0

    return r


def loss_sq(W, phi, y):

    score = np.dot(W, phi)
    pepi2 = score - y

    return pepi2**2


def grad_loss_sq(W, phi, y):
    # Se realiza el calculo algebraico del gradiente para llegar a esta funcion:
    score = np.dot(W, phi)
    pepi2 = score - y

    return 2*pepi2*phi


# data = pd.read_csv('data.csv', header=0,delimiter=';')#,dtype={'vector':np.ndarray})

# print(data)
# print(data.columns)

# print(data['vector'])
# print(type(data['vector'][0]))
# # print(eval(data['vector'][0]))
# # print(type(eval(data['vector'][0])))

# # dim = data['vector'][0].shape[0]
# dim = len(eval(data['vector'][0]))

# print(dim)

# W=np.zeros(dim)
# print(W)

####################################################################################################
# Fabricar el set de datos teórico: ################################################################
####################################################################################################

dim = 3
W_target = (np.random.random((dim)) * 10) - 5

# Amount of data in our dataset:
tds_len = 500

# Generate the [t]raining [d]ata [s]et:
tds = []

for i in range(tds_len):
    fv_i = (np.random.random((dim)) * 10) - 5
    y_i = get_y_prediction(W_target, fv_i)

    # fv_i += np.random.random((dim))*0.1

    tds.append((y_i, fv_i))

df_tds = pd.DataFrame(tds, columns=['etiqueta', 'vector'])

max_iterations = 100
W_training = np.zeros((dim))
# W_training = W_target

for iteration in range(max_iterations):
    mean_loss = np.mean([loss_sq(W_training, tds_i[1], tds_i[0]) for tds_i in tds])
    mean_grad_loss = np.mean([grad_loss_sq(W_training, tds_i[1], tds_i[0]) for tds_i in tds],axis=0)

    print(mean_grad_loss)
    W_training -= mean_grad_loss*0.01

    # print(mean_grad_loss)

    print(f'{W_training} ; {mean_loss}')
    # print(f'Current weight:\n\t{W_training}\nLoss value:\n\t{mean_loss}')


print(f'W_target: {W_target}\nW_traini: {W_training}')
print(f'{W_training/W_target}')


# Como plotear puntos con distintos colores:
fig,ax = plt.subplots(dim-1)
# print(df_tds)

if dim-1 == 1:
    ax = [ax]

label_to_color = {1:'C1',-1:'C2',0:'C0'}

df_tds['x'] = df_tds['vector'].apply(lambda row: row[0])
df_tds['color'] = df_tds['etiqueta'].apply(lambda label: label_to_color[label])

# print(df_tds)
# origin = np.zeros((2,2))

for i in range(dim-1):

    df_tds['y'] = df_tds['vector'].apply(lambda row: row[i+1])

    cls_1 = df_tds[df_tds['etiqueta'] == 1]
    cls_minus_1 = df_tds[df_tds['etiqueta'] == -1]
    ax[i].scatter(cls_1['x'],cls_1['y'],color='C1',label='+1')
    ax[i].scatter(cls_minus_1['x'],cls_minus_1['y'],color='C2',label='-1')


    ax[i].quiver(0,0,W_target[0],W_target[i+1])
    ax[i].quiver(0,0,W_training[0],W_training[i+1])

    # ax[i].legend(loc='upper right')
    # ax.plot(x_false,y_false,color='red')

fig.legend(['+1','-1'])

# fig.show()
fig.savefig('test.pdf')
plt.close(fig)
# print(tds)
# print(df_tds)

df_tds['etiqueta_trained'] = df_tds['vector'].apply(lambda vector: get_y_prediction(W_training,vector))
fig,ax = plt.subplots(dim-1)

if dim-1 == 1:
    ax = [ax]

for i in range(dim-1):

    df_tds['y'] = df_tds['vector'].apply(lambda row: row[i+1])

    cls_1 = df_tds[df_tds['etiqueta_trained'] == 1]
    cls_minus_1 = df_tds[df_tds['etiqueta_trained'] == -1]
    ax[i].scatter(cls_1['x'],cls_1['y'],color='C3',label='+1')
    ax[i].scatter(cls_minus_1['x'],cls_minus_1['y'],color='C4',label='-1')


    ax[i].quiver(0,0,W_target[0],W_target[i+1])
    ax[i].quiver(0,0,W_training[0],W_training[i+1])

fig.legend(['+1','-1'])

fig.savefig('trained.pdf')
plt.close(fig)


df_tds['comparacion'] = df_tds['etiqueta']*df_tds['etiqueta_trained']
fig,ax = plt.subplots(dim-1)

if dim-1 == 1:
    ax = [ax]

for i in range(dim-1):

    df_tds['y'] = df_tds['vector'].apply(lambda row: row[i+1])

    cls_1 = df_tds[df_tds['comparacion'] == 1]
    cls_minus_1 = df_tds[df_tds['comparacion'] == -1]
    ax[i].scatter(cls_1['x'],cls_1['y'],color='green',label='Acertado')
    ax[i].scatter(cls_minus_1['x'],cls_minus_1['y'],color='red',label='Fallado')


    ax[i].quiver(0,0,W_target[0],W_target[i+1])
    ax[i].quiver(0,0,W_training[0],W_training[i+1])

    # ax[i].legend(loc='upper right')
    # ax.plot(x_false,y_false,color='red')

fig.legend(['+1','-1'])

# fig.show()
fig.savefig('compare.pdf')
# print(tds)
# print(df_tds)



# Como plotear puntos con distintos colores:
    # fig,ax = plt.subplots()

    # ax.plot(x_true,y_true,color='green')
    # ax.plot(x_false,y_false,color='red')

    # ax.show()

    # st.pyplot(fig)

# Otra forma
    # df_puntos:pd.DataFrame # que tiene 3 columnas, la primera 'x' los valores de x,
    # la segunda 'y' los de y, la tercera 'correct' si acertaron o fallaron

    # px.Scatter(df_puntos,'x','y',color='correct')