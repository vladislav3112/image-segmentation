import copy
import numpy as np
from PIL import Image, ImageDraw
import math

#основная формула для сегментации
def edge_weight(C_p,C_q):
    return(math.exp(-1/(2 * sigma**2)*(C_p-C_q)**2))

#считываем изображение, строим из него граф, подаём на вход. Попавшие в min cut вершины - чёрные, остальные - белые. 
image = Image.open("banana.jpg").convert('LA') #Открываем изображение. 
draw = ImageDraw.Draw(image) #Создаем инструмент для рисования. 
width = image.size[0] #Определяем ширину. 
height = image.size[1] #Определяем высоту. 	
pix = image.load() #Выгружаем значения пикселей.
matrix = np.asarray(image)
matrix = matrix[:,:,0]
print(matrix)
intence_vals = matrix.ravel()
print(intence_vals)
# lambda and sigma
sigma = 5 

sink = np.where(intence_vals == np.min(intence_vals))[0][0]#!!!!
source = np.where(intence_vals == np.max(intence_vals))[0][0]
print(intence_vals[source])

vertex_count = len(intence_vals)
edge_count = 4 * (height-1) * (width-1) + 4 * (height - 2) * (width - 2) # если считать ребро неориентированным
flow_matrix = np.where(intence_vals == np.max(intence_vals))[0][0]
edge_array = []

#horizontal edges:
for i in range(width - 1):
    for j in range(height):
        weight = edge_weight(matrix[i][j],matrix[i + 1][j]) 
        flow_matrix[i * width + j][(i + 1)* width + j] = (weight)
        #add edge((i,j);(i + 1,j))

#vertical edges:
for i in range(width):
    for j in range(height - 1):
        weight = edge_weight(matrix[i][j],matrix[i][j + 1]) 
        flow_matrix[i * width + j][i * width + j + 1] = (weight)
        #add edge((i,j);(i,j + 1))

#graph fillina template 

print(flow_matrix)
for i in range (vertex_count):
    if(len(np.argwhere(flow_matrix[i]))>0):
        edge_list = np.hstack(np.argwhere(flow_matrix[i])).tolist()
    else:
        edge_list = []
    edge_array.append(edge_list)
n = vertex_count # число вершин
seg_graph = edge_array

c = flow_matrix
f = [[0 for i in range(n)] for j in range(n)]
e = [0 for i in range(n)] # массив размерности n, при заполнении
h = [0 for i in range(n)]
h[0] = n
overflow = [[] for i in range(n + 1)] # храним список вершин с высотой по индексу
overflow[0].extend(seg_graph[0]) # после оптимизации изменить
#далее - алгоритм максимального потока/минимального разреза