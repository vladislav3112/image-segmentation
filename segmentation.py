import copy
import numpy as np
from PIL import Image, ImageDraw

# max flow
def push(u, v, f, e, c, graf, h, H, overflow):
    d = min(e[u], c[u][v] - f[u][v])
    f[u][v] += d
    f[v][u] -= d
    e[u] -= d
    if f[u][v] != 0 and u not in graf[v]: #test6 out of range graf have max idx 98 but v is 99
        graf[v].append(u)
    if e[u] == 0:
        overflow[H].remove(u)
        if not overflow[H]:
            H = newH(H, overflow)# перерасчет наивысшей
    if c[u][v] == f[u][v]:
        graf[u].remove(v)
    if e[v] == 0 and v != n - 1 and v != 0 and v not in overflow[h[v]]:
        overflow[h[v]].append(v)
        if h[v] > H:
            H = h[v]
    e[v] += d
    return H


def newH(H, overflow):
    for i in range(H - 1, -1, -1):
        if overflow[i]:
            return i
    return -1

    
def key_min(u, h, c, f, v):
    if c[u][v] - f[u][v]> 0 and h[v] >= h[u]:
        return h[v]
    else:
        return 10000000


def lift(u, h, c, f, graf, H, overflow):
    min_h = min(graf[u], key = lambda k: key_min(u, h, c, f, k))
    overflow[h[u]].remove(u)
    h[u] = h[min_h] + 1
    if len(overflow) <= h[u]:
        overflow.append([u])
        H = h[u]
    else:
        overflow[h[u]].append(u)
        if H < h[u]:
            H = h[u]
    return H


def global_r_opt(c, h, H, overflow, graf0, graf1):
    listT = [[n - 1]]
    listPoints = [True for i in range(n)]
    listPoints[n - 1] = False
    level = 0
    temp = [0]
    while temp:
        temp = []
        for u in listT[level]:
            for v in graf1[u]:
                if listPoints[v]:
                    if c[v][u] > 0:
                        temp.append(v)
                        if h[v] < len(listT):
                            if e[v] > 0:
                                overflow[h[v]].remove(v)
                                overflow[len(listT)].append(v)
                            h[v] = len(listT)
                            if H < h[v] and e[v] >= 0:
                                H = h[v]
                        listPoints[v] = False
        level += 1
        if temp:
            listT.append(temp)
    listS = [[0]] # переделать через изначальный граф, сохранить граф
    temp = [0]
    level = 0
    listPoints[0] = False
    while temp:
        temp = []
        for u in listS[level]:
            for v in graf0[u]:
                if listPoints[v]:
                    if c[u][v] > 0:
                        temp.append(v)
                        if h[v] < len(listS):
                            if e[v] > 0:
                                overflow[h[v]].remove(v)
                                overflow[len(listS)].append(v)
                            h[v] = len(listS)
                            if H < h[v] and e[v] > 0:
                                H = h[v]
                        listPoints[v] = False
        level += 1
        if temp:
            listS.append(temp)
    return H


def count_e(graf, e, c, f):
    for v in graf[0]:
        f[0][v] = c[0][v]
        f[v][0] = -c[0][v]
        e[v] = f[0][v]
        # overflow.append(v)
        if(v != 0):
            graf[v].append(0) # out of range when 1 edge
    graf[0] = []


def c_for_opt(c, f):
    c_new = copy.deepcopy(c)
    for i in range(n):
        for j in range(n):
            if f[i][j] > 0:
                c_new[i][j] = c[i][j] - f[i][j]
    return c_new


#основная формула для сегментации   
def edge_weight(C_p,C_q):
    number = -(C_p-C_q)*(C_p-C_q)/(2 * sigma * sigma)
    return np.exp(number)

#считываем изображение, строим из него граф, подаём на вход. Попавшие в min cut вершины - чёрные, остальные - белые. 
image = Image.open("ceramic-gr-100.jpg").convert('L') #Открываем изображение. 
draw = ImageDraw.Draw(image) #Создаем инструмент для рисования. 
width = image.size[0] #Определяем ширину. 
height = image.size[1] #Определяем высоту. 	
pix = image.load() #Выгружаем значения пикселей.
matrix = np.asarray(image)
intence_vals = matrix.ravel()
# lambda and sigma 
sigma = 10

#0 - чёрный цвет, 255 - белый
#matrix2[(elem-1) % width][(elem-1) // height]

vertex_count = len(intence_vals)
flow_matrix = np.zeros((vertex_count + 2, vertex_count + 2))
edge_array = []


#пользовательский ввод:
print("Введите число вершин фона \n")
bcg_pixel_num = int(input())
# гистограмы (или частота встречаемости интенсивности пикселя) фона:
hist_bcg = np.zeros(256)

bcg = set() #вершины фона
print("Введите координаты пикселей фона через пробел \n")

for i in range (bcg_pixel_num):
    x, y = map(int,input().split())
    x-=1
    y-=1
    bcg.add(y * width + x + 1)
    hist_bcg[matrix[y][x]] += 1 # прибавляем число вхождений пикселя заднонной интенсивности на 1 

#!!! в matrix почему-то инвертированы размерноси по сравнению с size[0] и size[1]
print("Введите число вершин объекта \n")
obj = set()
obj_pixel_num = int(input())
# гистограма (или частота встречаемости интенсивности пикселя) объекта:
hist_obj = np.zeros(256)
print("Введите координаты пикселей объекта через пробел \n")
for i in range (obj_pixel_num):
    x, y = map(int,input().split())
    x-=1
    y-=1
    obj.add(y * width + x + 1)
    hist_obj[matrix[y][x]] += 1 # прибавляем число вхождений пикселя заднонной интенсивности на 1 

vertex_set = set()
for i in range(flow_matrix.shape[0]):
    vertex_set.add(i)

vertex_set = vertex_set ^ bcg ^ obj # вершины, не лежащие в объекте и фоне

lam = 0.001

#special edges:
for pixel in bcg:
    flow_matrix[0, pixel] = 0
    flow_matrix[pixel, flow_matrix.shape[0] - 1] = 100000

for pixel in obj:
    flow_matrix[0,pixel] = 100000
    flow_matrix[pixel, flow_matrix.shape[0] - 1] = 0

#histogram and lambda edges:
for pixel in vertex_set:
    if pixel != 0 and pixel != flow_matrix.shape[0] - 1:
        flow_matrix[0,pixel] = - lam * np.log(hist_bcg[intence_vals[pixel - 1]]/bcg_pixel_num + 1e-4) # частота встречаемости / кол-во = вероятность
        flow_matrix[pixel, flow_matrix.shape[0] - 1] = - lam * np.log(hist_obj[intence_vals[pixel - 1]]/obj_pixel_num + 1e-4)# * ln(Hist_obj[pixel])

#horizontal edges:
for i in range(0 , height - 1):
    for j in range(0 , width - 1):
        if i < height - 1:
            weight = edge_weight(int(matrix[i][j]),int(matrix[i + 1][j])) 
            flow_matrix[i * width + j + 1][(i + 1) * width + j + 1] = weight
        if j < width - 1:
            weight = edge_weight(int(matrix[i][j]),int(matrix[i][j + 1])) 
            flow_matrix[i * width + j + 1][i * width + j + 1 + 1] = weight
        if i > 0:
            weight = edge_weight(int(matrix[i][j]),int(matrix[i - 1][j])) 
            flow_matrix[i * width + j + 1][(i - 1) * width + j + 1 + 1] = weight
        if j > 0:
            weight = edge_weight(int(matrix[i][j]),int(matrix[i][j - 1])) 
            flow_matrix[i * width + j + 1][i * width + (j - 1) + 1 + 1] = weight
#graph fillina template 

#print(flow_matrix)
for i in range (vertex_count + 2):
    if(len(np.argwhere(flow_matrix[i]))>0):
        edge_list = np.hstack(np.argwhere(flow_matrix[i])).tolist()
    else:
        edge_list = []
    edge_array.append(edge_list)
n = vertex_count + 2 # число вершин
graf = edge_array
graf0 = copy.deepcopy(graf) # сохраним исходный граф
graf1 = [[] for i in range(n)] # сохраняем граф в обратном виде
index = 0
for u in graf:
     for v in u:
         graf1[v].append(index)
     index += 1
# заполнение за линейное время при вводе
# написать ввод


c = flow_matrix
f = [[0 for i in range(n)] for j in range(n)]
e = [0 for i in range(n)] # массив размерности n, при заполнении
h = [0 for i in range(n)]
h[0] = n
overflow = [[] for i in range(n + 1)] # храним список вершин с высотой по индексу
overflow[0].extend(graf[0]) # после оптимизации изменить
if n-1 in overflow[0]:
    overflow[0].remove(n-1)
count_e(graf, e, c, f) # проталкивание в смежные с истоком
H = global_r_opt(c, h, -1, overflow, graf0, graf1) # первая оптимизация

m = 50 # частота оптимизации
count = 0
# overflow.sort(key = lambda k: h[k], reverse = True)
# отсортировать по высоте от максимума к минимуму
#print(graf)
#print('c =', c)
#print('f =', f)
#print('e =', e)
#print('h =', h)
#print('H =', H)
#print(overflow)


while H >= 0:
    if(len(overflow[H])>0):
        u = overflow[H][0] 
    else:   
        H = H-1
        continue
    v = n
    for vi in graf[u]:
        if h[vi] + 1 == h[u] and c[u][vi] - f[u][vi] > 0:
            v = vi
            break
    if v < n:
        H = push(u, v, f, e, c, graf, h, H, overflow)
    else:
        H = lift(u, h, c, f, graf, H, overflow)
    #count += 1
    #if count > m:
    #   count = 0
    #    H = global_r_opt(c_for_opt(c, f), h, H, overflow, graf0, graf1)

#print('Ответ:', e[n - 1])
#print(graf)
visited = set() # Set to keep track of visited nodes.

def dfs(visited, graph, node):
    if node not in visited:
        #print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

dfs(visited, graf, 0)

matrix2 = matrix.copy()

for i in range(height):
    for j in range(width):
        matrix2[i][j] = 0
for elem in visited:
    if elem != 0 and elem < width * height:
        matrix2[(elem - 1) // width][(elem - 1) % width] = 255 
result = Image.fromarray(matrix2)
result.save('our-banana.jpg') 