import copy
import numpy as np

def push(u, v, f, e, c, graf, h, H, overflow):
    d = min(e[u], c[u][v] - f[u][v])
    f[u][v] += d
    f[v][u] -= d
    e[u] -= d
    if f[u][v] != 0 and u not in graf[v]: 
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
                            if H < h[v] and e[v] > 0:
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
        graf[v].append(0) # out of range when 1 edge
    graf[0] = []


def c_for_opt(c, f):
    c_new = copy.deepcopy(c)
    for i in range(n):
        for j in range(n):
            if f[i][j] > 0:
                c_new[i][j] = c[i][j] - f[i][j]
    return c_new

#считывание данных и их преобразование
vertex_count, edge_count = map(int, input().split())
flow_matrix = np.zeros((vertex_count, vertex_count))
vertex_array = []
edge_array = []

for i in range (edge_count):
    vertex1, vertex2, flow = map(int, input().split())
    flow_matrix[vertex1 - 1][vertex2 - 1] = int(flow)

#print(flow_matrix)
edge_array = []
for i in range (vertex_count):
    if(len(np.argwhere(flow_matrix[i]))>0):
        edge_list = np.hstack(np.argwhere(flow_matrix[i])).tolist()
    else:
        edge_list = []
    edge_array.append(edge_list)
if (vertex_count != edge_count):
    for i in range(edge_count - vertex_count):
        edge_array.append([])

n = vertex_count # число вершин
graf = edge_array
#print(graf)
# храним граф в виде списка смежных вершин
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

m = 10 # частота оптимизации
count = 0



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
    count += 1
    if count > m:
        count = 0
        H = global_r_opt(c_for_opt(c, f), h, H, overflow, graf0, graf1)
    #print('f =', f)
    #print('e =', e)
    #print('h =', h)
    #print('H =', H)
    #print(overflow)
    #input() # заглушка для пошагового вывода
    
print('Ответ:', e[n - 1])