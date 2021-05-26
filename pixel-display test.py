import copy
import numpy as np
from PIL import Image, ImageDraw
from numpy.core.records import array

image = Image.open("banana-resize.jpg").convert('L') #Открываем изображение. 
draw = ImageDraw.Draw(image) #Создаем инструмент для рисования. 
width = image.size[0] #Определяем ширину. 
height = image.size[1] #Определяем высоту. 	
pix = image.load() #Выгружаем значения пикселей.
matrix = np.asarray(image)
print(matrix)
intence_vals = matrix.ravel()
print(intence_vals)

matrix2 = matrix.copy()


vertex_array = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
for i in range(height):
    for j in range(width):
        matrix2[i][j] = 255
for elem in vertex_array:
    matrix2[(elem-1) // height][(elem-1) % width] = 0
result = Image.fromarray(matrix2)
result.save('out.jpg')
