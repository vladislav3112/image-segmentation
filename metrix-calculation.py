import copy
import numpy as np
from PIL import Image, ImageDraw

image1 = Image.open(r"segments\moon-segment.jpg").convert('L') #Открываем изображение. 
draw = ImageDraw.Draw(image1) #Создаем инструмент для рисования. 
matrix1 = np.asarray(image1)

image2 = Image.open(r"our-segments\our-moon.jpg").convert('L') #Открываем изображение. 
draw = ImageDraw.Draw(image2) #Создаем инструмент для рисования. 
width = image2.size[0] #Определяем ширину. 
height = image2.size[1] #Определяем высоту. 	
matrix2 = np.asarray(image2)

first_metrix = 0

for i in range(height):
    for j in range(width):
        if (abs(matrix2[i][j] - matrix1[i][j]) < 20):
            first_metrix += 1
print("first metrix: ",first_metrix/(width * height))

union = 0
intersect = 0
for i in range(height):
    for j in range(width):
        if(abs(matrix1[i][j] + 20) > 255) or (abs(matrix2[i][j] + 20) > 255):
            union += 1
        if(abs(matrix1[i][j] + 20) > 255) and (abs(matrix2[i][j] + 20) > 255):
            intersect += 1
print("second metrix: ", intersect/union)