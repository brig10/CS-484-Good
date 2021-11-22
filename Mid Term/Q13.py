import numpy as np
import math

arr = np.array([[65, 304, 530, 487, 140],
                [74, 185, 160,  55,  16],
                [33, 228, 623, 755, 363],
                [90, 290, 349, 213,  40]])

print(arr)


arr_1 = np.array([arr[0], arr[1]+arr[2]+arr[3]])
arr_2 = np.array([arr[0]+arr[1], arr[2]+arr[3]])
arr_3 = np.array([arr[0]+arr[1]+arr[2], arr[3]])

def ent(arr):
    t1 = 0
    t2 = 0
    for i in range(5):
        t1 += arr[0][i]
        t2 += arr[1][i]
    t = t1+t2

    e1 = 0
    e2 = 0
    for i in range(5):
        e1 += arr[0][i]/t1*math.log2(arr[0][i]/t1)
        e2 += arr[1][i]/t2*math.log2(arr[1][i]/t2)
    e1 *= -1
    e2 *= -1

    print((t1/t)*e1 + (t2/t)*e2)

ent(arr_1)
ent(arr_2)
ent(arr_3)

arr_4 = np.array([arr[0], arr[1]+arr[2]])
arr_5 = np.array([arr[0]+arr[1], arr[2]])
print('Second Layer')

ent(arr_4)
ent(arr_5)
