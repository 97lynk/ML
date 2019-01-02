from random import randrange

import numpy as np

# chuyển file size dạng string sang float với đơn vị kilobytes
def convertSize(str):
    if 'M' in str:
        return float(str.replace('M', '')) * 1000.0  # bỏ M và nhân với 1000.0
    elif 'k' in str:
        # print(str, float(str.replace('k', '')) * 1.0)
        return float(str.replace('k', '')) * 1.0 # chỉ bỏ

# xóa đấu + và chuyển sang float
def removePlus(str):
    str = str.replace(',', '')
    return float(str.replace('+', ''));

# random install
# ví dụ
# 1+ => trong khoảng 1 đến 4
# 5+ => trong khoảng 5 đến 9
# 10+ => trong khoảng 10 đến 49
# 50+ => trong khoảng 50 đến 99
# ...
def randomOutput(y):
    yy = np.unique(y)
    yy.sort()
    yy = yy[::-1]
    for j, v in enumerate(y):
        start = 1.0
        end = 1.0
        for index, el in enumerate(yy):
            if el <= v:
                if index != 0:
                    end = yy[index - 1]
                else:
                    end = 1.0
                start = el
                break

        if start == max(yy):
            end += start + randrange(0, 10000)
        if end == start:
            start = 1
            end = 5
        y[j] = randrange(start, end) * 1.0