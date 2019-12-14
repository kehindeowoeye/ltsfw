
import numpy as np
import numpy.matlib
import math
import xlrd
import xlsxwriter
import matplotlib.pyplot as plt
import sys


for iter in range(0,5):
    from d111 import*
    ls = avgh111()
    if iter == 0:
        acc = ls
    else:
        acc = np.vstack((acc,ls))
    del sys.modules['d111']

    from d112 import*
    ls = avgh112();
    acc = np.vstack((acc,ls))
    del sys.modules['d112']

    from d113 import*
    ls = avgh113();
    acc = np.vstack((acc,ls))
    del sys.modules['d113']


    from d114 import*
    ls = avgh114();
    acc = np.vstack((acc,ls))
    del sys.modules['d114']


    from d115 import*
    ls = avgh115()
    acc = np.vstack((acc,ls))
    del sys.modules['d115']


print('average')
print(np.mean(acc))




workbook = xlsxwriter.Workbook('result_.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(acc.T):
    worksheet.write_column(row, col, data)
workbook.close()



