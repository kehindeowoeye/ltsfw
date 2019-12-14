
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import math
import xlrd
import xlsxwriter
import matplotlib.pyplot as plt
import sys


for iter in range(0,5):
    from d112 import*
    ls = avgh112()
    if iter == 0:
        acc = ls
    else:
        acc = np.vstack((acc,ls))
    del sys.modules['d112']

    from d122 import*
    ls = avgh122();
    acc = np.vstack((acc,ls))
    del sys.modules['d122']

    from d132 import*
    ls = avgh132();
    acc = np.vstack((acc,ls))
    del sys.modules['d132']


    from d142 import*
    ls = avgh142();
    acc = np.vstack((acc,ls))
    del sys.modules['d142']


    from d152 import*
    ls = avgh152()
    acc = np.vstack((acc,ls))
    del sys.modules['d152']


print('average')
print(np.mean(acc))




workbook = xlsxwriter.Workbook('result_.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(acc.T):
    worksheet.write_column(row, col, data)
workbook.close()



