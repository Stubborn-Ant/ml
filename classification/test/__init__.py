# encoding=utf8
# import matplotlib.pyplot as pl
#
# x = range(10)  # 横轴的数据
# y = [i * i for i in x]  # 纵轴的数据
# pl.plot(x, y, 'hb-', label=u'y=x^2曲线图')  # 调用pylab的plot函数绘制曲线
# pl.legend()
# pl.xlabel(u"我是横轴")
# pl.ylabel(u"我是纵轴")
# pl.title(u'图像标题') # 字符串也需要是unicode编码
# pl.xlim(-1, 11)  # 限定横轴的范围
# pl.ylim(-1, 110)  # 限定纵轴的范围
# pl.show()  # 显示绘制出的图
from numpy import *
x = [8, 7, 6, 5]
sortedIndicies = array(x).argsort()
print sortedIndicies
