import numpy as np
import matplotlib.pyplot as plt

# input data
x = np.array([[3,1], [2,5], [1,8], [6,4], [5,2], [3,5], [4,7], [4,-1]])

# labels
y = [0, 1, 1, 0, 0, 1, 1, 0]

# 按照类型标记把样本数据分成两类
class_0 = np.array([x[i] for i in range(len(x)) if y[i] == 0])
class_1 = np.array([x[i] for i in range(len(x)) if y[i] == 1])

# plot input data
plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], color='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], color='black', marker='x')
plt.savefig('./data/figura1.pdf', format='pdf', dpi=1000)

# draw the separator line 画一条分割线
line_x = range(10)
line_y = line_x

# plot labeled data and separator line
plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], color='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], color='black', marker='x')
plt.plot(line_x, line_y, color='black', linewidth=3)

plt.show()


