import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


def plot_classifier(classifier, x, y):
    # define ranges to plot the figure 画出数据点和边界
    # 增加了一些 buffer = 1.0
    x_min, x_max = min(x[:, 0]) - 1.0, max(x[:, 0]) + 1.0
    y_min, y_max = min(x[:, 1]) - 1.0, max(x[:, 1]) + 1.0

    # 设置网格数据的步长
    step_size = 0.01

    # 定义网格
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # 计算分类器输出结果
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    # 数组维度变形
    mesh_output = mesh_output.reshape(x_values.shape)

    # 用彩图画出分类结果
    plt.figure()
    # 选择配色方案
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)
    # 把训练数据点画在图上
    plt.scatter(x[:, 0], x[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    # 设置图形的取值范围
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    # 设置 X轴 与 Y轴
    plt.xticks(np.arange(int(min(x[:, 0]) - 1), int(max(x[:, 0]) + 1), 1.0))
    plt.yticks(np.arange(int(min(x[:, 1]) - 1), int(max(x[:, 1]) + 1), 1.0))

    plt.show()


if __name__ == '__main__':
    # input data
    X = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2], [1.2, 1.9], [6, 2], [5.7, 1.5], [5.4, 2.2]])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    # 参数 solver 用于设置求解系统方程的算法类型，参数 C 表示正则化强度，数值越小，表示正则化强度越高
    classifier = linear_model.LogisticRegression(solver='liblinear', C=100)

    # train the classifier 训练分类器
    classifier.fit(X, y)

    plot_classifier(classifier, X, y)

