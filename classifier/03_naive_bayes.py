import numpy as np
from sklearn.naive_bayes import GaussianNB
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


input_file = 'data/data_multivar.txt'

X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])

X = np.array(X)
y = np.array(y)

# 建立 正态分布朴素贝叶斯模型
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)
y_pred = classifier_gaussiannb.predict(X)

# compute accuracy of the classifier
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print ("Accuracy of the classifier =", round(accuracy, 2), "%")

plot_classifier(classifier_gaussiannb, X, y)

###############################################
# Train test split
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
classifier_gaussiannb_new = GaussianNB()
classifier_gaussiannb_new.fit(X_train, y_train)
y_test_pred = classifier_gaussiannb_new.predict(X_test)

# compute accuracy of the classifier
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print ("Accuracy of the classifier =", round(accuracy, 2), "%")

plot_classifier(classifier_gaussiannb_new, X_test, y_test)

###############################################
# Cross validation and scoring functions

num_validations = 5
accuracy = model_selection.cross_val_score(classifier_gaussiannb, X, y, scoring='accuracy', cv=num_validations)
print("Accuracy: " + str(round(100*accuracy.mean(), 2)) + "%")

f1 = model_selection.cross_val_score(classifier_gaussiannb, X, y, scoring='f1_weighted', cv=num_validations)
print ("F1: " + str(round(100*f1.mean(), 2)) + "%")

precision = model_selection.cross_val_score(classifier_gaussiannb, X, y, scoring='precision_weighted', cv=num_validations)
print ("Precision: " + str(round(100*precision.mean(), 2)) + "%")

recall = model_selection.cross_val_score(classifier_gaussiannb, X, y, scoring='recall_weighted', cv=num_validations)
print ("Recall: " + str(round(100*recall.mean(), 2)) + "%")