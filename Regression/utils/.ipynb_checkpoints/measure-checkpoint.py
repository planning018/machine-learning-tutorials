import matplotlib.pyplot as plt



# measure performance 计算回归准确性
def print_measure_performance(test, test_predict):
    import sklearn.metrics as sm
    # 平均绝对误差
    print("Mean absolute error = ", round(sm.mean_absolute_error(test, test_predict), 2))
    # 均方误差
    print("Mean squared error = ", round(sm.mean_squared_error(test, test_predict), 2))
    # 中位数绝对误差
    print("Median absolute error = ", round(sm.median_absolute_error(test, test_predict), 2))
    # 解释方差分
    print("Explain variance score = ", round(sm.explained_variance_score(test, test_predict), 2))
    # R方得分
    print("R2 score = ", round(sm.r2_score(test, test_predict), 2))