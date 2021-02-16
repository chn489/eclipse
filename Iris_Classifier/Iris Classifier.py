import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('D:\desktop\py_csv\dataset2.CSV')
data_feature = np.array(data.iloc[:, [0, 1, 2, 3]])
data_target = np.array(data.iloc[:, [4]])
# print(data_feature,'\n',data_target)

feature_train, feature_test, target_train, target_test = train_test_split(data_feature, data_target, test_size=0.33,
                                                                          random_state=0)
dt_model = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=3)
dt_model.fit(feature_train, target_train)
predict_results = dt_model.predict(feature_test)

print(predict_results, '\n')
print(target_test, '\n')
print(accuracy_score(predict_results, target_test))

fig = plt.figure(figsize=(6, 6))
tree.plot_tree(dt_model, filled='True', feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'],
               class_names=['setosa', 'versicolor', 'virginica'])
plt.show()
