from sklearn import datasets, svm
import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(y)
np.random.seed(118)
np.random.shuffle(X)
np.random.seed(118)
np.random.shuffle(y)
print(y)
model = svm.SVC(C=0.8,gamma=0.1,decision_function_shape='ovo')
train_size, train_accuracy, test_accuracy = learning_curve(model, X, y, cv=10, scoring='accuracy')
plt.figure()
plt.plot(train_size, np.mean(train_accuracy, axis=1), label = 'train_accuracy')
plt.plot(train_size, np.mean(test_accuracy, axis=1), label = 'test_accuracy')
plt.xlabel('train_size')
plt.ylabel('accuracy')
plt.title('learning_curve')
plt.legend()
plt.show()