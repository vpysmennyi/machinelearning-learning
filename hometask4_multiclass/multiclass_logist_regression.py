import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def flatten(pictures):
    pictures_f = []

    for i in range(len(pictures)):
        pictmp = []

        for j in range(len(pictures[i])):
            pictmp += list(pictures[i][j])

        pictures_f.append(pictmp)
    return np.array(pictures_f)


def predict(x, bs, classes):
    preds = [np.argmax([sigmoid(xi @ bc) for bc in bs]) for xi in x]
    return [classes[p] for p in preds]


def accuracy_rate(labels, pred_labels):
    classes = np.unique(labels)
    acc_rate = {}

    for c in classes:
        counter = []

        for idx in range(len(labels)):
            if labels[idx] == c:
                if labels[idx] == pred_labels[idx]:
                    counter.append(True)
                else:
                    counter.append(False)
            else:
                continue

        acc_rate[c] = (counter.count(True)/len(counter))*100
    return acc_rate


dataset = datasets.MNIST('../data', train = True, download=True)

pictures = dataset.train_data.numpy()
labels = dataset.train_labels.numpy()

classes = np.unique(labels)

# flatten
pictures_f = flatten(pictures)

a = 9e-3  # learning rate
b0 = np.random.rand(pictures_f.shape[1], 1) / 1e5
bs = []

# one vs all approach
for c in classes:
    labels_bin = np.where(c == labels, 1, 0)
    labels_bin = labels_bin[:, None]
    b = b0

    for epoch in range(10):
        z = pictures_f @ b
        sigm = sigmoid(z)
        gradb = (labels_bin - sigm).T @ pictures_f
        b = b + a * gradb.T

    bs.append(b)

# doing prediction on same dataset
pred_labels = predict(pictures_f, bs, classes)

#calculate and print accuracy for each class
acc = accuracy_rate(labels, pred_labels)
for a in acc.keys():
    print(a, ' : ', acc[a])








