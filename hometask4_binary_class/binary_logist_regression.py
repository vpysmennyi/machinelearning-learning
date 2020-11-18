import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets


dataset = datasets.MNIST('../data', train = True, download=True)

X = dataset.train_data.numpy()
Y = dataset.train_labels.numpy()

# define lists for filtered data for 0 and 1 pictures only
Yn = []
Xn = []

# fiter 0 and 1 pictures
for idx in range(len(Y)):
    if Y[idx] == 0 or Y[idx] == 1:
        Yn.append(Y[idx])

        # flatten picture
        xtmp = []
        for i in range(len(X[idx])):
            xtmp += list(X[idx][i])

        Xn.append(np.array(xtmp))

Xn = np.array(Xn)
Yn = np.array(Yn)
Yn = Yn[:, None]

a = 9e-7  # learning rate
b = np.random.rand(Xn.shape[1],1)/1e5
step = 0.002 # threshold step for PR curve
threshold = 0
recalls = []
precisions = []

Xn = (Xn - Xn.mean()) / Xn.std()

for i in range(10):
  z = Xn @ b
  sigmoid = 1/(1 + np.exp(-z))
  gradb = (Yn - sigmoid).T @ Xn
  b = b + a * gradb.T
  print(b.shape)

# checking recall/precision for thresholds increased by defined step in range from 0 to 1
for k in range(int(divmod(1,step)[0])):
    threshold = k * step
    sigmoid_discr = []

    for i in range(len(sigmoid)):
        if sigmoid[i][0] >= threshold:
            sigmoid_discr.append(1)
        else:
            sigmoid_discr.append(0)

    tp = [1 for x in range(len(sigmoid_discr)) if sigmoid_discr[x] == 1 and 1 == Yn[x][0]]
    fp = [1 for x in range(len(sigmoid_discr)) if sigmoid_discr[x] == 0 and 1 == Yn[x][0]]
    fn = [1 for x in range(len(sigmoid_discr)) if sigmoid_discr[x] == 1 and 0 == Yn[x][0]]
    tn = [1 for x in range(len(sigmoid_discr)) if sigmoid_discr[x] == 0 and 0 == Yn[x][0]]

    try:
        precision = len(tp)/(len(tp) + len(fp))
        recall = len(tp) / (len(tp) + len(fn))
    except (ZeroDivisionError):
        precision = None
        recall = None

    recalls.append(recall)
    precisions.append(precision)

# plotting PR curve
plt.scatter(recalls, precisions)
plt.show()


