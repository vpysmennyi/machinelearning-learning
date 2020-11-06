from sklearn import datasets as ds
import matplotlib.pyplot as plt


class Tree():
    root = []

    def __init__(self, r):
        self.root = [r, [], []]

    def setLeftBranch(self, newBranch):
        self.root.pop(1)
        self.root.insert(1, newBranch)

    def setRightBranch(self, newBranch):
        self.root.pop(2)
        self.root.insert(2, newBranch)

    def getLeftBranch(self):
        return self.root[1]

    def getRightBranch(self):
        return self.root[2]

    def printTree(self, deepness=0):

        for i in range(3):
            if isinstance(self.root[i], Tree):
                self.root[i].printTree(deepness + 1)

            print('   ' * deepness + str(self.root[i]))


def analyze(samples, labels):
    label_values = set(labels)
    main_value = None
    gini = 999.9
    index = 0
    split_samples = list()
    split_labels = list()

    def do_split(samples, coord, split_value):

        left, right = list(), list()
        left_lbl, right_lbl = list(), list()

        for idx in range(len(samples)):
            if samples[idx][coord] <= split_value:
                left.append(samples[idx])
                left_lbl.append(labels[idx])
            else:
                right.append(samples[idx])
                right_lbl.append(labels[idx])
        return [left, right], [left_lbl, right_lbl]

    def calculate_gini(groups):
        total_items = sum([len(group) for group in groups])
        gini_total = 0

        for group in groups:
            group_size = float(len(group))
            sumP2 = 0

            if group_size == 0:
                continue

            for label in label_values:
                sumP2 += (group.count(label) / group_size) ** 2

            gini_total += (1 - sumP2) * (group_size / total_items)

        return gini_total

    for coord in range(len(samples[0])):
        for split_idx in range(len(samples)):
            split_value = samples[split_idx][coord]
            groups, groups_lbl = do_split(samples, coord, split_value)

            gini_current = calculate_gini(groups_lbl)

            if gini_current < gini:
                gini = gini_current
                main_value = split_value
                index = split_idx
                split_samples, split_labels = groups, groups_lbl

    return {'index': index, 'value': main_value, 'gini': gini, 'samples': split_samples, 'labels': split_labels}


def build_tree(tree, depth = 0):
    if not isinstance(tree, Tree):
        exit(1)

    sample_left = tree.root[0]['samples'][0]
    sample_right = tree.root[0]['samples'][1]
    labels_left = tree.root[0]['labels'][0]
    labels_right = tree.root[0]['labels'][1]

    tl, tr = None, None

    if len(sample_left) == 0:
        tree.setLeftBranch(None)
    elif tree.root[0]['gini'] == 0.0 or depth >= max_depth:
        tree.setLeftBranch(max(set(labels_left), key=labels_left.count))
    else:
        tl = Tree(analyze(sample_left, labels_left))
        tree.setLeftBranch(build_tree(tl, depth+1))

    if len(sample_right) == 0:
        tree.setRightBranch(None)
    elif tree.root[0]['gini'] == 0.0 or depth >= max_depth:
        tree.setRightBranch(max(set(labels_right), key=labels_right.count))
    else:
        tr = Tree(analyze(sample_right, labels_right))
        tree.setRightBranch(build_tree(tr, depth+1))
    return tree


def test_decision(tree, test_value):
    if isinstance(tree.root[0], dict):
        if test_value < tree.root[0]['value']:
            branch = tree.getLeftBranch()
            if isinstance(branch, Tree):
                return test_decision(branch, test_value)
            else:
                return branch
        else:
            branch = tree.getRightBranch()
            if isinstance(branch, Tree):
                return test_decision(branch, test_value)
            else:
                return branch
    else:
        return tree.root[0]


def calculate_accuracy(decisions):
    hit = lambda x, y: x == y
    l = [hit(decisions[i][1], decisions[i][2]) for i in range(len(decisions))]
    return (l.count(True)/len(l))*100


samples,labels = ds.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2,
                                         n_clusters_per_class=2)
test_sample,test_label = ds.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2,
                                         n_clusters_per_class=2)
decisionsX, decisionsY = [],[]
max_depth = 20

decisionTree = build_tree(Tree(analyze(samples,labels)))
decisionTree.printTree()

#iterate over X coordinate
for idx in range(len(test_sample)):
    decisionsX.append([test_sample[idx], test_label[idx], test_decision(decisionTree, test_sample[idx][0])])

#iterate over Y coordinate
for idx in range(len(test_sample)):
    decisionsY.append([test_sample[idx], test_label[idx], test_decision(decisionTree, test_sample[idx][1])])

print('Prediction accuracy: ', max([calculate_accuracy(decisionsX),calculate_accuracy(decisionsY)]), '%')