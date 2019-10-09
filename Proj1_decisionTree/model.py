from anytree import NodeMixin, RenderTree, LevelOrderIter, PreOrderIter
import numpy as np
import math

class decisionTree(NodeMixin):
    
    # this tree class uses NodeMixin in anytree, mainly for the convenience of
    # its tree printing function, and its breadth first search iterator, which
    # is used to iterate through the tree to find the best node to prune. Otherwise
    # the methods are build from scratch.
    
    
    def __init__(self, parent = None, *args):
        super().__init__(*args)
        # two children, lesf is less than the splitting point
        self.left = None
        self.right = None
        self.feature = None # splitting feature
        self.split = None # split point
        self.isLeaf = False # if it is leaf
        self.label = None # what is the class node decides if it is leaf
        self.parent = parent # parent pf this node

        
###############build a tree######################

    # recursively grow a tree by splitting the nodes.
    # grow greadily until the subsets of training data we are left with at a set of children nodes are pure 
    # (i.e., they contain only training examples of one class) or the feature vectors associated 
    # with a node are all identical (in which case we cannot split them) but their labels are different.
    def grow(self, X, y, criterion = 'gini'):
        # classify a node
        self.label = max([0,1], key = lambda lab: len(y[y == lab]))
        # stop if all data in the same category, or all features are the same
        # make it a leaf
        if np.all(y == 0) or np.all(y == 1) or np.all(X == X[0,:]):
            self.isLeaf = True
            return
        # split based on best feature
        bestfeature, bestsplit = self.__computeOptimalSplit(X, y, criterion)
        self.feature = bestfeature
        self.split = bestsplit
        ind_lessthansplit = X[:, bestfeature] <= bestsplit
        X_l = X[X[:, bestfeature] <= bestsplit]
        self.left = decisionTree(parent = self)
        self.left.lessthansplit = True
        self.right = decisionTree(parent = self)
        self.right.lessthansplit = False
        self.left.grow(X[ind_lessthansplit], y[ind_lessthansplit], criterion)
        self.right.grow(X[~ind_lessthansplit], y[~ind_lessthansplit], criterion)
        
                
    # return the best single split that produces the maximum gain
    def __computeOptimalSplit(self, X, y, criterion): 
        node_impurity = self.__cal_impurity(y, criterion)
        best_gain = 0.
        best_feature = None
        best_split = None
        # loop over features
        for dim in range(X.shape[1]):
            # find splitting points
            feature_values = np.unique(X[:, dim])
            splits = (feature_values[:-1] + feature_values[1:]) / 2.0         
            # loop over each splitting value
            for split in splits:
                # split y
                y0 = y[X[:, dim] <= split]
                y1 = y[X[:, dim] > split]
                ratio0 = y0.shape[0] / y.shape[0]
                ratio1 = y1.shape[0] / y.shape[0]
                # cal. gain
                impure0 = self.__cal_impurity(y0, criterion)
                impure1 = self.__cal_impurity(y1, criterion)
                # weighted gini
                if criterion == 'gini':
                    gain = node_impurity - (ratio0 * impure0 + ratio1 * impure1)
                # gain ratio as used in C4.5
                else:
                    gain = (node_impurity - (ratio0 * impure0 + ratio1 * impure1))
                    gain = gain / (-ratio0 * math.log2(ratio0) - ratio1 * math.log2(ratio1))
                # determine if it is the best
                if gain > best_gain or best_feature is None:
                    best_feature = dim
                    best_split = split
                    best_gain = gain
        return best_feature, best_split
            
    # calculate impurity based on selected criterion
    def __cal_impurity(self, y, criterion):
        if criterion == 'gini':
            return self.__gini(y)
        else:
            return self.__entropy(y)
        
    # gini index for binary class
    def __gini(self,y):
        p0 = len(y[y == 0]) / len(y)
        p1 = 1 - p0;
        return 1 - p0**2 - p1**2
    
    #entropy for binary class 
    def __entropy(self, y):
        p0 = len(y[y == 0]) / len(y)
        p1 = 1 - p0;
        if p0 in (0,1):
            return 0
        else:
            return -p0*math.log2(p0) - p1*math.log2(p1)

####### use a tree after building it######
    # print tree
    def print_tree(self, feature_names):
        for pre, _, node in RenderTree(self):
            if node.isLeaf:
                nodename = node.label
            else:
                nodename = feature_names[node.feature] + '<=' + str(round(node.split,2))
            treestr = "%s%s" % (pre, nodename)
            print(treestr)

    # classify the data after we build a tree
    def predict(self, X):
        y = []
        for row in range(len(X)):
            row_label = self.__classify_one_record(X[row, :])
            y.append(row_label)
        return np.array(y)
            
                
    # claasify one record
    def __classify_one_record(self, X):
         # return the label if it is leaf node
        if self.isLeaf:
            return self.label
        # else pass to the child node
        elif X[self.feature] <= self.split:
            return self.left.__classify_one_record(X)
        else:
            return self.right.__classify_one_record(X)        
           
    # cal. accuracy
    def cal_accuracy(self, ypred, yture):
        correct = np.sum(ypred == yture)
        return correct / len(ypred)
    
    # count total number of nodes
    def count_nodes(self):
        if self.isLeaf:
            return 1
        else:
            return self.left.count_nodes() + 1 + self.right.count_nodes()
        
    # count total number of leaves
    def count_leaves(self):
        if self.isLeaf:
            return 1
        else:
            return self.left.count_leaves() + self.right.count_leaves()
    
 ##########prune a tree ####################

    #an exhaustive search for the single node for which removing it (and its children) produces
    #the largest increase (or smallest decrease) in classification accuracy as measured using 
    #validation data.
    def pruneSingleGreedyNode(self, X_val, y_val):
        best_tree = None
        best_accu = float('-inf')
        list_pruned_trees = self.__list_all_trees()
        for tree in list_pruned_trees:
            y_pred = tree.predict(X_val)
            accu = tree.cal_accuracy(y_pred, y_val)
            if accu >= best_accu:
                best_accu = accu
                best_tree = tree
        return best_tree

    # return a listing of all possible trees that can be formed by removing a single node from a base tree    
    def __list_all_trees(self):
        list_trees = []
        for node_deleted in LevelOrderIter(self):
            if node_deleted.isLeaf:
                continue
            else:
                node_child = node_deleted.children
                node_deleted.children = []
                node_deleted.isLeaf = True
                tree_copy = copy.deepcopy(self)
                list_trees.append(tree_copy)
                node_deleted.children = node_child
                node_deleted.isLeaf = False
        return list_trees

