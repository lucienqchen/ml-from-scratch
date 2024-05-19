import numpy as np
from decision_tree import DecisionTree

class RandomForest():
    
    """
    RandomForest Classifier
    
    Attributes:
        n_trees: Number of trees. 
        trees: List store each individule tree
        n_features: Number of features to use during building each individule tree.
        n_split: Number of split for each feature.
        max_depth: Max depth allowed for the tree
        size_allowed : Min_size split, smallest size allowed for split 
    """
    
    def __init__(self,n_trees = 10, n_features = 'sqrt', n_split = 'sqrt', max_depth = None, size_allowed = 1):
        
        """
            Initilize all Attributes.
        """
        self.n_trees = n_trees
        self.trees = []
        self.n_features = n_features
        self.n_split = n_split
        self.max_depth = max_depth
        self.size_allowed = size_allowed
        
        
    def fit(self, X,y):
        
        """
            The fit function fits the Random Forest model based on the training data. 
            
            X_train is a matrix or 2-D numpy array, represnting training instances. 
            Each training instance is a feature vector. 
            
            y_train contains the corresponding labels. There might be multiple (i.e., > 2) classes.
        """
        
        for i in range(self.n_trees):
            np.random.seed()
            temp_clf = DecisionTree(
                max_depth=self.max_depth,
                size_allowed=self.size_allowed,
                n_features=self.n_features,
                n_split=self.n_split,  
            )
            temp_clf.fit(X, y)
            self.trees.append(temp_clf)
        return self
            
    def ind_predict(self, inp):
        
        """
            Predict the most likely class label of one test instance based on its feature vector x.
        
            TODO: 4. Modify the following code to predict using each Decision Tree.
        """
        result = [tree.ind_predict(inp) for tree in self.trees]
        labels, counts = np.unique(result, return_counts=True)
        max_count = np.max(counts)
        max_labels = labels[counts == max_count]
        return np.random.choice(max_labels)
    
    def predict_all(self, inp):
        
        """
            X is a matrix or 2-D numpy array, represnting testing instances. 
            Each testing instance is a feature vector. 
            
            Return the predictions of all instances in a list.
        """
        result = []
        for i in range(inp.shape[0]):
            result.append(self.ind_predict(inp[i]))
        return result
    