import numpy as np
import statistics

class DecisionTree():
    
    """
    
    Decision Tree Classifier
    
    Attributes:
        root: Root Node of the tree.
        max_depth: Max depth allowed for the tree
        size_allowed : Min_size split, smallest size allowed for split 
        n_features: Number of features to use during building the tree.(Random Forest)
        n_split:  Number of split for each feature. (Random Forest)
    
    """

    def __init__(self, max_depth = 1000, size_allowed = 1, n_features = None, n_split = None):
        
        """
        
            Initializations for class attributes.
        """
        
        self.root = None
        if max_depth == None:
            self.max_depth = 1000
        else:         
            self.max_depth = max_depth     
        self.size_allowed = size_allowed      
        self.n_features = n_features        
        self.n_split = n_split           
    
    
    class Node():
        
        """
            Node Class for the building the tree.

            Attribute: 
                threshold: The threshold like if x1 < threshold, for spliting.
                feature: The index of feature on this current node.
                left: Pointer to the node on the left.
                right: Pointer to the node on the right.
                pure: Bool, describe if this node is pure.
                predict: Class, indicate what the most common Y on this node.

        """
        def __init__(self, threshold = None, feature = None):
            """
            
                Initializations for class attributes.
            """
            
            
            self.threshold = threshold  
            self.feature = feature    
            self.left = None
            self.right = None
            self.pure = None
            self.depth = None
            self.predict = None
    
    
    def entropy(self, lst):
        
        """
            Function Calculate the entropy given lst.
            
            Attributes: 
                entro: variable store entropy for each step.
                classes: all possible classes. (without repeating terms)
                counts: counts of each possible classes.
                total_counts: number of instances in this lst.
                
            lst is vector of labels.
        """
        
        entro = 0  
        classes, counts = np.unique(lst, return_counts=True)  
        total_counts = np.sum(counts)  
        for count in counts:       
            p = count / total_counts
            if p > 0:
                entro -= p * np.log2(p)
        return entro

    def information_gain(self, lst, values, threshold):
        """
        
            Function Calculate the information gain, by using entropy function.

            lst is vector of labels.
            values is vector of values for individual feature.
            threshold is the split threshold we want to use for calculating the entropy.   
        """
        
        left = values <= threshold
        right = values > threshold
        left_prop = np.sum(left) / len(values)     
        right_prop = 1 - left_prop

        left_entropy =  self.entropy(lst[left]) 
        right_entropy = self.entropy(lst[right])
        
        S = self.entropy(lst)
        S_prime = left_entropy * left_prop + right_entropy * right_prop

        return S - S_prime
    
    def find_rules(self, data):
        
        """
        
            Helper function to find the split rules.
            
            data is a matrix or 2-D numpy array, represnting training instances. 
            Each training instance is a feature vector. 
        """
        n,m = data.shape        
        rules = []      
          
        for i in range(m):          
            unique_values = np.unique(data[:, i])       
            diffs  = (unique_values[:-1] + unique_values[1:]) / 2      
            rules.append(diffs)     
                    
        return rules
    
    def next_split(self, data, label):
        """
            Helper function to find the split with most information gain, by using find_rules, and information gain.
            
            data is a matrix or 2-D numpy array, represnting training instances. 
            Each training instance is a feature vector. 
            
            label contains the corresponding labels. There might be multiple (i.e., > 2) classes.
        """
        
        rules = self.find_rules(data)             
        max_info = -np.inf          
        num_col = None        
        threshold = None      
        
        if self.n_features == None:
            index_col = list(range(data.shape[1]))
        else:
            if type(self.n_features) == int: 
                num_index = self.n_features
            elif self.n_features == "sqrt":
                num_index = int(np.sqrt(data.shape[1]))  
            np.random.seed()  
            index_col = np.random.choice(data.shape[1], num_index, replace = False)

        for i in index_col:
            count_temp_rules = len(rules[i])
            
            if self.n_split == None:
                index_rules = range(count_temp_rules)
            else:
                if type(self.n_split) == int:
                    num_rules = np.min(self.n_split, count_temp_rules)   
                elif self.n_split == "sqrt":
                    num_rules = int(np.sqrt(count_temp_rules))  
                np.random.seed()
                
                index_rules = np.random.choice(count_temp_rules, num_rules, replace = False)
            
            for j in index_rules:
                info = self.information_gain(label, data[:, i], rules[i][j])     
                if info > max_info:  
                    max_info = info
                    num_col = i
                    threshold = rules[i][j]
   
        return threshold, num_col
        
    def build_tree(self, X, y, depth):
        
            """
                Helper function for building the tree.
            """
            
            
            first_threshold, first_feature = self.next_split(X, y)
            current = self.Node(first_threshold, first_feature)  
            current.depth = depth
            
            """
                Base Case 1: Check if we pass the max_depth, check if the first_feature is None, min split size.
                If some of those condition met, change current to pure, and set predict to the most popular label
                and return current   
            """
            
            if depth >= self.max_depth or first_feature == None or len(y) < self.size_allowed:
                current.predict = statistics.mode(y)
                current.pure = True
                return current
            
            """
               Base Case 2: Check if there is only 1 label in this node, change current to pure, and set predict to the label
            """
            
            if len(np.unique(y)) == 1:
                current.predict = y[0]
                current.pure = True
                return current

            left_index = X[:, first_feature] <= first_threshold
            right_index = X[:, first_feature] > first_threshold
            left_size = len(y[left_index])
            right_size = len(y[right_index])
            
            """
                Base Case 3: If we either side is empty, change current to pure, and set predict to the label
            """
            if left_size == 0 or right_size == 0:
                current.predict = y[0]
                current.pure = True
                return current
            
            
            left_X, left_y = X[left_index, :], y[left_index]
            current.left = self.build_tree(left_X, left_y, depth + 1)
                
            right_X, right_y = X[right_index, :], y[right_index]
            current.right = self.build_tree(right_X, right_y, depth + 1)
            
            return current
    

        
    def fit(self, X, y):
        
        """
            The fit function fits the Decision Tree model based on the training data. 
            
            X_train is a matrix or 2-D numpy array, represnting training instances. 
            Each training instance is a feature vector. 

            y_train contains the corresponding labels. There might be multiple (i.e., > 2) classes.
        """
        
        self.root = self.build_tree(X, y, 1)
        

        self.for_running = y[0]
        return self
            
    def ind_predict(self, inp):
        """
            Predict the most likely class label of one test instance based on its feature vector x.
        """
        cur = self.root  
        while not cur.pure:  
            
            feature = cur.feature  
            threshold = cur.threshold 
            
            if inp[feature] <= threshold:  
                cur = cur.left
            else:
                cur = cur.right
        return cur.predict
    
    def predict(self, inp):
        """
            X is a matrix or 2-D numpy array, represnting testing instances. 
            Each testing instance is a feature vector. 
            
            Return the predictions of all instances in a list.
        """
        
        result = []
        for i in range(inp.shape[0]):
            result.append(self.ind_predict(inp[i]))
        return result
    