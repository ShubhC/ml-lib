import numpy as np

class TreeEmsemble:
    def __init__(self,num_trees,bagging_fraction,max_depth,max_leaves=1000,min_samples=1,features_fraction=1.0):
        assert num_trees > 0 and bagging_fraction > 0 and max_depth > 0 and min_samples > 0 and 0 < features_fraction <= 1
        self.num_trees, self.bagging_fraction, self.max_depth, self.min_samples, self.features_fraction = num_trees,bagging_fraction,max_depth,min_samples,features_fraction
        self.trees = [DecisionTree(self.max_depth,self.min_samples) for _ in range(self.num_trees)]

    def train(self,x,y):
        self.n = x.shape[0]
        self.c = x.shape[1]

        for tree_idx in range(self.num_trees):
            # get random rows
            row_idx = np.random.permutation(self.n)[:int(self.bagging_fraction*self.n)]
            # get random columns
            col_idx = np.random.permutation(self.c)[:int(self.features_fraction*self.c)]
            
            x_decision_tree = x.iloc[row_idx, col_idx]
            y_decision_tree = y[row_idx]
            
            self.trees[tree_idx].train(x_decision_tree,y_decision_tree)

    def predict(self,x_test):
        return np.array([self.predict_single_row(x_test.iloc[i]) for i in range(0,x_test.shape[0])])
    
    def predict_single_row(self,x_test):
        return np.mean([tree.predict_single_row(x_test) for tree in self.trees])

class DecisionTree:
    def __init__(self,max_depth,min_samples):
        self.max_depth,self.min_samples = max_depth,min_samples

    def train(self,x,y):
        self.mean_val = np.mean(y)
        self.split_col = -1
        self.split_val = -1
        
        # stop constructing the tree when
        # 1. max depth is 1
        # 2. all elements have same independent value
        # 3. there are less samples than min_samples
        if self.max_depth == 1 or np.unique(y).size == 1 or x.shape[0] <= self.min_samples:
            self.left_node, self.right_node = None, None
            return
            
        self.left_node = DecisionTree(self.max_depth-1,self.min_samples)
        self.right_node = DecisionTree(self.max_depth-1,self.min_samples)
        
        left_idx,right_idx,self.split_col,self.split_val,self.min_error = self.get_best_split(x,y)
        
        if self.split_col == -1: return

        self.left_node.train(x[ left_idx ],y[left_idx])
        self.right_node.train(x[ right_idx ],y[right_idx])

    def get_best_split(self,x,y):
        split_col = -1
        split_val = np.inf
        left_idx = np.array([])
        right_idx = np.array([])
        
        min_error = np.inf
        for col_idx in range(x.shape[1]):
            split_val_tmp, error = self.get_best_split_col(x.iloc[:,col_idx], y)
            if error < min_error:
                min_error = error
                split_val = split_val_tmp
                split_col = col_idx
                left_idx = x.iloc[:,col_idx] <= split_val_tmp
                right_idx = x.iloc[:,col_idx] > split_val_tmp
        
        return left_idx, right_idx, split_col, split_val, min_error
    
    def get_weighted_variance(self,sum_square,s,count):
        return math.sqrt(abs((sum_square/count) - (s/count)**2))*count

    def get_best_split_col(self,x,y):

        # left split  : all values <= split_val
        # right_split : all values  > split_val
        left_sum = 0
        left_sum_squares = 0

        right_sum = sum([i for i in y])
        right_sum_squares = sum([i*i for i in y])

        min_rmse = np.inf
        split_val = np.inf

        sorted_args = np.argsort(x)
        left_count = 0
        right_count = y.shape[0]

        for i in sorted_args:
            
            left_count += 1
            right_count -= 1

            # handle case when index is at last element
            if right_count == 0:
                break

            xi = x[ x.axes[0].tolist()[i] ] 
            yi = y[i]

            left_sum += yi
            left_sum_squares += yi*yi

            right_sum -= yi
            right_sum_squares -= yi*yi
            
            rmse_curr_split = ( self.get_weighted_variance(right_sum_squares,right_sum,right_count) +\
                                self.get_weighted_variance(left_sum_squares,left_sum,left_count) )
 
            if rmse_curr_split < min_rmse:
                min_rmse = rmse_curr_split
                split_val = xi

        return split_val, min_rmse
    
    def predict_single_row(self,x):
        
        # for leaf node, return mean value of the node
        if self.left_node == None and self.right_node == None:
            return self.mean_val

        # decide where to step next
        if x[self.split_col] <= self.split_val:
            return self.mean_val if self.left_node == None else self.left_node.predict_single_row(x)
        else:
            return self.mean_val if self.right_node == None else self.right_node.predict_single_row(x)
    
    def predict(self,df):
        return np.array( [ self.predict_single_row(df.iloc[i]) for i in range(df.shape[0]) ] )
