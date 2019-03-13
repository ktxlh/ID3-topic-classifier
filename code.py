# -*- coding: UTF-8 -*-
import logging
import os
import pickle
from random import shuffle
import gensim
import time
import random
import numpy as np
from matplotlib import pyplot as plt
TIME_STAMP  = time.strftime("%Y%m%d%H%M%S", time.localtime())  # to avoid duplicated file names deleting files
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
    filename=f'output/{TIME_STAMP}.txt', filemode='w')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is the Project Root
CATEGORIES = {'证券': 0, '教育': 1, '健康': 2, '娱乐': 3, '房产': 4, '科技': 5, '财经': 6, '军事': 7, '体育': 8}
WORD2VEC_MODEL_SIZE = 100
WORD2VEC_WINDOW = 10
WORD2VEC_MIN_COUNT = 3
FEATURE_SPLIT = 30
C45_Z = 1.96
MIN_N_FILES = 99999
SCORE = 'ENTROPY'


"""
To be initialized in each experiment
"""
raw_result = np.array([[0 for i in range(9)] for i in range(9)])
pruned_result = np.array([[0 for i in range(9)] for i in range(9)])
raw_accuracy = []
pruned_accuracy = []


def lognprint(text):
    logging.info(text)
    print(text)

def shuffle_files():
    """
    This is one-off, facilitating cross validation.
    Usage: shuffle_files()
    """
    num_files = 0
    for root, dirs, files in os.walk("new_weibo_13638/"):
        path = root.split(os.sep)
        if len(files)!=0:    
            files_copy = [f for f in files]     # TODO: is it necessary?
            shuffle(files_copy)
            category = CATEGORIES[os.path.basename(root)]
            for nf, f in enumerate(files_copy):
                num_files += 1
                old_name = os.path.join(ROOT_DIR, root, f)
                new_name = os.path.join(ROOT_DIR, root, f"{category}-{nf}-{TIME_STAMP}.txt")
                os.rename(old_name, new_name)
    lognprint(f"{num_files} files shuffled.")

def load_files(fold):
    train_docs = []
    test_docs = []

    def read_data_from_file(root, f):
        with open(os.path.join(ROOT_DIR, root, f), "r") as fin:
            try:
                return fin.readline().strip().split('\t')
            except Exception as e:
                logging.warning(f"Exception {e} raised by {f} in {root}")
                return None

    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk("new_weibo_13638/"):
        path = root.split(os.sep)
        if len(files)==0:
            continue

        # ten-fold cross validation
        train_cat = []
        test_cat = []
        for nf, f in enumerate(sorted(files)):

            if nf > MIN_N_FILES:
                break
            
            data = read_data_from_file(root, f)
            if data:
                if nf % 10 != fold:
                    train_cat.append(data)
                else:
                    test_cat.append(data)
        train_docs.append(train_cat)
        test_docs.append(test_cat)

    lognprint(f'Data from categories:{[len(tr)+len(te) for tr,te in zip(train_docs,test_docs)]}')
    lognprint(f'Training data: {sum([len(cat) for cat in train_docs])}')
    lognprint(f'Testing data: {sum([len(cat) for cat in test_docs])}')


    #return train_docs, test_docs
    train_texts = [doc for cat in train_docs for doc in cat]  # flattened docs
    train_labels = [ncat for ncat,cat in enumerate(train_docs) for doc in cat]
    test_texts = [doc for cat in test_docs for doc in cat]  # flattened docs
    test_labels = [ncat for ncat,cat in enumerate(test_docs) for doc in cat]
    return train_texts, train_labels, test_texts, test_labels


def train_w2v(documents, save_name):
    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        documents,
        size=WORD2VEC_MODEL_SIZE,
        window=WORD2VEC_WINDOW,
        min_count=WORD2VEC_MIN_COUNT,
        workers=10)
    model.train(documents, total_examples=len(documents), epochs=10)
    pickle.dump(model, open(save_name,'wb'))
    return model

def docs2vecs(model, docs):
    """
    docs: nested list of shape (# of documents, # of words)
    x_matrix: float64 numpy array of shape (# of documents, WORD2VEC_MODEL_SIZE)
    """
    def doc2vec(doc):
        """
        doc: list (# of words)
        doc_vec: (WORD2VEC_MODEL_SIZE,)
        """
        doc_vec = np.mean([model[word] for word in doc if word in model], 0)
        return doc_vec
        
    x_matrix = np.array([doc2vec(doc) for doc in docs])
    return x_matrix

class DTreeNode:
    NODE_COUNT = 0  # static
    def __init__(self,count):
        """
        IN::count: [int * 9] # of labels in each category under this subtree/node
        """
        self.count = count  # All;      [int * 9]
        self.label = None   # Leaf;     int
        self.left = None    # Internal; DTreeNode
        self.right = None   # Internal; DTreeNode
        self.feature = None # Internal; int
        self.value = None   # Internal; float
        DTreeNode.NODE_COUNT += 1
        if DTreeNode.NODE_COUNT % 500 == 0:
            lognprint(f"{DTreeNode.NODE_COUNT:5d} nodes")
    def is_leaf(self):
        return self.label != None
    def is_internal(self):
        return self.label == None
    def set_leaf(self,label):
        self.label = label
        #logging.debug(f'\tLeaf: {label}; \t\t{self.count}')
    def set_internal(self,feature, value):
        self.feature = feature
        self.value = value
        #logging.debug(f'Inte: {feature}; {value:.3f}; \t{self.count}')
    def replace_with_leaf(self):
        self.label = self.count.index(max(self.count))
        self.left = None
        self.Right = None
        self.feature = None
        self.value = None


def DTree(data, labels, features, depth): # examples, features
    """
    IN::data: np.array (# of some docs, # of all features)
    IN::labels: list (# of some docs)
    IN::features: a list of remaining features' indices
    IN::depth: the depth of the current node
    OUT::rtn: a DTreeNode
    """
    def get_candidate_splits():
        """
        IN::data: (# of some docs, # of all features <= WORD2VEC_MODEL_SIZE)
        OUT::splits: (# of all features, # of splits <= FEATURE_SPLIT)
        NOTE: Do not send data of only one doc here.
        """
        num_docs = data.shape[0]
        num_features = data.shape[1]  # for simplicity. used feature rows contain zeros

        # How many rules(splits)?
        if num_docs-1 < FEATURE_SPLIT:
            num_splits = num_docs-1
        else:
            num_splits = FEATURE_SPLIT

        splits = np.zeros(shape=(num_features,num_splits))
        for feature in features:
            vals = np.sort(data[:,feature])
            for i in range(num_splits):
                split_index = int(num_docs/(num_splits+1)*(i+1))-1 
                splits[feature,i] = (vals[split_index] + vals[split_index+1]) / 2
        return splits

    def neg_sum_entropy(feature, value):
        """
        IN::feature: a feature
        IN::value: a value of that feature
        OUT::rtn: ∑_i ∑_j |S_i^j| log(|S_i^j|/|S_i|)
                  where i iterates < or >=; j iterates categories
        """
        s_v_c = [[0 for j in range(9)] for i in range(2)]
        for doc, label in zip(data,labels):  # TODO: don't iterate data by 'for'
            child_node = int(doc[feature] >= value)
            s_v_c[child_node][label] += 1
        sum_s_i = [sum(s_v_c[0]), sum(s_v_c[1])]
        rtn = sum([s_i_j * np.log(s_i_j/sum_s_i[i]) for i,s_i in enumerate(s_v_c) for s_i_j in s_i if s_i_j != 0 and sum_s_i[i] > 0])
        return rtn

    def gini_score(feature, value):
        s_v_c = [[0 for j in range(9)] for i in range(2)]
        for doc, label in zip(data,labels):  # TODO: don't iterate data by 'for'
            child_node = int(doc[feature] >= value)
            s_v_c[child_node][label] += 1
        sum_s_i = [sum(s_v_c[0]), sum(s_v_c[1])]
        sum_s = sum(sum_s_i)
        portion_s_i = [sum_s_i[0]/sum_s, sum_s_i[1]/sum_s]
        pk0 = sum([(s_v_c[0][i]/sum_s_i[0])**2 for i in range(9) if sum_s_i[0] != 0])
        pk1 = sum([(s_v_c[1][i]/sum_s_i[1])**2 for i in range(9) if sum_s_i[1] != 0])
        rtn= -portion_s_i[0]*(1-pk0)-portion_s_i[1]*(1-pk1)
        return rtn

    def get_best_rule(score_function):
        """
        IN::score_function: a function accepting feature & value, returning the preference score (the higher, the better)
        OUT::feature: int
        OUT::value: float
        """
        candidates = get_candidate_splits()
        scores = np.array([[
            score_function(feature, value) 
            for value in candidates[feature]] # dim 1
            for feature in features])         # dim 0
        a = np.max(scores, axis=1)       # a = features best values
        b = np.argmax(a, axis=0)         # b = argmax feature index in scoress
        c = np.argmax(scores[b], axis=0) # c = argmax value index of the feature
        return features[b], candidates[features[b],c]   # argmax feature index, argmax value
            

    # Will return a DTreeNode anyway. Create one first.
    counted_labels = [labels.count(i) for i in range(9)]
    rtn = DTreeNode([counted_labels[i] for i in range(9)])
    
    #If all examples are in one category, 
    #return a leaf node with that category label.
    for label in range(9):
        if counted_labels[label] == len(labels):
            rtn.set_leaf(label)
            return rtn
    
    #Else if the set of features is empty, 
    #return a leaf node with the category label 
    #that is the most common in examples. 
    most_common_label = counted_labels.index(max(counted_labels))
    if len(features) == 0:
        rtn.set_leaf(most_common_label)
        return rtn
    
    #Else pick a feature F and create a node R for it
    # TODO: May the same feature be used more than once given that the splits differ?
    if SCORE == 'GINI':
        score_function = gini_score
    else:
        score_function = neg_sum_entropy
    feature, value = get_best_rule(score_function)
    
    #For each possible value vi of F: (NOTE: we have only 2: < & >=)
    #Let examples_i be the subset of examples that have value v_i for F
    #Add an out-going edge E to node R labeled with the value v_i.
    rtn.set_internal(feature, value)
    
    #If examples_i is empty
    #then attach a leaf node to edge E labeled with the category that
    #is the most common in examples.
    
    #else call DTree(examplesi , features – {F}) and attach the resulting
    #tree as the subtree under edge E.    
    
    l_data = np.zeros(shape=(0,data.shape[1]))
    r_data = np.zeros(shape=(0,data.shape[1]))
    l_labels = []
    r_labels = []
    for n_doc in range(len(data)):  # TODO: don't iterate data by 'for'
        doc = data[n_doc]
        if doc[feature] < value:
            l_data = np.concatenate((l_data,[doc]))
            l_labels.append(labels[n_doc])
        else:
            r_data = np.concatenate((r_data,[doc]))
            r_labels.append(labels[n_doc])
        
    sub_features = [f for f in features if f != feature]

    if len(l_labels) == 0:
        rtn.left = DTreeNode([0 for i in range(9)])
        rtn.left.set_leaf(most_common_label)
    else:
        rtn.left = DTree(l_data, l_labels, sub_features, depth+1)
        
    if len(r_labels) == 0:
        rtn.right = DTreeNode([0 for i in range(9)])
        rtn.right.set_leaf(most_common_label)
    else:
        rtn.right = DTree(r_data, r_labels, sub_features, depth+1)

    #Return the subtree rooted at R.
    return rtn
    
def PruneDTree(node,depth):
    """
    IN::node: root node of DTree
    """
    def get_error(node):
        Z = C45_Z
        N = sum(node.count)
        if N == 0:
            return 0.0  #float('inf')
        f = 1 - max(node.count)/N
        e = (   f +\
                (Z**2)/(2*N)+\
                Z * np.sqrt( 
                    f/N - (f**2)/N + (Z**2)/(4*(N**2))))\
            /(1 + (Z**2)/N)
        return e
        
    # if leaf, return
    if node.is_leaf():
        return node
    
    ## recursion
    node.left = PruneDTree(node.left,depth+1)
    node.right = PruneDTree(node.right,depth+1)
    
    # elif l is internal, check if l should be leaf; update if so
    if node.left.is_internal():
        lc = sum(node.left.left.count)
        rc = sum(node.left.right.count)
        lp = lc/(lc+rc)
        rp = rc/(lc+rc)
        original = get_error(node.left.left)*lp + get_error(node.left.right)*rp
        replaced = get_error(node.left)
        if replaced < original:
            node.left.replace_with_leaf()
    if node.right.is_internal():
        lc = sum(node.right.left.count)
        rc = sum(node.right.right.count)
        lp = lc/(lc+rc)
        rp = rc/(lc+rc)
        original = get_error(node.right.left)*lp + get_error(node.right.right)*rp
        replaced = get_error(node.right)
        if replaced < original:
            node.right.replace_with_leaf()
    
    #print('-'*depth)
    return node
    

def Predict(node, doc):
    """
    IN::node: root node of DTree
    IN::doc: np.array (# of all features)
    """
    if not node:
        logging.error('Empty node accessed while predicting')
        return None
    if node.is_leaf():
        return node.label
    if doc[node.feature] < node.value:
        return Predict(node.left, doc)
    else:
        return Predict(node.right, doc)

def lognprint_parameters():
    lognprint(f"""
        WORD2VEC_MODEL_SIZE = {WORD2VEC_MODEL_SIZE}
        WORD2VEC_WINDOW = {WORD2VEC_WINDOW}
        WORD2VEC_MIN_COUNT = {WORD2VEC_MIN_COUNT}
        FEATURE_SPLIT = {FEATURE_SPLIT}
        C45_Z = {C45_Z}
        MIN_N_FILES = {MIN_N_FILES}
        SCORE = {SCORE}
        """)

## plot the result (热度图)
def plot_result(title, result):
    categories = ['0证券', '1教育', '2健康', '3娱乐', '4房产', '5科技', '6财经', '7军事', '8体育']
    plt.imshow(result)
    plt.xlabel('Real')
    plt.xticks(np.arange(9),categories)
    plt.ylabel('Predicted')
    plt.yticks(np.arange(9),categories)
    for i in range(9):
        for j in range(9):
            text = plt.text(j, i, result[i,j],
                ha='center', va='center', color='w')
    plt.title(title)
    plt.colorbar()
    plt.savefig(f'plot/{title}-{TIME_STAMP}.png')
    plt.clf()
    lognprint(f'{title}-Result plotted:\n{result}')
    lognprint(f'{title}-Result accuracy:{result.trace()/result.sum():.3f}')

def one_fold(fold, train_new_model):
    #Never call shuffle_files() again!!
    
    train_texts, train_y, test_texts, test_y = load_files(fold)
    
    # compute attr
    ## train word2vec
    model_name = f'model/model-{fold}.w2v'
    if train_new_model:
        model = train_w2v(train_texts, model_name)
    else:
        model = pickle.load(open(model_name,'rb'))
    
    ### tag words & docs
    train_x = docs2vecs(model, train_texts)
    test_x = docs2vecs(model, test_texts) 
    lognprint('Docs tagged')

    # decision tree
    ## impurity function
    ### -[*] Entropy
    ### -[ ] Gini index
    ### -[ ] Misclassification error

    lognprint('Growing tree')
    ## growing
    ### if termination condition not reached
    ### calculate each candidate's info gain
    ### choose the best one and split
    ### recursion
    tree = DTree(train_x, train_y, [i for i in range(train_x.shape[1])], 1)
    # NOTE: To pickle a self-defined type (e.g. DTreeNode),
    # see https://stackoverflow.com/questions/27351980/how-to-add-a-custom-type-to-dills-pickleable-types

    
    
    # Testing
    ## accuracy
    ## f-measure

    def get_result(tree, x, y):
        """
        OUT::result: np array (# of cats, # of cats)
        """
        result = np.array([[0 for i in range(9)] for i in range(9)])
        for i in range(len(y)):
            cat_ans = y[i]
            cat_pred = Predict(tree, x[i])
            result[cat_ans,cat_pred] += 1
        accuracy = result.trace()/result.sum()
        return result, accuracy     # TODO: f1_score?
    
    def evaluate(tree, result, accuracy):
        train_result, train_accuracy = get_result(tree, train_x, train_y)
        test_result, test_accuracy = get_result(tree, test_x, test_y)
        
        result += test_result
        accuracy.append(test_accuracy)
        
        logging.info(train_result)
        lognprint(f"Accuracy on training: {train_accuracy:.3f}")
        logging.info(test_result)
        lognprint(f"Accuracy on testing: {test_accuracy:.3f}")

    # raw tree
    lognprint("Raw tree:")
    evaluate(tree, raw_result, raw_accuracy)

    ## post-pruning
    ### Subtree replacement
    lognprint('Pruning tree')
    PruneDTree(tree,1)

    # pruned tree
    lognprint("Pruned tree:")
    evaluate(tree, pruned_result, pruned_accuracy)

def experiment():
    raw_result = np.array([[0 for i in range(9)] for i in range(9)])
    pruned_result = np.array([[0 for i in range(9)] for i in range(9)])
    raw_accuracy = []
    pruned_accuracy = []
    
    lognprint_parameters()
    for fold in range(10):
        lognprint("-"*100+f" Fold {fold} "+"-"*100)
        one_fold(fold=fold, train_new_model=False)
    plot_result(f'{SCORE} z={C45_Z} Raw Tree', raw_result)
    plot_result(f'{SCORE} z={C45_Z} Pruned Tree', pruned_result)



if __name__ == "__main__":
   
    #ZS = [0.67, 1.00, 1.64, 1.96]
    ########################################################################
    SCORE = 'GINI'
    C45_Z = 1.00
    experiment()

    ########################################################################
    SCORE = 'GINI'
    C45_Z = 1.96
    experiment()

    ########################################################################
    SCORE = 'ENTROPY'
    C45_Z = 1.96
    experiment()


    