# -*- coding: UTF-8 -*-
import logging
import os
import pickle
from random import shuffle
import gensim
import time
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is the Project Root
CATEGORIES = {'证券': 0, '教育': 1, '健康': 2, '娱乐': 3, '房产': 4, '科技': 5, '财经': 6, '军事': 7, '体育': 8}
MODEL_SIZE = 100    # TODO: tuning
FEATURE_SPLIT = 2   # TODO: tuning

def shuffle_files():
    """
    This is one-off, facilitating cross validation.
    Usage: shuffle_files()
    """
    num_files = 0
    time_string  = time.strftime("%Y%m%d%H%M%S", time.localtime())  # to avoid duplicated file names deleting files
    for root, dirs, files in os.walk("new_weibo_13638/"):
        path = root.split(os.sep)
        if len(files)!=0:    
            files_copy = [f for f in files]
            shuffle(files_copy)
            category = CATEGORIES[os.path.basename(root)]
            for nf, f in enumerate(files_copy):
                num_files += 1
                old_name = os.path.join(ROOT_DIR, root, f)
                new_name = os.path.join(ROOT_DIR, root, f"{category}-{nf}-{time_string}.txt")
                os.rename(old_name, new_name)
    logging.info(f"{num_files} files shuffled.")

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
        #print((len(path) - 1) * '---', os.path.basename(root), len(files))

        # ten-fold cross validation
        train_cat = []
        test_cat = []
        for nf, f in enumerate(sorted(files)):
            data = read_data_from_file(root, f)
            if data:
                if nf % 10 != fold:
                    train_cat.append(data)
                else:
                    test_cat.append(data)
        train_docs.append(train_cat)
        test_docs.append(test_cat)


    logging.info(f'Training data: {sum([len(cat) for cat in train_docs])} from {len(train_docs)} categories')
    logging.info(f'Testing data: {sum([len(cat) for cat in test_docs])} from {len(test_docs)} categories')


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
        size=MODEL_SIZE,
        window=10,
        min_count=2,
        workers=10)
    model.train(documents, total_examples=len(documents), epochs=10)    # TODO: tuning later (when finalizing)
    pickle.dump(model, open(save_name,'wb'))
    return model

def docs2vecs(model, docs):
    """
    docs: nested list of shape (# of documents, # of words)
    x_matrix: float64 numpy array of shape (# of documents, MODEL_SIZE)
    """
    def doc2vec(doc):
        """
        doc: list (# of words)
        x_array: (MODEL_SIZE,)
        """
        word_vecs = np.array([model[word] for word in doc if word in model])
        #unk_count = sum([1 for word in doc if word not in model]) # TODO: handle <unk>
        doc_vec = np.mean(word_vecs, 0)
        return doc_vec
        
    #print(doc2vec(docs[0]).shape)
    x_matrix = np.array([doc2vec(doc) for doc in docs])
    return x_matrix

class DTreeNode:
    def __init__(self):
        #self.parent = parent_node
        self.left = None
        self.right = None
        self.label = -1
        """
        if parent_node:
            if parent_left:
                parent_node.left = self
            else:
                parent_node.right = self
        """
    def split(feature, value):
        self.feature = feature
        self.value = value

def DTree(data, labels, features): # examples, features
    """
    data: np.array (# of some docs, # of all features)
    labels: list (# of some docs)
    features: a list of remaining features' indices
    rtn: a DTreeNode
    """
    def get_candidate_splits():
        """
        IN::data: (# of some docs, # of all features <= MODEL_SIZE)
        OUT::splits: (# of all features, # of splits <= FEATURE_SPLIT)
        NOTE: Do not send data of only one doc here.
        """
        num_docs = data.shape[0]
        num_features = data.shape[1]  # for simplicity. used feature rows contain zeros

        # How many rules(splits)?
        if num_docs-1 <= FEATURE_SPLIT:
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
        s_v_c = np.zeros((2,2), dtype=np.int8)
        for doc, label in zip(data,labels):
            child_node = int(doc[feature] >= value)
            s_v_c[child_node,label] += 1
        sum_s_i = s_v_c.sum(axis=1)
        rtn = sum([s_i_j * np.log(s_i_j/sum_s_i[i]) for i,s_i in enumerate(s_v_c) for s_i_j in s_i if s_i_j != 0 and sum_s_i[i] > 0])
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
        a = np.max(scores, 1)    # a = features_best_values
        b = np.argmax(a, 0)      # b = argmax_feature_index_in_scores
        return features[b], a[b] # argmax_feature, max_value
        
    # Will return a DTreeNode anyway. Create one first.
    rtn = DTreeNode()
    
    #If all examples are in one category, 
    #return a leaf node with that category label.
    counted_label = [labels.count(i) for i in range(9)]
    for label in range(9):
        if counted_label[label] == len(labels):
            rtn.label = label
            return rtn
    
    #Else if the set of features is empty, 
    #return a leaf node with the category label 
    #that is the most common in examples. 
    most_common_label = counted_label.index(max(counted_label))
    if len(features) == 0:
        rtn.label = most_common_label
        return rtn
    
    #Else pick a feature F and create a node R for it
    # TODO: The same feature can be used more than once given that the splits differ
    feature, value = get_best_rule(neg_sum_entropy)

    #For each possible value vi of F: (NOTE: we have only 2: < & >=)
    #Let examples_i be the subset of examples that have value v_i for F
    #Add an out-going edge E to node R labeled with the value v_i.
    rtn.feature = feature
    rtn.value = value    
    
    #If examples_i is empty
    #then attach a leaf node to edge E labeled with the category that
    #is the most common in examples.
    
    #else call DTree(examplesi , features – {F}) and attach the resulting
    #tree as the subtree under edge E.    
    
    lr_data = [np.zeros(shape=(0,data.shape[1])) for i in range(2)]
    lr_labels = [[] for i in range(2)]
    for n_doc in range(len(data)):
        doc = data[n_doc]
        lr = int(doc[feature] >= value)
        lr_data[lr] = np.concatenate((lr_data[lr],[doc]))
        lr_labels[lr].append(labels[n_doc])    

    features.remove(feature)    

    if len(lr_labels[0]) == 0:
        rtn.left = DTreeNode()
        rtn.left.label = most_common_label
    else:
        rtn.left = DTree(lr_data[0], lr_labels[0], features)
        
    if len(lr_labels[1]) == 0:
        rtn.right = DTreeNode()
        rtn.right.label = most_common_label
    else:
        rtn.right = DTree(lr_data[1], lr_labels[1], features)

    #Return the subtree rooted at R.
    return rtn
    



if __name__ == "__main__":
    #Never call shuffle_files() again!!

    fold = 3    # TODO: fold - from 0 to 9
    train_texts, train_y, test_texts, test_y = load_files(fold)
    
    # compute attr
    ## train word2vec
    model_name = f'model/model-{fold}.w2v'
    #model = train_w2v(train_texts, model_name)
    model = pickle.load(open(model_name,'rb'))
    
    ### tag words & docs
    train_x = docs2vecs(model, train_texts)
    test_x = docs2vecs(model, test_texts)
    
    
    # decision tree
    ## impurity function
    ### -[*] Entropy
    ### -[ ] Gini index
    ### -[ ] Misclassification error

    ## growing
    ### if termination condition not reached
    ### calculate each candidate's info gain
    ### choose the best one and split
    ### recursion
    tree = DTree(train_x, train_y, )
    
    ## pruning
    
    # prediction
    ## accuracy
    ## f-measure
    
    
        
