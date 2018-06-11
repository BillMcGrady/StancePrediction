import codecs
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from deepwalk import graph
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import KeyedVectors
import networkx as nx
import node2vec
from utils import *
import six.moves.cPickle as pickle

class DataLoader(object):
    
    def __init__(self, path, random_seed, fold):
        np.random.seed(random_seed)
        self.path = path
        self.linkfile = path + 'allPostLinkMap.pickle'
        # self.edgelistfile = path + 'edgelist.txt'
        self.labelfile = path + 'allPostLabelMap.pickle'
        self.authorfile = path + 'allPostAuthorMap.pickle'
        self.authorattrifile = path + 'allAuthorAttrisProc.pickle'
        self.authorlinkfile = path + 'allAuthorLinks.pickle'
        self.textfile = path + 'allUserTextSkip.pickle2'
        self.foldfile = path + 'allFolds.pickle'
        self.threadfile = path + 'allThreadPost.pickle'
        self.embfile = path + 'node.emb'
        self.fold = fold
        self.nodes_infor = []
        self.node_map = {} 
        with open(self.textfile, 'rb') as fin:
            allTextEmbed = pickle.load(fin, encoding='latin1')
            self.allTextMap = pickle.load(fin, encoding='latin1')
            fin.close()
        self.node_count = len(self.allTextMap)
        for i in range(self.node_count):
            self.add_node(i)
        self.read_label()
        self.read_text()
        self.read_link()
        self.label_count = len(self.label_map)
        # print('label count:', self.label_count)
        self.construct_data()
        
    def add_node(self, name):
        if (name not in self.node_map):
            self.node_map[name] = len(self.node_map) # map the node
            node_infor_map = {}
            node_infor_map['node_name'] = name
            node_infor_map['node_connected'] = []
            node_infor_map['node_neighbors'] = []
            node_infor_map['node_labels'] = []
            node_infor_map['node_author'] = []
            node_infor_map['node_author_attri'] = []
            node_infor_map['node_author_allies'] = []
            node_infor_map['node_author_enemies'] = []
            self.nodes_infor.append(node_infor_map)
        return self.node_map[name]
            
    def read_text(self):
        '''read text '''

        with open(self.textfile, 'rb') as fin:
            allTextEmbed = pickle.load(fin, encoding='latin1')
            allTextMap = pickle.load(fin, encoding='latin1')
            fin.close()
        with open(self.authorfile, 'rb') as fin:
            allPostAuthor = pickle.load(fin)
            fin.close()
        # with open(self.authorattrifile, 'rb') as fin:
        #     allAuthorAttris = pickle.load(fin)
        #     author_attri_vals = pickle.load(fin)
        #     fin.close()    
        
        # for attri in author_attri_vals:
        #     self.label_map[attri] = len(self.label_map)
        author_map = {}
        test_author_set = set([])
        for post in allPostAuthor.keys():
            if post not in allTextMap:
                continue
            post_id = allTextMap[post]
            author_raw = allPostAuthor[post]
            if (self.domain in post):
                test_author_set.add(author_raw)
            if (post_id in self.training_nodes):
                author = author_raw + '_train'
            else:
                author = author_raw + '_test'
            if (author not in author_map):
                author_map[author] = len(author_map)
            self.nodes_infor[post_id]['node_author'].append(author_map[author])
            # whether use author attributes or not
            # author_attris = allAuthorAttris[author_raw]
            # self.nodes_infor[post_id]['node_author_attri'] = [self.label_map[attri] for attri in author_attris]
            
        self.author_count = len(author_map)

        '''
        with open(self.authorlinkfile, 'rb') as fin:
            allAuthorLinks = pickle.load(fin)
            fin.close()
        for post in allPostAuthor.keys():
            if post not in allTextMap:
                continue
            post_id = allTextMap[post]
            author_raw = allPostAuthor[post]
            if (post_id in self.training_nodes):
                author = author_raw + '_train'
            else:
                author = author_raw + '_test'
            author_allies = allAuthorLinks[author_raw]['allies']
            for ally in author_allies:
                if (ally + '_train' in author_map):
                    self.nodes_infor[post_id]['node_author_allies'].append(author_map[ally + '_train'])
                if (ally + '_test' in author_map):
                    self.nodes_infor[post_id]['node_author_allies'].append(author_map[ally + '_test'])
        '''
        embed_rng = np.random.RandomState(1)
        self.embed_rng = embed_rng
        self.text2vec = allTextEmbed
        self.embedding_size = 300
        # print('node_count:', self.node_count)
        # print('author_count:', self.author_count)
        print('topic_author_count', len(test_author_set))

    def read_label(self):
        '''read labels '''
        self.label_list_4pred = [6, 7]
        self.test_domain = 'gun'
        self.domain = 'gun'

        with open(self.labelfile, 'rb') as fin:
            allPostLabel = pickle.load(fin)
            fin.close()
        
        self.label_map = {}
        self.labeled_nodes = []
        self.all_nodes = []
        # self.label_for_pred = {}
        self.label_node_list = {}
        for i in range(8):
            self.label_node_list[i] = []
            self.label_map[i] = i
        sorted_keys = sorted(allPostLabel.keys())
        for post in sorted_keys:
            if (post not in self.allTextMap):
                continue
            post_id = self.allTextMap[post]
            self.nodes_infor[post_id]['node_labels'].append(allPostLabel[post])
            self.all_nodes.append(post_id)
            
            # stance label
            for item in [allPostLabel[post]]:
                if (item in self.label_list_4pred):
                    self.labeled_nodes.append(post_id)
                    # self.label_for_pred[post_id] = item
                self.label_node_list[item].append(post_id)

        # split data according to folds
        n = len(self.labeled_nodes)
        print('topic_post_count:', n)
        # split_rng = np.random.RandomState(1)
        # permutated_divide = [self.labeled_nodes[i] for i in split_rng.permutation(n)]
        # training_percent = 80
        # valid_percent = 10
        # self.training_nodes = permutated_divide[:training_percent*n//100]
        # self.test_nodes = permutated_divide[(training_percent+valid_percent)*n//100:]
        self.valid_nodes = [] # permutated_divide[training_percent*n//100:] #(training_percent+valid_percent)*n//100]
        with open(self.foldfile, 'rb') as fin:
            allFolds = pickle.load(fin)
            fin.close()

        self.training_nodes = []
        self.test_nodes = []
        total = 0
        for num_fold in allFolds[self.test_domain].keys():
            total += len(allFolds[self.test_domain][num_fold])
            for post in allFolds[self.test_domain][num_fold]:
                post_id = self.allTextMap[post]
                if (num_fold == self.fold):
                    self.test_nodes.append(post_id)
                    for label in self.label_list_4pred:
                        if (post_id in self.label_node_list[label]):
                            self.label_node_list[label].remove(post_id)
                else:
                    self.training_nodes.append(post_id)
        print('train/test/total size:', len(self.training_nodes), len(self.test_nodes), total)
        print('pos/neg size: ', len(self.label_node_list[self.label_list_4pred[0]]), len(self.label_node_list[self.label_list_4pred[1]]))


    def read_link(self): 
        '''read links '''

        with open(self.linkfile, 'rb') as fin:
            allPostLink = pickle.load(fin)
            fin.close()

        trainX, trainy, testX, testy = [], [], [], []
        for post in allPostLink:
            if (post not in self.allTextMap):
                continue
            post_id = self.allTextMap[post]
            neighbors = allPostLink[post]
            for neigh, rebuttal in neighbors:
                if (neigh not in self.allTextMap):
                    continue
                neigh_id = self.allTextMap[neigh]
                self.nodes_infor[post_id]['node_connected'].append((neigh_id, rebuttal))

                this_x = np.stack((self.text2vec[post_id], self.text2vec[neigh_id])).reshape(-1) 
                if (post_id in self.test_nodes):
                    testX.append(this_x)
                    testy.append(rebuttal)
                else:
                    trainX.append(this_x)
                    trainy.append(rebuttal)

        # trainX = np.asarray(trainX)
        # trainy = np.asarray(trainy)
        # testX = np.asarray(testX)
        # testy = np.asarray(testy)
        # print(trainX.shape, trainy.shape, testX.shape, testy.shape)
        # from sklearn.svm import SVC, LinearSVC
        # from sklearn.metrics import f1_score
        # from sklearn.metrics import confusion_matrix
        # clf = LinearSVC()
        # clf.fit(trainX, trainy) 
        # pred_all = clf.predict(testX)
        # gold_all = testy
        # print(sum(gold_all == 1), sum(gold_all == 1)/len(gold_all), sum(trainy == 1), sum(trainy == 1)/len(trainy))
        # print(f1_score(gold_all, pred_all, pos_label=None, average='micro'))
        # print(f1_score(gold_all, pred_all, pos_label=None, average='macro'))
        # print(confusion_matrix(gold_all, pred_all))

        # pred_all = clf.decision_function(testX)
        # print(pred_all)
        # exit()

        # for post in self.allTextMap:
        #     print(post, self.nodes_infor[self.allTextMap[post]]['node_connected'])
        
    def read_embeddings(self):
        '''read trained embeddings for nodes'''
        self.node_emb = {}
        f = open(self.embfile)
        f.readline()

        for line in f.readlines():
            vals = line.strip().split()
            self.node_emb[self.node_map[vals[0]]] = [float(val) for val in vals[1:]]

        f.close()

    def process_features(self, node_emb, nodes):
        X, y = [], []
        for node_id in nodes:
            X.append(node_emb[node_id])
            y.append(self.nodes_infor[node_id]['node_labels'][0])
        X, y = np.asarray(X), np.asarray(y)

        print(X.shape, y.shape)
        return X, y

    def eval(self, node_emb, label_emb, text_emb, node_fc1):
        fcw, fcb = node_fc1
        total = len(self.test_nodes)
        correct = 0
        correct_ptext = 0
        correct_all = 0
        # fout = open('result/eval_result_noself.txt', 'w')
        for node_id in self.test_nodes:
            label = self.nodes_infor[node_id]['node_labels'][0]
            text = self.nodes_infor[node_id]['node_author']
            neighbor = self.nodes_infor[node_id]['node_neighbors']
            pred_val = -100000
            pred_id = 0
            pred_val_ptext = -100000
            pred_val_ptextsum = -100000
            pred_id_ptext = 0
            pred_val_all = -100000
            pred_val_allsum = -100000
            pred_id_all = 0

            neighbor_sim = 0
            neighbor_set = set(neighbor)
            for j in neighbor_set:
                neighbor_sim += np.dot(node_emb[j], node_emb[node_id])
            if (len(neighbor_set) > 0):
                neighbor_sim /= len(neighbor_set)
            this_node_emb = np.matmul(fcw, node_emb[node_id]) + fcb
            for i in [0, 1]:
                cur = np.dot(this_node_emb, label_emb[i])
                cur_text = 0
                for j in text:
                    cur_text += np.dot(text_emb[j], label_emb[i])
                if (len(text) > 0):
                    cur_text /= len(text)
                cur_neigh = 0
                neighbor = np.random.choice(self.label_node_list[i], 20)
                for j in neighbor:
                    neigh_emb = np.matmul(fcw, node_emb[j]) + fcb
                    cur_neigh += np.dot(this_node_emb, neigh_emb)

                if (len(neighbor) > 0):
                    cur_neigh /= len(neighbor)
                # print(i, cur, cur_neigh)
                if (cur > pred_val):
                    pred_val = cur
                    pred_val_text = cur_text
                    pred_val_neigh = cur_neigh
                    pred_id = i
                if (cur + cur_text > pred_val_ptext):
                    pred_val2 = cur
                    pred_val_text2 = cur_text
                    pred_val_neigh2 = cur_neigh
                    pred_val_ptext = cur + cur_text
                    pred_id_ptext = i
                # if (cur + cur_text * len(text) > pred_val_ptextsum):
                #     pred_val3 = cur
                #     pred_val_text3 = cur_text * len(text)
                #     pred_val_neigh3 = cur_neigh * len(neighbor)
                #     pred_val_ptextsum = cur + cur_text * len(text)
                #     pred_id_ptextsum = i
                if (cur + cur_neigh > pred_val_all):
                    pred_val4 = cur
                    pred_val_text4 = cur_text
                    pred_val_neigh4 = cur_neigh
                    pred_val_all = cur + cur_neigh
                    pred_id_all = i
                # if (cur + cur_text * len(text) + cur_neigh * len(neighbor) > pred_val_allsum):
                #     pred_val5 = cur
                #     pred_val_text5 = cur_text * len(text)
                #     pred_val_neigh5 = cur_neigh * len(neighbor)
                #     pred_val_allsum = cur + cur_text * len(text) + cur_neigh * len(neighbor)
                #     pred_id_allsum = i
                if (label == i):
                    gold_val = cur
                    gold_val_text = cur_text
                    gold_val_neigh = cur_neigh
            if (pred_id == label):
                correct += 1
            if (pred_id_ptext == label):
                correct_ptext += 1
            if (pred_id_all == label):
                correct_all += 1
            # fout.write('%d %d %d %d %f %f %f %f %s %f %f %f %s %f %f %f %s %f %f %f %s %f %f %f %s %s %s %s\n' % (node_id, len(text), len(neighbor), 
            #     len(neighbor_set), neighbor_sim,
            #     pred_val, pred_val_text, pred_val_neigh, pred_id == label, 
            #     pred_val2, pred_val_text2, pred_val_neigh2, pred_id_ptext == label,
            #     pred_val3, pred_val_text3, pred_val_neigh3, pred_id_ptextsum == label,
            #     pred_val4, pred_val_text4, pred_val_neigh4, pred_id_all == label,
            #     pred_val5, pred_val_text5, pred_val_neigh5, pred_id_allsum == label,
            #     gold_val, gold_val_text, gold_val_neigh))
        print(correct, correct_ptext, correct_all, total)
        # fout.close()

        return correct, total


    def construct_data(self):
        shuffle_rng = np.random.RandomState(2017)
        self.shuffle_rng = shuffle_rng
        
        n = len(self.labeled_nodes)
        self.node_list = np.asarray([self.labeled_nodes[i] for i in shuffle_rng.permutation(n)])
        # n = len(self.all_nodes)
        # self.node_list = np.asarray([self.all_nodes[i] for i in shuffle_rng.permutation(n)])
        self.data_size = len(self.node_list)
        # print(self.node_list[:20])

        with open(self.threadfile, 'rb') as fin:
            allThreadPost = pickle.load(fin)
            fin.close()
        allThreadPostIndex = {}
        self.thread_list = []
        for thread_id in allThreadPost.keys():
            posts = allThreadPost[thread_id]
            post_ids = [self.allTextMap[post] for post in posts if post in self.allTextMap]
            allThreadPostIndex[thread_id] = post_ids
            if (self.test_domain == 'abortion'):
                if (thread_id < 171):
                    self.thread_list.append(post_ids)
            elif (self.test_domain == 'evolution'):        
                if (thread_id >= 171 and thread_id < 386):
                    self.thread_list.append(post_ids)
            elif (self.test_domain == 'gay'):
                if (thread_id >= 386 and thread_id < 578):
                    self.thread_list.append(post_ids)
            elif (self.test_domain == 'gun'):
                if (thread_id >= 578):    
                    self.thread_list.append(post_ids)
        self.thread_list = np.asarray(self.thread_list)
        self.thread_size = len(self.thread_list)
        # print(self.thread_size, self.thread_list)
            

