import collections
import glob
import math
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from ILPInferencer4Forum import *
import matplotlib.pyplot as plt
from seaborn import distplot
from utils import *

class DynamicNet(torch.nn.Module):
    def __init__(self, data, num_sampled, train_method='local'):
        """
        In the constructor we construct instances that we will use
        in the forward pass.
        """
        super(DynamicNet, self).__init__()
        torch.cuda.manual_seed(1)
        torch.manual_seed(1)
        self.node_count = data.node_count
        self.label_count = data.label_count
        self.author_count = data.author_count
        self.embedding_size = data.embedding_size
        self.text_embedding_size = 4800
        self.labeled_nodes_set_train = set(data.training_nodes)
        self.label_list_4pred = get_torch_var(data.label_list_4pred)
        self.num_hidden_layers = 0
        self.num_sampled = num_sampled # Number of negative examples to sample.
        # Embeddings
        embed_rng = data.embed_rng
        scale=1.0/math.sqrt(self.embedding_size)
        self.node_embeddings = nn.Embedding(self.node_count, self.text_embedding_size).cuda()
        self.node_embeddings.weight.data.copy_(torch.from_numpy(data.text2vec))
        self.node_embeddings.weight.requires_grad=False

        # self.label_count = 8
        self.label_embeddings = nn.Embedding(self.label_count, self.embedding_size).cuda()
        label_embed_values = embed_rng.normal(scale=scale, size=(self.label_count, self.embedding_size))
        self.label_embeddings.weight.data.copy_(torch.from_numpy(label_embed_values))
        
        self.author_embeddings = nn.Embedding(self.author_count, self.embedding_size).cuda()
        author_embed_values = embed_rng.normal(scale=scale, size=(self.author_count, self.embedding_size))
        self.author_embeddings.weight.data.copy_(torch.from_numpy(author_embed_values))
        
        # self.word_embeddings.weight = nn.Parameter(torch.from_numpy(data.word2vec).cuda())
        self.CrossEntropyLoss = nn.CrossEntropyLoss(size_average=False)
        self.softmax = nn.Softmax()
        self.node_fc1 = nn.Linear(self.text_embedding_size, self.embedding_size).cuda() 
        # node_fc1_weight = embed_rng.normal(scale=scale, size=(self.text_embedding_size, self.embedding_size))
        # node_fc1_bias = embed_rng.normal(scale=scale, size=(self.embedding_size))
        # self.node_fc1.weight.data.copy_(torch.from_numpy(node_fc1_weight))
        # self.node_fc1.bias.data.copy_(torch.from_numpy(node_fc1_bias))
        self.tanh = nn.Tanh()
        self.node_fc2 = nn.Linear(self.embedding_size, self.embedding_size).cuda() 
        
        self.zero = Variable(torch.from_numpy(np.float32([0])).cuda(), requires_grad=False)
        self.one = Variable(torch.from_numpy(np.float32([1])).cuda(), requires_grad=False)
        self.dropout = nn.Dropout(p=0.7)
        self.train_method = train_method
        if (train_method == 'local'):
            self.aspects = ['author', 'author_allies', 'neighbor_pos', 'neighbor_neg', 'label', 'author_label']
        else:
            self.aspects = ['author', 'author_allies', 'neighbor_pos', 'neighbor_neg', 'author_label']
        self.coeff_sim = nn.Parameter(torch.from_numpy(np.float32([2])).cuda(), requires_grad=True)
        self.coeff_const = nn.Parameter(torch.from_numpy(np.float32([2])).cuda(), requires_grad=True)
        
    def apply_text_embed(self, x, train=True):
        if train:
            # x1 = self.dropout(self.tanh(self.node_fc1(x)))
            # x2 = self.dropout(self.node_fc2(x1))
            x2 = self.dropout(self.tanh(self.node_fc1(x)))
        else:
            # x1 = self.tanh(self.node_fc1(x))
            # x2 = self.node_fc2(x1)
            x2 = self.tanh(self.node_fc1(x))
        return x2

    def forward(self, batch_input_index, batch_gold_index, batch_negs_index, batch_sizes, neighbor_labels, infer_data):
        """
        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        embedding_size = self.embedding_size
        batch_target_index = {}
        batch_input = {}
        batch_gold = {}
        batch_target = {}
        input_embed = {}
        target_embed = {}
        sim_score = {}
        loss_all = {}

        loss = 0
        for aspect in self.aspects:
            if (len(batch_gold_index[aspect]) == 0):
                continue
            # Target Size/embeddings 
            if (aspect.startswith('neighbor')):
                target_size = self.node_count
                target_embeddings = self.node_embeddings
                input_type, target_type = 'node', 'node'
                num_neg_sampled = self.num_sampled
            elif (aspect == 'label'):
                target_size = self.label_count
                target_embeddings = self.label_embeddings
                input_type, target_type = 'node', 'label'
                num_neg_sampled = 1
            elif (aspect == 'author'):
                target_size = self.author_count
                target_embeddings = self.author_embeddings    
                input_type, target_type = 'node', 'author'
                num_neg_sampled = self.num_sampled
            elif (aspect == 'author_label'):
                target_size = self.label_count
                target_embeddings = self.label_embeddings
                input_type, target_type = 'author', 'label'
                num_neg_sampled = 1
            elif (aspect == 'author_allies'):
                target_size = self.author_count
                target_embeddings = self.author_embeddings
                input_type, target_type = 'author', 'author'
                num_neg_sampled = self.num_sampled
                
            # Input data.
            batch_target_index[aspect] = torch.cat((batch_gold_index[aspect], batch_negs_index[aspect]), 1)

            # Model.
            # Look up embeddings for inputs.
            if (aspect in ['author_label', 'author_allies']):
                batch_input[aspect] = self.author_embeddings(batch_input_index[aspect]).view((-1, self.embedding_size))
            else:
                batch_input[aspect] = self.node_embeddings(batch_input_index[aspect]).view((-1, self.text_embedding_size))
            if (aspect == 'neighbor_neg'):
                batch_target[aspect] = -target_embeddings(batch_target_index[aspect]).view((-1, num_neg_sampled+1, self.text_embedding_size)) 
            elif (aspect == 'neighbor_pos'):
                batch_target[aspect] = target_embeddings(batch_target_index[aspect]).view((-1, num_neg_sampled+1, self.text_embedding_size)) 
            else:
                batch_target[aspect] = target_embeddings(batch_target_index[aspect]).view((-1, num_neg_sampled+1, self.embedding_size)) 
            
            # Hidden layers
            input_layers = [self.dropout(batch_input[aspect])]
            if (input_type == 'node'):
                for i in range(1):
                    new_hidden_layer = self.apply_text_embed(input_layers[-1])
                    input_layers.append(new_hidden_layer)
            input_embed[aspect] = input_layers[-1]

            target_layers = [self.dropout(batch_target[aspect])]
            if (target_type == 'node'):
                for i in range(1):
                    new_hidden_layer = (self.apply_text_embed(target_layers[-1].view(-1, self.text_embedding_size))
                        .view(-1, num_neg_sampled+1, self.embedding_size))
                    target_layers.append(new_hidden_layer)
            target_embed[aspect] = target_layers[-1]

            sim_score[aspect] = torch.bmm(
                target_embed[aspect],
                input_embed[aspect].view(-1, embedding_size, 1)).view(-1, num_neg_sampled+1)
            
            # if (aspect == 'neighbor_neg'):
            #     _, pred_label = torch.max(sim_score[aspect], dim=1)
            #     print(sum(to_value(pred_label) == 0), pred_label.size(0), len(batch_input_index['label']))       
            # Compute the softmax loss, using a sample of the negative labels each time.
            target = Variable(torch.cuda.LongTensor(batch_input[aspect].size(0)).zero_(), requires_grad=False)
            loss_all[aspect] = self.CrossEntropyLoss(sim_score[aspect], target)
            
            # Compute the logistic loss, using a sample of the negative labels each time.
            # loss_all[aspect] = -tf.reduce_sum(tf.log(tf.sigmoid(tf.multiply(sim_score[aspect], tf.constant([1.0] + [-1.0] * self.num_sampled)))))
            
            rate = 1.0
            loss += loss_all[aspect] * rate
        
        if (self.train_method == 'local'):
            return loss, loss_all

        unknown_labels, known_labels, all_labels, neighbor_pos_candis, neighbor_neg_candis, author_candis, batch_sizes = infer_data
        # print(unknown_labels, known_labels, all_labels, neighbor_pos_candis, neighbor_neg_candis, author_candis, batch_sizes)
        batch_inferencer = ILPInferencer(unknown_labels, known_labels, all_labels, neighbor_pos_candis, neighbor_neg_candis, author_candis,
            batch_sizes, self, relaxation='ILP', train_method=self.train_method)
        # batch_inferencer.model.setParam('PoolSearchMode', 2)
        # batch_inferencer.model.setParam('PoolSolutions', 9)
        batch_inferencer.add_variables_constraints()
        batch_inferencer.optimize()
        result = batch_inferencer.get_train_result()
        # Hinge loss
        temp_list = [self.zero, self.zero + result[1] - result[0]]
        final_scores = torch.cat(temp_list).view(1, -1)
        rate = 1.0
        size = 1
        loss += torch.max(final_scores) * rate / size
        
        # print(to_value(final_scores), loss)
        # exit()
        if (final_scores[0, 1].data[0] < 0):
            print(unknown_labels, known_labels, all_labels, neighbor_pos_candis, neighbor_neg_candis, author_candis, batch_sizes)
            print('\t error happened')
            exit()
        
        return loss, loss_all

    def predict(self, batch_input_index, batch_gold_index, batch_negs_index):
        # Test (predict label)
        # Look up embeddings for inputs.
        label_test = self.node_embeddings(batch_input_index['label_test']).view((-1, self.text_embedding_size))
        # Hidden layers
        test_input_layers = [label_test]
        for i in range(1):
            new_hidden_layer = self.apply_text_embed(test_input_layers[-1], train=False)
            test_input_layers.append(new_hidden_layer)
        input_embed = test_input_layers[-1]

        label_embed = self.label_embeddings(self.label_list_4pred)
        test_target_layers = [label_embed]
        # for i in range(self.num_hidden_layers):
        #     new_hidden_layer = tf.nn.softsign(tf.matmul(test_target_layers[-1], all_weights['%s_W%d' % ('label', i)]) 
        #         + all_weights['%s_B%d' % ('label', i)])
        #     test_target_layers.append(new_hidden_layer)
        target_embed = test_target_layers[-1]

        probs = input_embed.mm(target_embed.t())
        _, pred_label = torch.max(probs, dim=1)
        pred_label = pred_label.view(-1)
        # print(probs.size(), batch_gold_index['label_test'][:10])
        loss = self.CrossEntropyLoss(probs, batch_gold_index['label_test'])

        return probs, pred_label, loss

    '''
    def predict_edge(self, batch_input_index, batch_gold_index, batch_negs_index):
        embedding_size = self.embedding_size
        batch_target_index = {}
        batch_input = {}
        batch_gold = {}
        batch_target = {}
        input_embed = {}
        target_embed = {}
        sim_score = {}
        loss_all = {}

        loss = 0
        for aspect in ['neighbor', 'text']:
            if (len(batch_gold_index[aspect].size()) == 0):
                continue
            # Target Size/embeddings 
            if (aspect == 'neighbor'):
                target_size = self.node_count
                target_embeddings = self.node_embeddings
                input_type, target_type = 'node', 'node'
            elif (aspect == 'label'):
                target_size = self.label_count
                target_embeddings = self.label_embeddings
                input_type, target_type = 'node', 'label'
            elif (aspect == 'text'):
                target_size = self.word_count
                target_embeddings = self.word_embeddings    
                input_type, target_type = 'node', 'text'
            elif (aspect == 'text_label'):
                target_size = self.label_count
                target_embeddings = self.label_embeddings
                input_type, target_type = 'text', 'label'
            
            # Input data.
            batch_target_index[aspect] = torch.cat((batch_gold_index[aspect], batch_negs_index[aspect]), 1)
        
            # Softmax layer
            # all_weights['softmax_W'][aspect] = tf.Variable(tf.truncated_normal([target_size, softmax_width],
            #                          stddev=1.0 / np.sqrt(embedding_size)))
            # all_weights['softmax_B'][aspect] = tf.Variable(tf.zeros([target_size]))
            
            # Model.
            # Look up embeddings for inputs.
            if (aspect == 'text_label'):
                batch_input[aspect] = self.word_embeddings(batch_input_index[aspect]).view((-1, self.embedding_size))
            else:
                batch_input[aspect] = self.node_embeddings(batch_input_index[aspect]).view((-1, self.embedding_size))
            # batch_gold[aspect] = tf.nn.embedding_lookup(target_embeddings, batch_gold_index[aspect])
            batch_target[aspect] = target_embeddings(batch_target_index[aspect]).view((-1, self.num_sampled+1, self.embedding_size)) 
        
            dropout = nn.Dropout(p=0.7)
            # Hidden layers
            input_layers = [batch_input[aspect]]
            # for i in range(self.num_hidden_layers):
            #     new_hidden_layer = tf.nn.softsign(tf.matmul(input_layers[-1], all_weights['%s_W%d' % (input_type, i)]) + all_weights['%s_B%d' % (input_type, i)])
            #     input_layers.append(new_hidden_layer)
            input_embed[aspect] = input_layers[-1]

            target_layers = [batch_target[aspect]]
            # for i in range(self.num_hidden_layers):
            #     new_hidden_layer = tf.nn.softsign(tf.reshape(tf.matmul(tf.reshape(target_layers[-1], [-1, self.embedding_size//pow(2, i)]), all_weights['%s_W%d' % (target_type, i)]), 
            #         [-1, self.num_sampled+1, self.embedding_size//pow(2, i+1)]) 
            #         + all_weights['%s_B%d' % (target_type, i)])
            #     target_layers.append(new_hidden_layer)
            target_embed[aspect] = target_layers[-1]

            # Compute the nce loss, using a sample of the negative labels each time.
            # loss_all[aspect] = tf.cond(tf.equal(tf.shape(batch_target[aspect])[0], 0),   
            #     lambda: 0.0,
            #     lambda: tf.reduce_mean(tf.nn.nce_loss(target_embeddings, all_weights['softmax_B'][aspect], batch_gold_index[aspect], 
            #                                           input_embed[aspect], self.num_sampled, target_size)))

            sim_score[aspect] = torch.bmm(
                target_embed[aspect],
                input_embed[aspect].view(-1, embedding_size//pow(2, self.num_hidden_layers), 1)).view(-1, self.num_sampled+1)
            
            _, pred = torch.max(sim_score[aspect], dim=1)
            pred_val = to_value(pred)
            correct = sum(pred_val == 0)
            total = len(pred_val)
            print(aspect, correct/total)
    '''