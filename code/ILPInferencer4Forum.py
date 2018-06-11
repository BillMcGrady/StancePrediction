import copy
import gurobipy as grb
import numpy as np
from collections import OrderedDict
from utils import *
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from seaborn import distplot

class ILPInferencer(object):

    def __init__(self, unknown_labels, known_labels, all_labels, neighbor_pos_candis, neighbor_neg_candis, author_candis, batch_sizes,
        model, class_weight=None, relaxation = 'ILP', train_method='test', fold=0):
        """
        Initializes a Gurobi ILP Inferencer given a set of debate decisions.
        """
        # self.nodes_infor = nodes_infor
        self.unknown_labels = unknown_labels
        self.known_labels = known_labels
        self.all_labels = all_labels
        self.neighbor_pos_candis = neighbor_pos_candis
        self.neighbor_neg_candis = neighbor_neg_candis
        self.author_candis = author_candis
        self.batch_sizes = batch_sizes
        self.embedding_size = 300
        self.text_embedding_size = 4800
        self.node_embeds = model.node_embeddings
        self.author_embeds = model.author_embeddings
        self.label_embeds = model.label_embeddings
        self.node_fc1 = model.node_fc1
        self.tanh = nn.Tanh()
        self.class_weight = class_weight
        self.label_var_4pred = model.label_list_4pred 
        self.label_list_4pred = [int(val) for val in to_value(self.label_var_4pred)]
        # print(self.label_list_4pred)
        self.num_labels = 2
        self.neighbor_bonus = 1.0
        self.model = grb.Model('model')
        self.model.setParam('OutputFlag', 0)
        # self.model.setParam('PoolSearchMode', 2)
        # Set the sense to maximization
        self.model.setAttr("ModelSense", -1)
        self.relaxation = relaxation
        if (relaxation == 'LP'):
            self.var_type = grb.GRB.CONTINUOUS 
        else:
            self.var_type = grb.GRB.BINARY
        if (train_method == 'test'):
            self.dropout = nn.Dropout(p=0)
        else:
            self.dropout = nn.Dropout(p=0.7)
        self.coeff_sim = model.coeff_sim
        self.coeff_const = model.coeff_const
        self.train_method = train_method
        self.infer_method = 'const' # 'const'
        self.infer_consecutive = True
        self.infer_ac = True
        self.fold = fold
        self.zero = Variable(torch.from_numpy(np.float32([0])).cuda(), requires_grad=False)
    
    def apply_text_embed(self, x):
        x2 = self.dropout(self.tanh(self.node_fc1(x)))
        # x2 = self.dropout(self.node_fc2(x1))
        return x2

    def add_variables_constraints(self):
        """
        Adds variables and constraints based on all debate decisions in this instance.
        """
        #print ("Adding Variables...")
        self.varDict = OrderedDict()
        self.objDict = OrderedDict()
        self.torchDict = OrderedDict()
        self.same_node_vars = []
        nnsoftmax = nn.Softmax()
        
        varCounter = 0
        constrCounter = 0
        # debugout = open('ruleWeights.txt', 'w')

        for node_id in self.known_labels.keys():
            gold_label = self.known_labels[node_id]
            same_node_vars = []
            for label_id in self.label_list_4pred:
                var_name = 'node_%d_label_%d' % (node_id, label_id)
                rvar = self.model.addVar(
                    obj=0,
                    lb=0.0, ub=1.0, vtype=self.var_type,
                    name=var_name)
                self.varDict[var_name] = rvar
                varCounter += 1
                same_node_vars.append(rvar)

                # Constraint type 1: fix the label for known nodes
                if (gold_label == label_id):
                    self.model.addConstr(rvar == 1, name="c_"+str(constrCounter))
                    constrCounter += 1
            
            # Constraint type 2: a node can only take one label
            self.model.addConstr(grb.quicksum(same_node_vars) == 1, name="c_"+str(constrCounter))
            constrCounter += 1

        gold_label_vars = []
        for ind, node_id in enumerate(self.unknown_labels):
            label_start, label_end = self.batch_sizes['label'][ind:ind+2]
            label_size = label_end - label_start
            node_id_var = get_torch_var(np.asarray([node_id]), 'Long')
            node_id_embed = self.apply_text_embed(self.dropout(self.node_embeds(node_id_var)))

            # label decisions
            if (label_size != 0):
                all_label_embeds = self.dropout(self.label_embeds(self.label_var_4pred))
                label_scores = torch.mm(node_id_embed, all_label_embeds.t()).view(-1)
                same_node_vars = []
                for label_id in self.label_list_4pred:
                    score = label_scores[label_id-self.label_list_4pred[0]] # * self.class_weight[label_id]
                    var_name = 'node_%d_label_%d' % (node_id, label_id)
                    rvar = self.model.addVar(
                            obj=score.data[0] + 0.1 * (self.train_method != 'test') * (self.all_labels[node_id] != label_id), 
                            lb=0.0, ub=1.0, vtype=self.var_type,
                            name=var_name)
                    self.varDict[var_name] = rvar
                    self.torchDict[var_name] = score + 0.1 * (self.train_method != 'test') * (self.all_labels[node_id] != label_id)
                    varCounter += 1
                    same_node_vars.append(rvar)

                    if (self.all_labels[node_id] == label_id):
                        gold_label_vars.append(rvar)
                        # self.model.addConstr(rvar == 1, name="c_"+str(constrCounter))
                        # constrCounter += 1
            

                # Constraint type 2: a node can only take one label
                self.model.addConstr(grb.quicksum(same_node_vars) == 1, name="c_"+str(constrCounter))
                constrCounter += 1
        
        # if (len(gold_label_vars) > 0):
        #     self.model.addConstr(grb.quicksum(gold_label_vars) <= len(gold_label_vars)-1, name="c_"+str(constrCounter))
        #     constrCounter += 1
        if (self.infer_consecutive):
            pos_error, neg_error = 0, 0
            pos_list, neg_list = [], []
            for ind, node_id in enumerate(self.unknown_labels):
                neighbor_start_pos, neighbor_end_pos = self.batch_sizes['neighbor_pos'][ind:ind+2]
                neighbor_size_pos = neighbor_end_pos - neighbor_start_pos
                neighbor_start_neg, neighbor_end_neg = self.batch_sizes['neighbor_neg'][ind:ind+2]
                neighbor_size_neg = neighbor_end_neg - neighbor_start_neg
                node_id_var = get_torch_var(np.asarray([node_id]), 'Long')
                node_id_embed = self.apply_text_embed(self.dropout(self.node_embeds(node_id_var)))
                
                # neighbor decisions
                if (neighbor_size_pos > neighbor_size_neg):
                    node_neighbor_candis = self.neighbor_pos_candis[neighbor_start_pos:neighbor_start_pos+1]
                    neighbor_decisions = get_torch_var(node_neighbor_candis, 'Long')
                    neighbor_embeds = (self.apply_text_embed(
                        self.dropout(self.node_embeds(neighbor_decisions).view(-1, self.text_embedding_size)))
                        .view(1, 1, self.embedding_size))
                    node_embed_neigh = (node_id_embed.view(1, self.embedding_size, 1)
                        .expand(len(neighbor_decisions), self.embedding_size, 1))
                    neighbor_scores = torch.bmm(neighbor_embeds, node_embed_neigh)
                    for i in range(len(node_neighbor_candis)):       
                        neigh_id = node_neighbor_candis[i, 0]             
                        # same label neighbors
                        if (node_id not in self.all_labels or neigh_id not in self.all_labels):
                            continue
                        # if (neigh_id in self.unknown_labels and node_id >= neigh_id):
                        #     continue
                        pos_list.append(neighbor_scores[i, 0].data[0])
                        if (score.data[0] < 0):
                            pos_error += .0001
                        pos_error += 1
                        for node_lid in self.label_list_4pred:
                            if (node_lid % 2 == 0):
                                next_lid = node_lid + 1
                            else:
                                next_lid = node_lid - 1
                            if (self.infer_method == 'const'):
                                next_lids = [next_lid]
                            else:
                                next_lids = self.label_list_4pred
                            for neigh_lid in next_lids:
                                var_name = 'node_label_%d_%d_node_label_%d_%d' % (node_id, node_lid, neigh_id, neigh_lid)
                                if (var_name in self.varDict):
                                    break
                                if (self.infer_method == 'const'):
                                    score = self.coeff_const + self.zero
                                else:
                                    score = (-1 + 2 * (node_lid == neigh_lid)) * (neighbor_scores[i, 0]-self.coeff_sim)
                                pvar = self.model.addVar(
                                    obj= score.data[0],
                                    lb=0.0, ub=1.0, vtype=self.var_type,
                                    name=var_name)
                                self.varDict[var_name] = pvar
                                self.torchDict[var_name] = score.clone()
                                varCounter += 1
                                
                                # Constraint type 3: node pair decision variables should be consistent with each node decision
                                avar = self.varDict['node_%d_label_%d' % (node_id, node_lid)]
                                bvar = self.varDict['node_%d_label_%d' % (neigh_id, neigh_lid)]
                                self.model.addGenConstrAnd(pvar, [avar, bvar], name="c_"+str(constrCounter))
                                constrCounter += 1

                # neighbor decisions
                if (neighbor_size_neg > neighbor_size_pos):
                    node_neighbor_candis = self.neighbor_neg_candis[neighbor_start_neg:neighbor_start_neg+1]
                    neighbor_decisions = get_torch_var(node_neighbor_candis, 'Long')
                    neighbor_embeds = (self.apply_text_embed(
                        self.dropout(self.node_embeds(neighbor_decisions).view(-1, self.text_embedding_size)))
                        .view(1, 1, self.embedding_size))
                    node_embed_neigh = (node_id_embed.view(1, self.embedding_size, 1)
                        .expand(len(neighbor_decisions), self.embedding_size, 1))
                    neighbor_scores = torch.bmm(neighbor_embeds, node_embed_neigh)
                    for i in range(len(node_neighbor_candis)):       
                        neigh_id = node_neighbor_candis[i, 0]             
                        # same label neighbors
                        if (node_id not in self.all_labels or neigh_id not in self.all_labels):
                            continue
                        # if (neigh_id in self.unknown_labels and node_id >= neigh_id):
                        #     continue
                        neg_list.append(neighbor_scores[i, 0].data[0])
                        if (score.data[0] > 0):
                            neg_error += .0001
                        neg_error += 1
                        for node_lid in self.label_list_4pred:
                            if (node_lid % 2 == 0):
                                next_lid = node_lid + 1
                            else:
                                next_lid = node_lid - 1
                            if (self.infer_method == 'const'):
                                next_lids = [next_lid]
                            else:
                                next_lids = self.label_list_4pred
                            for neigh_lid in next_lids:
                                var_name = 'node_label_%d_%d_node_label_%d_%d' % (node_id, node_lid, neigh_id, neigh_lid)
                                if (var_name in self.varDict):
                                    break
                                if (self.infer_method == 'const'):
                                    score = self.coeff_const + self.zero
                                else:
                                    score = (-1 + 2 * (node_lid == neigh_lid)) * (neighbor_scores[i, 0]-self.coeff_sim)
                                pvar = self.model.addVar(
                                    obj= score.data[0],
                                    lb=0.0, ub=1.0, vtype=self.var_type,
                                    name=var_name)
                                self.varDict[var_name] = pvar
                                self.torchDict[var_name] = score.clone()
                                varCounter += 1
                                
                                # Constraint type 3: node pair decision variables should be consistent with each node decision
                                avar = self.varDict['node_%d_label_%d' % (node_id, node_lid)]
                                bvar = self.varDict['node_%d_label_%d' % (neigh_id, neigh_lid)]
                                self.model.addGenConstrAnd(pvar, [avar, bvar], name="c_"+str(constrCounter))
                                constrCounter += 1
            self.pos_list, self.neg_list = pos_list, neg_list
        
            # if (self.train_method == 'test'):
            #     print('error', pos_error, neg_error)
            #     plt.clf() 
            #     distplot(pos_list, color='b')
            #     distplot(neg_list, color='g')
            #     plt.savefig('result/test_%s.png' % self.fold)
                # plt.show()
        
        if (not self.infer_ac):
            self.model.update()
            return

        for ind, node_id in enumerate(self.unknown_labels):
            author_start, author_end = self.batch_sizes['author'][ind:ind+2]
            author_size = author_end - author_start
            node_id_var = get_torch_var(np.asarray([node_id]), 'Long')
            node_id_embed = self.apply_text_embed(self.dropout(self.node_embeds(node_id_var)))

            if (author_size != 0):
                # text decisions
                node_author_candis = self.author_candis[author_start:author_end]
                author_decisions = get_torch_var(node_author_candis, 'Long')
                author_embeds = self.dropout(self.author_embeds(author_decisions))

                # author label decisions
                if (node_id not in self.all_labels):
                    continue
                # text_atten = nnsoftmax(text_scores[:,0].view(1, -1)).view(-1)
                
                all_label_embeds = self.dropout(self.label_embeds(self.label_var_4pred))
                label_embeds_expand = (all_label_embeds.t().contiguous().view(1, self.embedding_size, self.num_labels)
                    .expand(len(author_decisions), self.embedding_size, self.num_labels))
                author_label_scores = torch.bmm(author_embeds, label_embeds_expand)
                for i in range(len(node_author_candis)):
                    for j in range(len(node_author_candis[0])):
                        author_id = node_author_candis[i, j]
                        same_node_vars = []
                        for label_id in self.label_list_4pred:
                            score = author_label_scores[i, j, label_id-self.label_list_4pred[0]]
                            score_val = author_label_scores[i, j, label_id-self.label_list_4pred[0]].data[0] # * self.class_weight[label_id] # /len(node_text_candis)
                            var_name = 'author_%d_label_%d' % (author_id, label_id)
                            if (var_name not in self.varDict):
                                rvar = self.model.addVar(
                                    obj=score_val, 
                                    lb=0.0, ub=1.0, vtype=self.var_type,
                                    name=var_name)
                                self.varDict[var_name] = rvar
                                self.objDict[var_name] = score_val
                                self.torchDict[var_name] = score.clone()
                                varCounter += 1
                            else:
                                rvar = self.varDict[var_name]
                                # rvar.setAttr("obj", self.objDict[var_name]+score_val)
                                # self.objDict[var_name] += score_val
                                # self.torchDict[var_name] += score.clone()
                            same_node_vars.append(rvar)
                            # Constraint type 4: text to label only added when this text is linked with this node
                            # node_author_var = self.varDict['node_%d_author_%d' % (node_id, author_id)]
                            # self.model.addConstr(rvar <= node_author_var, name="c_"+str(constrCounter))
                            # constrCounter += 1
                        
                            # Constraint type 5: node and its author should choose the same label
                            var_name = 'node_%d_label_%d' % (node_id, label_id)
                            if (var_name in self.varDict):
                                node_label_var = self.varDict[var_name]
                                self.model.addConstr(rvar == node_label_var, name="c_"+str(constrCounter))
                                # self.model.addGenConstrIndicator(node_author_var, True, node_label_var == rvar, name="c_"+str(constrCounter))
                                constrCounter += 1
                
                        # Constraint type 5: a text can only take one label, small or equal to 1 because this text may not link to this node
                        self.model.addConstr(grb.quicksum(same_node_vars) <= 1, name="c_"+str(constrCounter))
                        constrCounter += 1
        
        self.model.update()
        
    def optimize(self):
        """
        Optimizes the current model.
        """
        # print(self.model.NumConstrs, self.model.NumVars, self.model.NumGenConstrs)
        self.model.optimize()

    def get_train_result(self, result_typ='x'):
        sol_count = 1 # self.model.getAttr('SolCount')
        # status = self.model.getAttr('Status')
        # print(sol_count)
        # print('gold count', self.same_label_count)
        # status = self.model.getAttr('Status')
        # print(status)
        all_result = [self.get_gold_score()]
        # fout = open('train_infer_result.txt', 'w')
        # first, second = True, True
        for sol_num in range(sol_count):
            # self.model.setParam('SolutionNumber', sol_num)
            current = 0
            cur_count = 0
            # fout.write('----------------- %d -----------------\n' % sol_num)
            for key in self.torchDict.keys():
                # if (first and 'node_label' in key):
                #     print('label pred', current)
                #     first = False
                # if (second and 'author' in key):
                #     print('label+neighbor pred', current)
                #     second = False     
                rvar = self.varDict[key]
                rvalue = rvar.getAttr(result_typ)
                # print(key, rvalue)
                if ('label' not in key):
                    continue        
                if (rvalue > 0.99):
                    # print('infer', key, self.torchDict[key].data[0])
                    current += self.torchDict[key]
                    cur_count += 1
                
                # fout.write(str(key) + ' ' + str(rvalue) + ' ' + str(rvalue > 0.99) + ' ' + str(self.torchDict[key]) + ' \n')
            # print('final pred', current, cur_count, len(self.torchDict))
            all_result.append(current)
        # fout.close()
        # print(all_result)

        return all_result

    def get_gold_score(self, result_typ='x'):
        mis = 0
        corr = 0
        
        self.node_author = {}
        self.author_labels_count = {}
        self.author_labels = {}
        for ind, node_id in enumerate(self.unknown_labels):
            label_start, label_end = self.batch_sizes['label'][ind:ind+2]
            label_size = label_end - label_start
            text_start, text_end = self.batch_sizes['author'][ind:ind+2]
            text_size = text_end - text_start
            if (label_size != 0 and text_size != 0):
                node_text = self.author_candis[text_start, 0]
                self.node_author[node_id] = node_text
                if (node_text not in self.author_labels_count):
                    self.author_labels_count[node_text] = [0, 0]
                self.author_labels_count[node_text][self.all_labels[node_id] % 2] += 1

        for author in self.author_labels_count.keys():
            author_post_counts = self.author_labels_count[author]
            if (author_post_counts[0] > author_post_counts[1]):
                self.author_labels[author] = self.label_list_4pred[0]
            else:
                self.author_labels[author] = self.label_list_4pred[1]

        gold_score, gold_count, gold_score_label, gold_score_neigh, gold_score_text = 0, 0, 0, 0, 0
        for ind, node_id in enumerate(self.unknown_labels):
            label_start, label_end = self.batch_sizes['label'][ind:ind+2]
            label_size = label_end - label_start
            if (label_size != 0):
                var_name = 'node_%d_label_%d' % (node_id, self.author_labels[self.node_author[node_id]])
                gold_score += self.torchDict[var_name]
                # print('gold', var_name, self.torchDict[var_name].data[0])
                gold_score_label += self.torchDict[var_name]

        for ind, node_id in enumerate(self.unknown_labels):
            neighbor_start_pos, neighbor_end_pos = self.batch_sizes['neighbor_pos'][ind:ind+2]
            neighbor_size_pos = neighbor_end_pos - neighbor_start_pos
            neighbor_start_neg, neighbor_end_neg = self.batch_sizes['neighbor_neg'][ind:ind+2]
            neighbor_size_neg = neighbor_end_neg - neighbor_start_neg
            if (neighbor_size_pos > neighbor_size_neg):
                node_neighbor_candis = self.neighbor_pos_candis[neighbor_start_pos:neighbor_start_pos+1, 0]
                for i in range(len(node_neighbor_candis)):
                    neigh_id = node_neighbor_candis[i]
                    var_name = 'node_%d_node_%d' % (node_id, neigh_id)
                    
                    # gold_score += self.torchDict[var_name]
                    # gold_score_neigh += self.torchDict[var_name]
                    
                    if (node_id not in self.all_labels or neigh_id not in self.all_labels):
                        continue
                    node_lid = self.author_labels[self.node_author[node_id]]
                    neigh_lid = self.author_labels[self.node_author[neigh_id]]
                    if (self.infer_method == 'const' and node_lid == neigh_lid):
                        continue
                    var_name = 'node_label_%d_%d_node_label_%d_%d' % (node_id, node_lid, neigh_id, neigh_lid)
                    gold_score += self.torchDict[var_name]
                    # print('gold', var_name, self.torchDict[var_name].data[0])
                    gold_score_neigh += self.torchDict[var_name]
                            
                    rvar = self.varDict[var_name]
                    rvalue = rvar.getAttr(result_typ)
                    # if (rvalue < 0.99):
                    #     mis += 1
                    # else:
                    #     corr += 1

            if (neighbor_size_neg > neighbor_size_pos):
                node_neighbor_candis = self.neighbor_neg_candis[neighbor_start_neg:neighbor_start_neg+1, 0]
                for i in range(len(node_neighbor_candis)):
                    neigh_id = node_neighbor_candis[i]
                    var_name = 'node_%d_node_%d' % (node_id, neigh_id)
                    # gold_score += self.torchDict[var_name]
                    # gold_score_neigh += self.torchDict[var_name]
                    
                    if (node_id not in self.all_labels or neigh_id not in self.all_labels):
                        continue
                    node_lid = self.author_labels[self.node_author[node_id]]
                    neigh_lid = self.author_labels[self.node_author[neigh_id]]
                    if (self.infer_method == 'const' and node_lid == neigh_lid):
                        continue
                    var_name = 'node_label_%d_%d_node_label_%d_%d' % (node_id, node_lid, neigh_id, neigh_lid)
                    gold_score += self.torchDict[var_name]
                    # print('gold', var_name, self.torchDict[var_name].data[0])
                    gold_score_neigh += self.torchDict[var_name]
                            
                    rvar = self.varDict[var_name]
                    rvalue = rvar.getAttr(result_typ)
                    # if (rvalue < 0.99):
                    #     mis += 1
                    # else:
                    #     corr += 1
        
        gold_set = set([])
        for ind, node_id in enumerate(self.unknown_labels):
            text_start, text_end = self.batch_sizes['author'][ind:ind+2]
            text_size = text_end - text_start
            if (text_size != 0):
                node_text_candis = self.author_candis[text_start:text_end, 0]
                for i in range(len(node_text_candis)):
                    text_id = node_text_candis[i]
                    var_name = 'node_%d_author_%d' % (node_id, text_id)
                    # gold_score += self.torchDict[var_name]
                    # gold_count += 1
                    # gold_score_text += self.torchDict[var_name]
                    # print(var_name, self.torchDict[var_name].data[0])

                    if (node_id not in self.all_labels):
                        continue
                    label_id = self.author_labels[self.node_author[node_id]]
                    var_name = 'author_%d_label_%d' % (text_id, label_id)
                    if (text_id in gold_set):
                        continue
                    gold_set.add(text_id)
                    gold_score += self.torchDict[var_name]
                    # print('gold', var_name, self.torchDict[var_name].data[0])
                    gold_count += 1
                    gold_score_text += self.torchDict[var_name]
                    # print(var_name, self.torchDict[var_name].data[0])

                    rvar = self.varDict[var_name]
                    rvalue = rvar.getAttr(result_typ)
                    # if (rvalue < 0.99):
                    #     mis += 1
                    # else:
                    #     corr += 1
            
        # print('miss', mis, corr, gold_count, gold_score.data[0], gold_score_label.data[0], gold_score_neigh.data[0], gold_score_text.data[0])
            
        return gold_score

    def get_result(self, result_typ='x'):
        # print(self.model.getAttr('SolCount'))
        # print('same label count', self.same_label_count)
        sol_count = self.model.getAttr('SolCount')
        status = self.model.getAttr('Status')
        all_result = []
        for sol_num in range(1):
            # self.model.setParam('SolutionNumber', sol_num)

            result = {}
            for ind, node_id in enumerate(self.unknown_labels):
                label_start, label_end = self.batch_sizes['label'][ind:ind+2]
                label_size = label_end - label_start
                # text_start, text_end = self.batch_sizes['text'][ind:ind+2]
                # text_size = text_end - text_start
                # neighbor_start, neighbor_end = self.batch_sizes['neighbor'][ind:ind+2]
                # neighbor_size = neighbor_end - neighbor_start
                result[node_id] = {'label': [], 'text': [], 'neighbor': []}
                
                if (label_size != 0):
                    best_val = -1
                    for label_id in self.label_list_4pred:
                        rvar = self.varDict['node_%d_label_%d' % (node_id, label_id)]
                        rvalue = rvar.getAttr(result_typ)
                        if (rvalue > best_val):
                            best_val = rvalue
                            best_id = label_id
                    result[node_id]['label'].append(best_id)
                '''    
                if (text_size != 0):
                    # text decisions
                    node_text_candis = self.text_candis[text_start:text_end]
                    for i in range(len(node_text_candis)):
                        best_val = -1
                        for j in range(len(node_text_candis[0])):
                            text_id = node_text_candis[i, j]
                            var_name = 'node_%d_text_%d' % (node_id, text_id)
                            rvar = self.varDict[var_name]
                            rvalue = rvar.getAttr(result_typ)
                            if (rvalue > best_val):
                                best_val = rvalue
                                best_id = text_id
                        result[node_id]['text'].append(best_id)
                
                if (neighbor_size != 0):
                    # neighbor decisions
                    node_neighbor_candis = self.neighbor_candis[neighbor_start:neighbor_end]
                    for i in range(len(node_neighbor_candis)):
                        best_val = -1
                        for j in range(len(node_neighbor_candis[0])):
                            neigh_id = node_neighbor_candis[i, j]
                            var_name = 'node_%d_node_%d' % (node_id, neigh_id)
                            rvar = self.varDict[var_name]
                            rvalue = rvar.getAttr(result_typ)
                            if (rvalue > best_val):
                                best_val = rvalue
                                best_id = neigh_id
                        result[node_id]['neighbor'].append(best_id)
                '''
            all_result.append(result)

        return all_result

