from seaborn import distplot
from sklearn.metrics import f1_score
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ILPInferencer4Forum import *
from utils import *

from DynamicNet4Forum import * 

use_cuda = True

class NetworkEmbedding(object):

    def __init__(self, data, model_folder='result/test/', fold=None):
        self.data = data
        self.fold = fold
        if (not os.path.isdir(model_folder)):
            os.mkdir(model_folder)
        load_file_name = '%s_joint_tanh_mean_d7' % data.domain
        save_file_name = '%s_global_tanh_mean_d7' % data.domain
        # 'abortion_joint+3+attri_l1_b10_shuf_mean_d7'
        if (fold is not None):
            load_file_name += '_%d' % fold
            save_file_name += '_%d' % fold #  + '_0'
        self.load_model_path = os.path.join(model_folder, load_file_name) 
        self.save_model_path = os.path.join(model_folder, save_file_name) 
        self.batch_size = 10
        self.labeled_nodes_set = set(data.labeled_nodes)
        print('total label in this domain', len(self.labeled_nodes_set))
        self.train_nodes = data.training_nodes
        self.train_nodes_set = set(self.train_nodes)
        self.train_size = len(data.training_nodes)
        # self.valid_nodes = data.valid_nodes
        # self.valid_size = len(data.valid_nodes)
        self.test_nodes = data.test_nodes
        self.test_size = len(data.test_nodes)
        self.node_list = data.node_list
        self.thread_list = data.thread_list
        self.weight_decay = 0
        self.init_patience = 5
        self.num_sampled = 5
        torch.cuda.manual_seed(1)
        torch.manual_seed(1)
        np.random.seed(1)
        self.neg_rng = np.random.RandomState(1)
        self.shuffle_rng = data.shuffle_rng
        self.train_method = 'global'
        self.data_size = data.data_size
        if (self.train_method != 'local'):
            self.data_size = data.thread_size
            self.init_patience = 1
        self.retrain = False
            
    ############################
    # Chunk the data to be passed into the tensorflow Model
    ###########################

    def negative_sampling(self, total, pos_list, n=None):
        if n is None:
            n = len(pos_list)
        pos_list = set(pos_list)
        sampled = 0
        neg_list = []
        while sampled < self.num_sampled * n:
            neg = self.neg_rng.choice(total)
            if (neg not in pos_list):
                neg_list.append(neg)
                sampled += 1

        return neg_list

    def negative_sampling_label(self, pos_list):
        neg_list = []
        for item in pos_list:
            if (item <= 7):
                if (item % 2 == 0):
                    neg_list.append(item+1)
                else:
                    neg_list.append(item-1)
            elif (item <= 10):
                neg = self.neg_rng.choice([k for k in range(8, 11) if k != item])
                neg_list.append(neg)
            elif (item <= 13):
                neg = self.neg_rng.choice([k for k in range(11, 14) if k != item])
                neg_list.append(neg)
            elif (item <= 19):
                neg = self.neg_rng.choice([k for k in range(14, 20) if k != item])
                neg_list.append(neg)
            elif (item <= 33):
                neg = self.neg_rng.choice([k for k in range(20, 34) if k != item])
                neg_list.append(neg)
            else:
                neg = self.neg_rng.choice([k for k in range(34, 40) if k != item])
                neg_list.append(neg)
        return neg_list

    def generate_batch_create(self, start_idx, batch_size, data='train'):
        end_idx = start_idx + batch_size
        if (data == 'train'):
            if (end_idx > self.data_size):
                end_idx = self.data_size
            if (self.train_method == 'local'):
                node_list = self.node_list[start_idx:end_idx]
            else:
                thread_list = self.thread_list[start_idx:end_idx]
                node_list =[]
                for thread in thread_list:
                    node_list.extend(thread)
                # print('len node list', node_list)
        elif (data == 'train_test'):
            if (end_idx > self.train_size):
                end_idx = self.train_size
            node_list = self.train_nodes[start_idx:end_idx]
        elif (data == 'valid'):
            if (end_idx > self.valid_size):
                end_idx = self.valid_size
            node_list = self.valid_nodes[start_idx:end_idx]
        elif (data == 'test'):
            if (end_idx > self.test_size):
                end_idx = self.test_size
            node_list = self.test_nodes[start_idx:end_idx]        

        batch_input = {'neighbor_pos': [], 'neighbor_neg': [], 'author_allies': [],'label': [], 'author': [], 'author_label': [], 'label_test': []}
        batch_gold = {'neighbor_pos': [], 'neighbor_neg': [], 'author_allies': [],'label': [], 'author': [], 'author_label': [], 'label_test': []}
        batch_negs = {'neighbor_pos': [], 'neighbor_neg': [], 'author_allies': [],'label': [], 'author': [], 'author_label': []}
        gold_label = []
        neighbor_labels = []
        batch_sizes = {'neighbor_pos': [0], 'neighbor_neg': [0], 'author_allies': [], 'label': [0], 'author': [0]}

        unknown_labels = []
        known_labels = {}
        all_labels = {}
        # adjedges = {}
        # node_text = {}
        # train_label = []

        for node_id in node_list:
            if (node_id in self.labeled_nodes_set):
                batch_input['label_test'].append(node_id)
                batch_gold['label_test'].append(self.data.nodes_infor[node_id]['node_labels'][0])
                gold_label.append(self.data.nodes_infor[node_id]['node_labels'][0])

            label_size = len(self.data.nodes_infor[node_id]['node_labels'])
            author_size = len(self.data.nodes_infor[node_id]['node_author'])          
            if (self.train_method == 'local'):
                author_attris = self.data.nodes_infor[node_id]['node_author_attri']
            else:
                author_attris = [attri for attri in self.data.nodes_infor[node_id]['node_author_attri'] 
                    if attri > 7]
            author_attri_size = len(author_attris)
            author_allies_size = len(self.data.nodes_infor[node_id]['node_author_allies'])
            
            unknown_labels.append(node_id)

            if ('train' in data and node_id in self.test_nodes):
                # or self.data.nodes_infor[node_id]['node_labels'][0] not in [0, 1]):
                batch_sizes['neighbor_pos'].append(batch_sizes['neighbor_pos'][-1])
                batch_sizes['neighbor_neg'].append(batch_sizes['neighbor_neg'][-1])
                batch_sizes['label'].append(batch_sizes['label'][-1])
                
            else:
                batch_input['label'].extend([node_id] * label_size)
                batch_gold['label'].extend(self.data.nodes_infor[node_id]['node_labels'])
                batch_negs['label'].extend(self.negative_sampling_label(
                    self.data.nodes_infor[node_id]['node_labels']))
                batch_sizes['label'].append(batch_sizes['label'][-1]+label_size)
                all_labels[node_id] = self.data.nodes_infor[node_id]['node_labels'][0]
                
                if (self.train_method == 'local'):
                    batch_input['author_label'].extend(self.data.nodes_infor[node_id]['node_author'] * author_size)
                    batch_gold['author_label'].extend(self.data.nodes_infor[node_id]['node_labels'])
                    batch_negs['author_label'].extend(self.negative_sampling_label(
                        self.data.nodes_infor[node_id]['node_labels']))

                pos_label = all_labels[node_id]
                if (pos_label % 2 == 0):
                    neg_label = pos_label + 1
                else:
                    neg_label = pos_label - 1
                neighbor_pos_size = 0
                neighbor_neg_size = 0
                for neigh_id, rebuttal in self.data.nodes_infor[node_id]['node_connected']:
                    if (rebuttal == 1):
                        batch_input['neighbor_pos'].append(node_id)
                        batch_gold['neighbor_pos'].append(neigh_id)
                        neighbor_pos_size += 1
                    else:
                        batch_input['neighbor_neg'].append(node_id)
                        batch_gold['neighbor_neg'].append(neigh_id)
                        neighbor_neg_size += 1
                for rebuttal in [-1, 1] * 3:
                    if (rebuttal == 1):
                        neigh_id = self.neg_rng.choice(self.data.label_node_list[pos_label])
                        batch_input['neighbor_pos'].append(node_id)
                        batch_gold['neighbor_pos'].append(neigh_id)
                        neighbor_pos_size += 1
                    else:
                        neigh_id = self.neg_rng.choice(self.data.label_node_list[neg_label])
                        batch_input['neighbor_neg'].append(node_id)
                        batch_gold['neighbor_neg'].append(neigh_id)
                        neighbor_neg_size += 1
                
                if (neighbor_pos_size != 0):
                    batch_negs['neighbor_pos'].extend(self.negative_sampling(
                        self.data.label_node_list[neg_label], batch_gold['neighbor_pos'][-neighbor_pos_size:]))
                        # self.data.node_count, batch_gold['neighbor_pos'][-neighbor_pos_size:]))
                if (neighbor_neg_size != 0):
                    batch_negs['neighbor_neg'].extend(self.negative_sampling(
                        self.data.label_node_list[pos_label], batch_gold['neighbor_neg'][-neighbor_neg_size:]))
                        # self.data.node_count, batch_gold['neighbor_neg'][-neighbor_neg_size:]))
                batch_sizes['neighbor_pos'].append(batch_sizes['neighbor_pos'][-1]+neighbor_pos_size)
                batch_sizes['neighbor_neg'].append(batch_sizes['neighbor_neg'][-1]+neighbor_neg_size)

            batch_input['author_label'].extend(self.data.nodes_infor[node_id]['node_author'] * author_attri_size)
            batch_gold['author_label'].extend(author_attris)
            batch_negs['author_label'].extend(self.negative_sampling_label(
                author_attris))

            batch_input['author_allies'].extend(self.data.nodes_infor[node_id]['node_author'] * author_allies_size)
            batch_gold['author_allies'].extend(self.data.nodes_infor[node_id]['node_author_allies'])
            batch_negs['author_allies'].extend(self.negative_sampling(self.data.author_count,
                self.data.nodes_infor[node_id]['node_author_allies']))
        
            batch_input['author'].extend([node_id] * author_size)
            batch_gold['author'].extend(self.data.nodes_infor[node_id]['node_author'])
            batch_sizes['author'].append(batch_sizes['author'][-1]+author_size)
            batch_negs['author'].extend(self.negative_sampling(self.data.author_count,
                self.data.nodes_infor[node_id]['node_author']))
        
        def converter(x):
            tensor = torch.from_numpy(np.int64(x)).cuda() if use_cuda else torch.LongTensor(np.int64(x))
            return Variable(tensor, requires_grad=False)

        # neighbor_candis = np.concatenate((np.reshape(batch_gold['neighbor'], (-1, 1)), batch_negs_neighbor), axis=1)
        # text_candis = np.concatenate((np.reshape(batch_gold['text'], (-1, 1)), batch_negs_text), axis=1)
        neighbor_pos_candis = np.reshape(batch_gold['neighbor_pos'], (-1, 1))
        neighbor_neg_candis = np.reshape(batch_gold['neighbor_neg'], (-1, 1))
        author_candis = np.reshape(batch_gold['author'], (-1, 1))
        infer_data = [unknown_labels, known_labels, all_labels, neighbor_pos_candis, neighbor_neg_candis, 
            author_candis, batch_sizes]

        for aspect in batch_input.keys():
            batch_input[aspect] = get_torch_var(batch_input[aspect]) if len(batch_input[aspect]) > 0 else []
        for aspect in batch_gold.keys():
            if (aspect == 'label_test'):
                batch_gold['label_test'] = get_torch_var(batch_gold['label_test']) if len(batch_input['label_test']) > 0 else []
            else:
                batch_gold[aspect] = get_torch_var(np.reshape(batch_gold[aspect], (len(batch_gold[aspect]), 1))) if len(batch_gold[aspect]) > 0 else []
        for aspect in batch_negs.keys():
            if ('label' in aspect):
                batch_negs[aspect] = get_torch_var(np.reshape(batch_negs[aspect], (-1, 1))) if len(batch_input[aspect]) > 0 else []
            else:
                batch_negs[aspect] = get_torch_var(np.reshape(batch_negs[aspect], (-1, self.num_sampled))) if len(batch_input[aspect]) > 0 else []
        gold_label = np.asarray(gold_label)
        
        return batch_input, batch_gold, batch_negs, batch_sizes, gold_label, neighbor_labels, infer_data

    def train(self):
        # print(self.data_size)
        step_delta = self.data_size//self.batch_size + 1 * (self.data_size % self.batch_size > 0)
        # print('step delta', step_delta)

        # negative = None 

        model = DynamicNet(self.data, self.num_sampled, self.train_method)
        # from VariationalAutoEncoder import *
        # model = VariationalAutoEncoder(self.data)
        # model = GraphGaussianEmbedding(self.data)
        if (self.train_method != 'local'):
            model_dict = model.state_dict()
            model_dict.update(torch.load(self.load_model_path))
            model.load_state_dict(model_dict)
        self.model = model
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(parameters, lr=0.001, weight_decay=self.weight_decay)
        # print(parameters)
        
        epoch = 0    
        best_loss = 10000000000.0
        patience = self.init_patience
        average_loss = 0
        start_idx = 0
        loss = [0, 0, 0, 0, 0, 0]
        step = 0

        while (True):
            
            if (not self.retrain):
                break

            batch_input, batch_gold, batch_negs, batch_sizes, gold_label, neighbor_labels, infer_data = \
                self.generate_batch_create(start_idx, self.batch_size, data='train')
            start_idx += self.batch_size
            # if (start_idx % 100 == 0):
            #     print(start_idx)
            if (start_idx >= self.data_size):
                start_idx = 0
            
            if (len(batch_gold['label']) != 0):
            # if ((len(batch_gold['neighbor_pos'].size()) != 0)
            #     or (len(batch_gold['neighbor_neg'].size()) != 0)):
                # or len(batch_gold['author'].size()) != 0):
                # print(batch_input, batch_gold)
                model.zero_grad()
                # print(start_idx)
                l, loss_all = model(batch_input, batch_gold, batch_negs, batch_sizes, neighbor_labels, infer_data)
                l.backward()
                optimizer.step()

                if np.isnan(l.data[0]):
                    print('nan')
                    exit()
                
                average_loss += l.data[0]
                for idx, aspect in enumerate(model.aspects):
                    if (aspect in loss_all):
                        loss[idx] += loss_all[aspect].data[0] 
                del l, loss_all

            if step > 0 and (step + 1) % step_delta == 0:
                # shuffle
                permutation = self.shuffle_rng.permutation(self.data_size)
                if (self.train_method == 'local'):
                    self.node_list = self.node_list[permutation]
                else:
                    self.thread_list = self.thread_list[permutation]

                batch_input, batch_gold, batch_negs, batch_sizes, gold_label, neighbor_labels, infer_data = \
                    self.generate_batch_create(0, self.test_size, data='train_test')
                probs_var, pred_label_var, pred_loss = model.predict(batch_input, batch_gold, batch_negs)
                pred_label = to_value(pred_label_var)+self.data.label_list_4pred[0]
                print('train', pred_label[:10], gold_label[:10], sum(pred_label==gold_label), pred_loss.data[0])
                del probs_var, pred_label_var, pred_loss
                
                batch_input, batch_gold, batch_negs, batch_sizes, gold_label, neighbor_labels, infer_data = \
                    self.generate_batch_create(0, self.test_size, data='test')
                probs_var, pred_label_var, pred_loss = model.predict(batch_input, batch_gold, batch_negs)
                pred_label = to_value(pred_label_var)+self.data.label_list_4pred[0]
                print('test', pred_label[:10], gold_label[:10], sum(pred_label==gold_label), pred_loss.data[0])
                # del probs_var, pred_label_var, pred_loss
                
                # average_loss = len(gold_label) - sum(pred_label==gold_label) # average_loss # / step_delta
                # if (self.train_method != 'local'):
                #     torch.save(model.state_dict(), self.save_model_path + '_%d' % epoch)
                if (average_loss < best_loss):
                    best_loss = average_loss
                    torch.save(model.state_dict(), self.save_model_path)
                    patience = self.init_patience
                else:
                    patience -= 1
                
                if (patience == 0):
                    break
                
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at epoch %d (step %d): %f %s %f %f %d' % (epoch, step, average_loss, str(loss), average_loss-sum(loss), best_loss, patience))
                epoch += 1
                average_loss = 0
                loss = [0, 0, 0, 0, 0, 0]
                # break
            step += 1
        

    def test(self):

        model = self.model
        model.load_state_dict(torch.load(self.save_model_path))
        print('load coeff_const', model.coeff_const.data[0])
        # print('load coeff_sim', model.coeff_sim.data[0])

        best_node_embeds = model.node_embeddings.weight.cpu().data.numpy()
        best_author_embeds = model.author_embeddings.weight.cpu().data.numpy()
        best_label_embeds = model.label_embeddings.weight.cpu().data.numpy()
        # print(np.dot(best_label_embeds[0], best_label_embeds[1]))
        # print(np.dot(best_label_embeds[2], best_label_embeds[3]))
        # print(np.dot(best_label_embeds[4], best_label_embeds[5]))
        # print(np.dot(best_label_embeds[6], best_label_embeds[7]))

        # fcw, fcb = to_value(model.node_fc1.weight), to_value(model.node_fc1.bias)
        # same_stance, diff_stance = [], []
        # for ind1, node1 in enumerate(self.test_nodes):
        #     for ind2 in range(ind1+1, len(self.test_nodes)):
        #         node2 = self.test_nodes[ind2]
        #         node1_embed = np.tanh(np.matmul(fcw, best_node_embeds[node1]) + fcb)
        #         node2_embed = np.tanh(np.matmul(fcw, best_node_embeds[node2]) + fcb)
        #         this_sim = np.dot(node1_embed, node2_embed)
        #         label1 = self.data.nodes_infor[node1]['node_labels'][0]
        #         label2 = self.data.nodes_infor[node2]['node_labels'][0]
        #         if (label1 == label2):
        #             same_stance.append(this_sim)
        #         else:
        #             diff_stance.append(this_sim)
        # print(len(self.test_nodes), len(same_stance), len(diff_stance))
        # return same_stance, diff_stance
        
        # Local Result (train)
        batch_input, batch_gold, batch_negs, batch_sizes, gold_label, neighbor_labels, infer_data = \
            self.generate_batch_create(0, self.train_size, data='train_test')
        probs_var, pred_label_var, loss = model.predict(batch_input, batch_gold, batch_negs)
        probs = to_value(probs_var)
        absv = np.absolute(probs[:, 0] - probs[:, 1])
        # print(absv.shape, np.mean(absv), np.median(absv))
        pred_label = to_value(pred_label_var)
        del probs_var, pred_label_var, loss
        # print('train', sum(pred_label==gold_label), probs[0])
        pred_local = pred_label+self.data.label_list_4pred[0] # np.concatenate((pred_label, pred_label_test))
        gold_local = gold_label # np.concatenate((gold_label, gold_label_test))
        # print(sum(pred_local==gold_local), sum(pred_local==gold_local)/len(pred_local), sum(0==gold_local))
        # print(f1_score(gold_local, pred_local, pos_label=None, average='micro'), 
        #      f1_score(gold_local, pred_local, pos_label=None, average='macro'))
        
        if (self.retrain and self.train_method == 'local'): # self.retrain and 
            model.coeff_const.data[0] = float(np.percentile(absv, 25)) 
            print('change coeff const to %f' % model.coeff_const.data[0]) 
            torch.save(model.state_dict(), self.save_model_path)  
        ret_result = [model.coeff_const.data[0]]
        ret_result.append(sum(pred_local==gold_local)/len(gold_local))
        # coeff_const = [1.82016313, 1.33751941, 1.41392815, 1.14783442, 1.22203231]
        # model.coeff_const.data[0] = float(coeff_const[self.fold-1])
        # print('load coeff_const', model.coeff_const.data[0])

        # inference at prediction time (train)
        unknown_labels, known_labels, all_labels, neighbor_pos_candis, neighbor_neg_candis, author_candis = [], {}, {}, [], [], []
        batch_sizes = {'label': [0], 'author': [0], 'neighbor_pos': [0], 'neighbor_neg': [0]}
        for idx, node_id in enumerate(self.train_nodes): #  
            unknown_labels.append(node_id)
            all_labels[node_id] = self.data.nodes_infor[node_id]['node_labels'][0]
            neighbor_pos_size = 0
            neighbor_neg_size = 0
            for neigh_id, rebuttal in self.data.nodes_infor[node_id]['node_connected']:
                if (rebuttal == 1):
                    neighbor_pos_size += 1
                    neighbor_pos_candis.append(neigh_id)
                else:
                    neighbor_neg_size += 1
                    neighbor_neg_candis.append(neigh_id)
            batch_sizes['neighbor_pos'].append(batch_sizes['neighbor_pos'][-1]+neighbor_pos_size)
            batch_sizes['neighbor_neg'].append(batch_sizes['neighbor_neg'][-1]+neighbor_neg_size)
            
            author_candis.extend(self.data.nodes_infor[node_id]['node_author'])
            author_size = len(self.data.nodes_infor[node_id]['node_author'])
            batch_sizes['author'].append(batch_sizes['author'][-1]+author_size)
            
            label_size = len(self.data.nodes_infor[node_id]['node_labels'])
            batch_sizes['label'].append(batch_sizes['label'][-1]+label_size)
        
        unknown_labels = np.asarray(unknown_labels)
        neighbor_pos_candis = np.reshape(neighbor_pos_candis, (-1, 1))
        neighbor_neg_candis = np.reshape(neighbor_neg_candis, (-1, 1))
        author_candis = np.reshape(author_candis, (-1, 1))
        
        my_inferencer = ILPInferencer(unknown_labels, known_labels, all_labels, neighbor_pos_candis, 
            neighbor_neg_candis, author_candis, batch_sizes,
            model, [])
        my_inferencer.add_variables_constraints()
        my_inferencer.optimize()
        result = my_inferencer.get_result()[0]
        # print('got result')
        pred_infer = []
        gold_infer = []
        for node_id in self.train_nodes: # 
            pred_infer.append(result[node_id]['label'][0])
            gold_infer.append(self.data.nodes_infor[node_id]['node_labels'][0])
        pred_infer = np.asarray(pred_infer)
        gold_infer = np.asarray(gold_infer)
        # print(sum(pred_infer==gold_infer), sum(pred_infer==gold_infer)/len(pred_infer))
        # print(f1_score(gold_infer, pred_infer, pos_label=None, average='micro'), 
        #      f1_score(gold_infer, pred_infer, pos_label=None, average='macro'))
        ret_result.append(sum(pred_infer==gold_infer)/len(gold_infer))

        if (self.retrain and self.train_method == 'local'): #  
            coeffX = np.reshape(my_inferencer.pos_list + my_inferencer.neg_list, (-1, 1))
            coeffy = len(my_inferencer.pos_list) * [1] + len(my_inferencer.neg_list) * [0]
            from sklearn import tree
            clf = tree.DecisionTreeClassifier(max_depth=1)
            clf.fit(coeffX, coeffy)
            model.coeff_sim.data[0] = clf.tree_.threshold[0]
            print('change coeff sim to %f' % model.coeff_sim.data[0])
            torch.save(model.state_dict(), self.save_model_path)
        
        # Local Result (test)
        batch_input, batch_gold, batch_negs, batch_sizes, gold_label, neighbor_labels, infer_data = \
            self.generate_batch_create(0, self.test_size, data='test')
        probs_var, pred_label_var, loss = model.predict(batch_input, batch_gold, batch_negs)
        probs = to_value(probs_var)
        pred_label = to_value(pred_label_var)
        del probs_var, pred_label_var, loss
        # print('test', sum(pred_label==gold_label), probs[0])
        
        pred_local = pred_label+self.data.label_list_4pred[0] # np.concatenate((pred_label, pred_label_test))
        gold_local = gold_label # np.concatenate((gold_label, gold_label_test))
        # print(sum(pred_local==gold_local), sum(pred_local==gold_local)/len(pred_local), sum(0==gold_local))
        # print(f1_score(gold_local, pred_local, pos_label=None, average='micro'), 
        #       f1_score(gold_local, pred_local, pos_label=None, average='macro'))
        
        # inference at prediction time (test)
        unknown_labels, known_labels, all_labels, neighbor_pos_candis, neighbor_neg_candis, author_candis = [], {}, {}, [], [], []
        batch_sizes = {'label': [0], 'author': [0], 'neighbor_pos': [0], 'neighbor_neg': [0]}
        for idx, node_id in enumerate(self.test_nodes): #  
            unknown_labels.append(node_id)
            all_labels[node_id] = self.data.nodes_infor[node_id]['node_labels'][0]
            neighbor_pos_size = 0
            neighbor_neg_size = 0
            for neigh_id, rebuttal in self.data.nodes_infor[node_id]['node_connected']:
                if (rebuttal == 1):
                    neighbor_pos_size += 1
                    neighbor_pos_candis.append(neigh_id)
                else:
                    neighbor_neg_size += 1
                    neighbor_neg_candis.append(neigh_id)
            batch_sizes['neighbor_pos'].append(batch_sizes['neighbor_pos'][-1]+neighbor_pos_size)
            batch_sizes['neighbor_neg'].append(batch_sizes['neighbor_neg'][-1]+neighbor_neg_size)
            
            author_candis.extend(self.data.nodes_infor[node_id]['node_author'])
            author_size = len(self.data.nodes_infor[node_id]['node_author'])
            batch_sizes['author'].append(batch_sizes['author'][-1]+author_size)
            
            label_size = len(self.data.nodes_infor[node_id]['node_labels'])
            batch_sizes['label'].append(batch_sizes['label'][-1]+label_size)
        # for node_id in self.train_nodes:
        #     known_labels[node_id] = self.data.nodes_infor[node_id]['node_labels'][0]
        #     all_labels[node_id] = known_labels[node_id]

        unknown_labels = np.asarray(unknown_labels)
        neighbor_pos_candis = np.reshape(neighbor_pos_candis, (-1, 1))
        neighbor_neg_candis = np.reshape(neighbor_neg_candis, (-1, 1))
        author_candis = np.reshape(author_candis, (-1, 1))
        
        my_inferencer = ILPInferencer(unknown_labels, known_labels, all_labels, neighbor_pos_candis, 
            neighbor_neg_candis, author_candis, batch_sizes,
            model, [], fold=self.fold)
        my_inferencer.add_variables_constraints()
        my_inferencer.optimize()
        result = my_inferencer.get_result()[0]
        
        # print('got result')
        pred_infer = []
        gold_infer = []
        for node_id in self.test_nodes: # 
            pred_infer.append(result[node_id]['label'][0])
            gold_infer.append(self.data.nodes_infor[node_id]['node_labels'][0])
        pred_infer = np.asarray(pred_infer)
        gold_infer = np.asarray(gold_infer)
        # print(pred_infer[:10], gold_infer[:10])
        # print(sum(pred_infer==gold_infer), sum(pred_infer==gold_infer)/len(pred_infer))
        # print(f1_score(gold_infer, pred_infer, pos_label=None, average='micro'), 
        #       f1_score(gold_infer, pred_infer, pos_label=None, average='macro'))

        major_vote = sum(self.data.label_list_4pred[0]==gold_local)/len(gold_local)
        if (major_vote < 0.5):
            major_vote = 1 - major_vote
        ret_result.extend([major_vote, sum(pred_local==gold_local)/len(pred_local), sum(pred_infer==gold_infer)/len(pred_infer)])
        return (best_node_embeds, best_author_embeds, best_label_embeds, 
            (to_value(model.node_fc1.weight), to_value(model.node_fc1.bias)), ret_result)
