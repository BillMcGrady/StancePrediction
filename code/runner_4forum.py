from DataLoader4Forum import *
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score
from gensim.models.doc2vec import Doc2Vec
from word2vec import Word2Vec
import argparse
import time

def evaluate(node_emb, label_emb=None, text_emb=None):
    trainX, trainy = my_data_loader.process_features(node_emb, my_data_loader.training_nodes)
    # validX, validy = my_data_loader.process_features(node_emb, my_data_loader.valid_nodes)
    testX, testy = my_data_loader.process_features(node_emb, my_data_loader.test_nodes)
    clf = LinearSVC()
    clf.fit(trainX, trainy) 
    # valid_pred = clf.predict(validX)
    test_pred = clf.predict(testX)
    
    # print(sum(valid_pred == validy))
    print(sum(test_pred == testy))

    gold_all = testy # np.concatenate((validy, testy))
    pred_all = test_pred # np.concatenate((valid_pred, test_pred))
    print(f1_score(gold_all, pred_all, pos_label=None, average='micro'))
    print(f1_score(gold_all, pred_all, pos_label=None, average='macro'))

def trainWord2Vec(doc_list=None, buildvoc=1, passes=10, sg=1, size=100,
              dm_mean=0, window=5, hs=0, negative=5, min_count=1, workers=1):
    model = Word2Vec(size=size,  sg=sg,  window=window,
                     hs=hs, negative=negative, min_count=min_count, workers=workers, compute_loss=True)

    if buildvoc == 1:
        print('Building Vocabulary')
        model.build_vocab(doc_list)  # build vocabulate with words + nodeID

    for epoch in range(passes):
        print('Iteration %d ....' % epoch)
        # shuffle(doc_list)  # shuffling gets best results

        model.train(doc_list, total_examples=len(doc_list), epochs=1)
        print(model.running_training_loss)

    print(model.sg, model.window, model.hs, model.min_count)
    print('batch words', model.batch_words)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Run Stance Prediction.")
    parser.add_argument('--data_path', nargs='?', default='../UNC/',
                        help='Input data path')
    parser.add_argument('--model', type=str, default='torch',
                        help='Model to use')
    # parser.add_argument('--python', type=str, default=3,
    #                     help='python version to use, 2 or 3')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # print(args.model)
    all_result = []

    all_same, all_diff = [], []
    for fold in range(1, 6):
        start = time.time()

        my_data_loader = DataLoader('data/4forum/', 2017, fold)
        
        # using node2vec
        # evaluate(my_data_loader.node_emb)

        if (args.model == 'gensim'):
            print('using gensim')
            # using deepwalk
            model = trainWord2Vec(my_data_loader.raw_walks)
            new_emb = {}
            for word in model.wv.vocab.keys():
                new_emb[my_data_loader.node_map[word]] = model[word]
            evaluate(new_emb)

        else:
            print('using our torch code')
            from NetworkEmbeddingTorch4forum import *
            my_network_embedding = NetworkEmbedding(my_data_loader, 'result/4forum/', fold)
            print(my_network_embedding)
            try:
                my_network_embedding.train()
            except KeyboardInterrupt:
                print('Interrupt')
                print('Exiting from training early')
            node_emb, author_emb, label_emb, node_fc1, ret_result = my_network_embedding.test()
            all_result.append(ret_result)
            
            # this_same, this_diff = my_network_embedding.test()
            # all_same.extend(this_same)
            # all_diff.extend(this_diff)
            # evaluate(node_emb)
            # my_data_loader.eval(node_emb, label_emb, author_emb, node_fc1)
        # exit()
        end = time.time()
        print('total time: %f' % (end - start))
        print(all_result[-1])

    all_result = np.asarray(all_result)
    print('coeff_const, train_acc, train_infer_acc, test_majority, test_acc, test_infer_acc')
    print(all_result, np.mean(all_result, axis=0))
    
    # print(len(all_same), len(all_diff), sum(all_same)/len(all_same), sum(all_diff)/len(all_diff))
    # plt.clf() 
    # distplot(all_same, color='b')
    # distplot(all_diff, color='g')
    # plt.savefig('result/global.pdf')
    # plt.show()    