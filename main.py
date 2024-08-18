import sys
import time
import argparse
import pickle
import os
import logging
import numpy as np
import pandas as pd
import torch
import tqdm
import json
from sklearn.model_selection import train_test_split
from model import *
from utils import *


current_dir = os.getcwd()
print(current_dir)
path = os.path.join(current_dir,'data',)
print(path)

logging.basicConfig(filename= current_dir + '/{}.log'.format(str(int(time.time()))),
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.info('*******start1***********')
logging.info('start time: %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
logging.info("文本内容 + 文本处理方式:{}".format('title_vecs.npy'))

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', default=path, help='path to the dataset')
parser.add_argument('--dataset', default='dblp', help='blockchain/dblp/acm') 
parser.add_argument('--model', default='Intent_UniGCNII', help='[Intent_UniGCNII, HIDE, GCN_rec, SASRec, GRU4Rec, CORE, SINE, LightGCN, KGCN]')
parser.add_argument('--gcn', action='store_true', default=True, help='if use gcn')
parser.add_argument('--hiddenSize', type=int, default = 448 ) # 100, 96, 768  text_vec   96，6*64=384
parser.add_argument('--hidden_channels', type=int, default = 256, help='Hidden number for GNN')  # 128, *256, 512, 1024
parser.add_argument('--n_factor', type=int, default=7, help='Disentangle factors number')   # 3,5,4,
parser.add_argument('--title_vec', type=int, default=384, help='journal vector dim') # 100,96, 768  text_vec
parser.add_argument('--journal_vec', type=int, default=384, help='title and abstract dim') # 100,96, 768  text_vec
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=4)
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--gpu_id', type=str,default="0")
parser.add_argument('--batch_size', type=int, default=512)  # 512
parser.add_argument('--lr', type=float, default=0.005, help='learning rate.')   # 0.001, 0.005,0.0005
parser.add_argument('--lr_dc', type=float, default=0.2, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=5, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-6, help='l2 penalty ') # 1e-6
parser.add_argument('--layer', type=int, default=1, help='the number of layer used')
parser.add_argument('--n_iter', type=int, default=1)    
parser.add_argument('--seed', type=int, default=2024)                                 # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.1, help='Dropout rate.')
parser.add_argument('--e', type=float, default=0.4, help='Disen H sparsity. select relative paper for intent')  # 0.4, dblp(0.3)  acm(0.2)
parser.add_argument('--journal', action='store_true', default=True, help='add journal vector')
parser.add_argument('--title', action='store_true', default=True, help='add title vector')
parser.add_argument('--disen', action='store_true', default=True, help='use disentangle')
parser.add_argument('--lamda', type=float, default=1e-5, help='aux loss weight') # 1e-4, 
parser.add_argument('--norm', action='store_true', help='use norm')
parser.add_argument('--sw_edge', action='store_true', help='slide_window_edge')
parser.add_argument('--item_edge', action='store_true', default=True, help='item_edge')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--first_aggregate', type=str, default='mean',help='the type of graph first aggregate')
parser.add_argument('--nhid', type=int, default= 2,help='number of hidden features, note that actually it\'s #nhid x #nhead')
parser.add_argument('--nhead', type=int, default=2,help='number of conv heads for transformer or MA')
parser.add_argument('--nlayer', type=int, default=2, help='number of hidden layers')
parser.add_argument('--activation', type=str, default='relu', help='activation layer between UniConvs')
parser.add_argument('--dropout', type=float, default=0, help='dropout probability after UniConv layer in citaton_graph')
parser.add_argument('--input-drop', type=float, default=0, help='dropout probability for input layer in citaton_graph')
parser.add_argument('--use-norm', action="store_true", help='use norm in the final layer')
parser.add_argument('--hidden_dim', type=int, default=512, help='transformer hidden dimension')  # *512, 1024
parser.add_argument('--integrate_model', default='transformer', help='[multi_head_att, transformer, mean], generate citing embedding,[HIDE, Intent_UniGCNII, GCN_rec]')
parser.add_argument('--lamda_1', type=float, default= 0.85, help='two model weight')
parser.add_argument('--co_show', action='store_true', default=False, help='use co_show')
parser.add_argument('--citation', action='store_true', default=True, help='use citaiton')
parser.add_argument('--GraphSAGE', action='store_true', default=False, help='use citaiton')
parser.add_argument('--max_seq_length', type=int, default=50, help='MAX_CIT_LEN')
parser.add_argument('--num_layers', type=int, default=1, help='The number of layers in GCN')

opt = parser.parse_args()


os.environ["CUDA_LAUNCH_BLOCKING"] = str(1) 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
os.environ['PYTHONHASHSEED'] = str(opt.seed)


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(TensorEncoder, self).default(obj)


def main():
    exp_seed = opt.seed
    top_K = [5, 10, 15, 20]
    init_seed(exp_seed)


    if opt.dataset == 'acm':
        num_node = 23133   
        opt.n_iter = 1
        opt.dropout_gcn = 0.4
        opt.dropout_local = 0.0

    elif opt.dataset == 'dblp':
        num_node = 14573   
        opt.n_iter = 1
        opt.dropout_gcn = 0.4
        opt.dropout_local = 0.0
        
    else:
        num_node = 13315 
        opt.n_iter = 1
        opt.dropout_gcn = 0.4
        opt.dropout_local = 0.0
     
    logging.info('>>SEED:{}'.format(exp_seed))
    logging.info('===========config================')
    logging.info("model:{}".format(opt.model))
    logging.info("if_gcn:{}".format(opt.gcn))
    logging.info("dataset:{}".format(opt.dataset))
    logging.info("gpu:{}".format(opt.gpu_id))
    logging.info("Disentangle:{}".format(opt.disen))
    logging.info("Intent factors:{}".format(opt.n_factor))
    logging.info("journal vec:{}".format(opt.journal))  
    logging.info("title vec:{}".format(opt.title))
    logging.info("Test Topks{}:".format(top_K))
    logging.info("hiddenSize:{}".format(opt.hiddenSize))
    logging.info("n_factor:{}".format(opt.n_factor))  
    logging.info("lr:{}".format(opt.lr))  
    logging.info("batch_size:{}".format(opt.batch_size))  
    logging.info("lamda_1:{}".format(opt.lamda_1))  
    logging.info("if co_show:{}".format(opt.co_show))  
    logging.info("if citation:{}".format(opt.citation))
    logging.info("integrate_model:{}".format(opt.integrate_model))  
    logging.info("select relative paper for intent:{}".format(opt.e))  
    logging.info("Hidden number for GNN:{}".format(opt.hidden_channels))  
    logging.info("aux loss weight:{}".format(opt.lamda))  



    logging.info('===========end===================')

    datapath = path 

    hypergraph_data = pd.read_csv(datapath + '/{}/hypergraph.csv'.format(opt.dataset))  # DBLP
    journal_vectors = np.load(opt.datapath + '/{}/journal_emb.npy'.format(opt.dataset), allow_pickle=True)
    title_vectors = np.load(opt.datapath + '/{}/t_a_emb.npy'.format(opt.dataset), allow_pickle=True)


    hypergraph_data['cited'] = hypergraph_data['cited'].apply(lambda x: list(map(int, x.split(','))))
    train_data, test_data = train_test_split(hypergraph_data, test_size=0.2, random_state=42)
    start1 = time.time()
    print(start1)

    node_features, H, H_content_tensor, MAX_CIT_LEN = load_graph(hypergraph_data, title_vectors, journal_vectors, num_node, opt)
        

    start2 = time.time()
    print(start1, start2-start1)
 
    if opt.model == 'Intent_UniGCNII':
        model = trans_to_cuda(Intent_UniGCNII(opt, num_node))
    elif opt.model == 'GCN_rec':
        model = trans_to_cuda(GCN_rec(opt, num_node))

    start = time.time()
    MAX_CIT_LEN = 50
    train_data = Data(train_data, MAX_CIT_LEN, opt, n_node=num_node)
    test_data = Data(test_data, MAX_CIT_LEN, opt, n_node=num_node)

    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    bad_counter = 0
    recomend_list = {}
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        logging.info(f'EPOCH:{epoch}')
        logging.info(f'Time:{time.strftime("%Y/%m/%d %H:%M:%S")}')
        total_loss, rec_loss, intent_loss, metrics = train_test(model, train_data, test_data, H, H_content_tensor, node_features, top_K, opt)
        # recomend_list[epoch] = top_k_rec
        # filename = current_dir + '/output/{}_{}.json'.format(opt.model, opt.dataset)
        # with open(filename, 'w') as f:
        #     json.dump(recomend_list, f, cls=TensorEncoder, indent=1)

        logging.info("loss: {}; rec_loss: {}; intent_loss: {} ".format(total_loss, rec_loss, intent_loss))
        # flag = 0
        for K in top_K:
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['recall%d' % K] = np.mean(metrics['recall%d' % K]) * 100

            if best_results['metric%d' % K][0] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][0] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][0] = epoch
                flag = 1
            if best_results['metric%d' % K][1] < metrics['recall%d' % K]:
                best_results['metric%d' % K][1] = metrics['recall%d' % K]
                best_results['epoch%d' % K][1] = epoch
                flag = 1

        for K in top_K:
            logging.info('Current Result:')
            logging.info('MRR%d: %.4f\tRecall%d: %.4f' %
                (K, metrics['mrr%d' % K], K, metrics['recall%d' % K]))
            logging.info('Best Result:')
            logging.info('\tMRR%d: %.4f\tRecall%d: %.4f\tEpoch: %d, %d' %
                (K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
            bad_counter += 1 - flag

        if bad_counter >= opt.patience:
            break

    
    logging.info('-------------------------------------------------------')
    end = time.time()
    logging.info("Run time: %f s" % (end - start))


def forward(model, data, H, H_content, node_features):

    citings, h_items, targets, u_seqs,  u_seqs_num = data
    citings = trans_to_cuda(citings).long()
    h_items = trans_to_cuda(h_items).long()      
    u_seqs = trans_to_cuda(u_seqs).long()     
    u_seqs_num = trans_to_cuda(u_seqs_num).long()        
    intent_score = model(citings, h_items, u_seqs,  u_seqs_num, H, H_content, node_features)

    
    return targets, intent_score


def train_test(model, train_data, test_data, H, H_content, node_features, top_K, opt):
    #print('start training: ', datetime.datetime.now())
    model.train()
    rec_loss, intent_loss, intentloss = 0.0, 0.0, 0.0
    total_loss, loss = 0.0, 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):   
        model.optimizer.zero_grad()
        targets, scores = forward(model, data, H, H_content, node_features)
        targets = trans_to_cuda(targets).long()
        recloss = model.loss_function(scores, targets)
        loss = recloss
        if opt.disen:
            intentloss = opt.lamda * model.intent_loss
            loss += intentloss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        rec_loss += recloss
        intent_loss += intentloss
    model.scheduler.step()

    metrics = {}
    for K in top_K:
        metrics['mrr%d' % K] = []
        metrics['recall%d' % K] = []


    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    rec_test_results = {}
    for index, data in tqdm(enumerate(test_loader)):
        batch_result = {}
        batch_result["source"] = data[0].cpu().detach().numpy().tolist()
        targets, scores = forward(model, data, H, H_content, node_features)
        targets = targets.numpy()  
        for K in top_K:
            sub_scores = scores.topk(K)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target in zip(sub_scores, targets):  
                if len(np.where(score == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(score == target)[0][0] + 1))
                recall = np.sum(np.isin(target, score)) / len(np.atleast_1d(target))
                metrics['recall%d' % K].append(recall)

        # score, result = scores.topk(20)[0].cpu().detach().numpy(),scores.topk(20)[1].cpu().detach().numpy()
        # batch_result["result_topk20_paper"] = result


        rec_test_results[index] = batch_result

        
        sample_num = targets.shape[0]
    rec_test_results = rec_test_results
    return total_loss, rec_loss, intent_loss, metrics #rec_test_results

if __name__ == '__main__':
    main()
