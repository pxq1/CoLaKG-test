import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['lastfm', 'ml-1m', 'mind', 'fund']
all_models  = ['mf', 'lgn', 'colakg']

config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['use_drop_edge'] = args.use_drop_edge
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['neighbor_k'] = args.neighbor_k
config['dropout_i'] = args.dropout_i
config['dropout_u'] = args.dropout_u
config['dropout_n'] = args.dropout_n
config['fusion_gate'] = args.fusion_gate
config['gate_type'] = args.gate_type
config['prop_norm'] = args.prop_norm
config['graph_gamma'] = args.graph_gamma
config['use_social'] = args.use_social
config['social_alpha'] = args.social_alpha
config['semantic_score_alpha'] = args.semantic_score_alpha
config['raw_semantic_score_alpha'] = args.raw_semantic_score_alpha
config['semantic_cl_weight'] = args.semantic_cl_weight
config['semantic_cl_tau'] = args.semantic_cl_tau
config['pop_score_alpha'] = args.pop_score_alpha
config['neighbor_score_alpha'] = args.neighbor_score_alpha
config['neighbor_score_steps'] = args.neighbor_score_steps
config['neighbor_score_mutual'] = args.neighbor_score_mutual
config['neighbor_gate_type'] = args.neighbor_gate_type
config['neighbor_gate_beta'] = args.neighbor_gate_beta
config['neighbor_gate_center'] = args.neighbor_gate_center
config['neighbor_gate_min'] = args.neighbor_gate_min
config['neighbor_score_norm'] = args.neighbor_score_norm
config['neighbor_rrf_k'] = args.neighbor_rrf_k
config['neighbor_train_alpha'] = args.neighbor_train_alpha
config['cf_score_alpha'] = args.cf_score_alpha
config['cf_neighbor_k'] = args.cf_neighbor_k
config['simgcl_weight'] = args.simgcl_weight
config['simgcl_tau'] = args.simgcl_tau
config['simgcl_eps'] = args.simgcl_eps
config['simgcl_start_epoch'] = args.simgcl_start_epoch
config['simgcl_stop_epoch'] = args.simgcl_stop_epoch
config['hard_neg_k'] = args.hard_neg_k
config['hard_neg_start_epoch'] = args.hard_neg_start_epoch
config['hard_neg_stop_epoch'] = args.hard_neg_stop_epoch
config['loss_type'] = args.loss_type
config['softmax_weight'] = args.softmax_weight
config['softmax_tau'] = args.softmax_tau
config['softmax_mask_pos'] = args.softmax_mask_pos
config['softmax_label_smoothing'] = args.softmax_label_smoothing
config['softmax_start_epoch'] = args.softmax_start_epoch
config['softmax_stop_epoch'] = args.softmax_stop_epoch


GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

item_semantic_emb_file = args.item_semantic_emb_file
user_semantic_emb_file = args.user_semantic_emb_file


TRAIN_epochs = args.epochs
LOAD = args.load
EVAL_ONLY = bool(args.eval_only)
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")
