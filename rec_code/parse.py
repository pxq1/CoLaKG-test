
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go Model")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--neighbor_k', type=int,default=10,
                        help="the num of neighbors")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--use_drop_edge', type=int,default=1,
                        help="using the drop_edge or not for lightgcn")
    parser.add_argument('--keepprob', type=float,default=0.7,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--dropout_i', type=float,default=0.6,
                        help="the dropout for item semantic embeddings")
    parser.add_argument('--dropout_u', type=float,default=0.6,
                        help="the dropout for user semantic embeddings")
    parser.add_argument('--dropout_n', type=float,default=0.6,
                        help="the dropout for neighbor embeddings")
    parser.add_argument('--fusion_gate', type=int, default=0,
                        help="use learnable gates for id/semantic and neighbor fusion")
    parser.add_argument('--gate_type', type=str, default='scalar',
                        choices=['scalar', 'vector'],
                        help="gate parameterization for fusion_gate")
    parser.add_argument('--prop_norm', type=int, default=0,
                        help="L2-normalize embeddings before each graph propagation layer")
    parser.add_argument('--graph_gamma', type=float, default=-1.0,
                        help="if >= 0, use gamma * ego + (1 - gamma) * propagated layer aggregation")
    parser.add_argument('--use_social', type=int, default=0,
                        help="use available user-user social graph for user representation smoothing")
    parser.add_argument('--social_alpha', type=float, default=0.1,
                        help="residual weight for social user smoothing")
    parser.add_argument('--semantic_score_alpha', type=float, default=0.0,
                        help="weight of the semantic residual scoring head")
    parser.add_argument('--raw_semantic_score_alpha', type=float, default=0.0,
                        help="weight of fixed raw semantic cosine scores")
    parser.add_argument('--semantic_cl_weight', type=float, default=0.0,
                        help="weight of semantic collaborative contrastive alignment loss")
    parser.add_argument('--semantic_cl_tau', type=float, default=0.2,
                        help="temperature for semantic collaborative contrastive alignment")
    parser.add_argument('--pop_score_alpha', type=float, default=0.0,
                        help="weight of train-set item popularity prior in ranking scores")
    parser.add_argument('--neighbor_score_alpha', type=float, default=0.0,
                        help="weight of semantic item-neighbor score diffusion during ranking")
    parser.add_argument('--neighbor_score_steps', type=int, default=1,
                        help="number of semantic-neighbor diffusion steps for ranking scores")
    parser.add_argument('--simgcl_weight', type=float, default=0.0,
                        help="weight of SimGCL-style graph contrastive loss")
    parser.add_argument('--simgcl_tau', type=float, default=0.2,
                        help="temperature for SimGCL-style graph contrastive loss")
    parser.add_argument('--simgcl_eps', type=float, default=0.1,
                        help="embedding perturbation magnitude for SimGCL-style views")
    parser.add_argument('--simgcl_start_epoch', type=int, default=0,
                        help="first epoch to apply SimGCL-style graph contrastive loss")
    parser.add_argument('--simgcl_stop_epoch', type=int, default=-1,
                        help="disable SimGCL-style graph contrastive loss from this epoch; -1 keeps it enabled")
    parser.add_argument('--hard_neg_k', type=int, default=1,
                        help="number of sampled negative candidates for dynamic hard negative mining")
    parser.add_argument('--hard_neg_start_epoch', type=int, default=0,
                        help="first epoch to apply dynamic hard negative mining")
    parser.add_argument('--hard_neg_stop_epoch', type=int, default=-1,
                        help="disable dynamic hard negative mining from this epoch; -1 keeps it enabled")
    parser.add_argument('--loss_type', type=str, default='bpr',
                        choices=['bpr', 'softmax', 'bpr_softmax'],
                        help="training objective: BPR, full-item softmax, or their combination")
    parser.add_argument('--softmax_weight', type=float, default=1.0,
                        help="weight of the full-item softmax loss when loss_type is bpr_softmax")
    parser.add_argument('--softmax_tau', type=float, default=1.0,
                        help="temperature for full-item softmax logits")
    parser.add_argument('--softmax_mask_pos', type=int, default=1,
                        help="mask a user's other train positives from full-item softmax negatives")
    parser.add_argument('--softmax_label_smoothing', type=float, default=0.0,
                        help="label smoothing for full-item softmax loss")
    parser.add_argument('--softmax_start_epoch', type=int, default=0,
                        help="first epoch to apply full-item softmax loss")
    parser.add_argument('--softmax_stop_epoch', type=int, default=-1,
                        help="disable full-item softmax loss from this epoch; -1 keeps it enabled")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='ml-1m',
                        help="available datasets")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--item_semantic_emb_file', type=str,default=" ",
                        help="the path of item_semantic_emb_file")
    parser.add_argument('--user_semantic_emb_file', type=str,default=" ",
                        help="the path of user_semantic_emb_file")
    parser.add_argument('--topks', nargs='?',default="[10,20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--eval_only', type=int, default=0,
                        help="load weights and run one test pass without training or saving")
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='colakg', help='rec-model, support [mf, lgn, colakg]')
    return parser.parse_args()
