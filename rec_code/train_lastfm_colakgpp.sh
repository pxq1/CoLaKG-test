#!/bin/bash

decay=1e-4
lr=0.001
layer=3
seed=2020
dataset="lastfm"
topks="[10,20]"
epochs=1000
recdim=256
use_drop_edge=0
keepprob=1.0
batch_size=1024
dropout_i=0.4
dropout_u=0.2
dropout_n=0.4
neighbor_k=10
fusion_gate=0
gate_type="scalar"
prop_norm=0
graph_gamma=-1.0
use_social=0
social_alpha=0.1
semantic_score_alpha=0.0
semantic_cl_weight=0.0
semantic_cl_tau=0.2
pop_score_alpha=0.0
item_semantic_emb_file='../data/lastfm/lastfm_embeddings_simcse_kg.pt'
user_semantic_emb_file='../data/lastfm/lastfm_embeddings_simcse_kg_user.pt'

CUDA_VISIBLE_DEVICES=0 python main.py --epochs=$epochs --bpr_batch=$batch_size --decay=$decay --lr=$lr --layer=$layer --seed=$seed --dataset=$dataset --topks=$topks --recdim=$recdim --use_drop_edge=$use_drop_edge --keepprob=$keepprob --neighbor_k=$neighbor_k --dropout_i=$dropout_i --dropout_u=$dropout_u --dropout_n=$dropout_n --fusion_gate=$fusion_gate --gate_type=$gate_type --prop_norm=$prop_norm --graph_gamma=$graph_gamma --use_social=$use_social --social_alpha=$social_alpha --semantic_score_alpha=$semantic_score_alpha --semantic_cl_weight=$semantic_cl_weight --semantic_cl_tau=$semantic_cl_tau --pop_score_alpha=$pop_score_alpha --item_semantic_emb_file=$item_semantic_emb_file --user_semantic_emb_file=$user_semantic_emb_file
