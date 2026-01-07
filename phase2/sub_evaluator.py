#!/usr/bin/env python
import sys, os, os.path
import numpy as np
import json
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)
    

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def parse_line(l):
    impid, ranks = l.strip('\n').split()
    ranks = json.loads(ranks)
    return impid, ranks

def scoring(truth_f, sub_f):
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []
    
    line_index = 1
    for lt in tqdm(truth_f, desc='Scoring', unit='impr'):
        ls = sub_f.readline()
        impid, labels = parse_line(lt)
        
        # ignore masked impressions
        if labels == []:
            continue 
        
        if ls == '':
            # empty line: filled with 0 ranks
            sub_impid = impid
            sub_ranks = [1] * len(labels)
        else:
            try:
                sub_impid, sub_ranks = parse_line(ls)
            except:
                raise ValueError("line-{}: Invalid Input Format!".format(line_index))       
        
        if sub_impid != impid:
            raise ValueError("line-{}: Inconsistent Impression Id {} and {}".format(
                line_index,
                sub_impid,
                impid
            ))        
        
        lt_len = float(len(labels))
        
        y_true =  np.array(labels,dtype='float32')
        y_score = []
        for rank in sub_ranks:
            score_rslt = 1./rank
            if score_rslt < 0 or score_rslt > 1:
                raise ValueError("Line-{}: score_rslt should be int from 0 to {}".format(
                    line_index,
                    lt_len
                ))
            y_score.append(score_rslt)
        
        auc = roc_auc_score(y_true,y_score)
        mrr = mrr_score(y_true,y_score)
        ndcg5 = ndcg_score(y_true,y_score,5)
        ndcg10 = ndcg_score(y_true,y_score,10)
        
        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)
        
        line_index += 1

    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)
        

def prod_scoring(truth_f, sub_f):
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []
    
    line_index = 1
    for lt in tqdm(truth_f, desc='Scoring', unit='impr'):
        ls = sub_f.readline()
        impid, labels = parse_line(lt)
        
        # ignore masked impressions
        if labels == []:
            continue 
        
        if ls == '':
            # empty line: filled with 0 ranks
            sub_impid = impid
            sub_ranks = [1] * len(labels)
        else:
            try:
                sub_impid, sub_ranks = parse_line(ls)
            except:
                raise ValueError("line-{}: Invalid Input Format!".format(line_index))       
        
        if sub_impid != impid:
            raise ValueError("line-{}: Inconsistent Impression Id {} and {}".format(
                line_index,
                sub_impid,
                impid
            ))        
        
        lt_len = float(len(labels))
        
        y_true =  np.array(labels,dtype='float32')
        y_score = []
        for rank in sub_ranks:
            score_rslt = rank
            if score_rslt < 0 or score_rslt > 1:
                raise ValueError("Line-{}: score_rslt should be int from 0 to {}".format(
                    line_index,
                    lt_len
                ))
            y_score.append(score_rslt)
        
        auc = roc_auc_score(y_true,y_score)
        mrr = mrr_score(y_true,y_score)
        ndcg5 = ndcg_score(y_true,y_score,5)
        ndcg10 = ndcg_score(y_true,y_score,10)
        
        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)
        
        line_index += 1

    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate submission against truth')
    parser.add_argument('input', help='Input directory (old behavior) or ignored when using explicit files')
    parser.add_argument('output_dir', help='Directory to write scores.txt' ,default='output_dir')
    parser.add_argument('--prediction-file', '-p', help='Path to prediction file (overrides input/res/prediction.txt)')
    parser.add_argument('--truth-file', '-t', help='Path to truth file (overrides input/ref/truth.txt)')
    args = parser.parse_args()

    # Determine truth and prediction paths. Prefer explicit files if provided.
    if args.prediction_file:
        pred_path = args.prediction_file
    else:
        pred_path = os.path.join(args.input, 'res', 'prediction.txt')

    if args.truth_file:
        truth_path = args.truth_file
    else:
        truth_path = os.path.join(args.input, 'ref', 'truth.txt')

    # Fallback: if the constructed paths do not exist, try the original directory layout
    if not os.path.isfile(pred_path):
        fallback_pred = os.path.join(args.input, 'res', 'prediction.txt')
        if os.path.isfile(fallback_pred):
            pred_path = fallback_pred

    if not os.path.isfile(truth_path):
        fallback_truth = os.path.join(args.input, 'ref', 'truth.txt')
        if os.path.isfile(fallback_truth):
            truth_path = fallback_truth

    if not os.path.isfile(pred_path):
        print(f'Prediction file not found: {pred_path}')
        sys.exit(1)
    if not os.path.isfile(truth_path):
        print(f'Truth file not found: {truth_path}')
        sys.exit(1)

    # Ensure output directory exists
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.join(output_dir, 'scores.txt')
    with open(output_filename, 'w') as output_file, \
         open(truth_path, 'r', encoding='utf-8') as truth_file, \
         open(pred_path, 'r', encoding='utf-8') as submission_answer_file:

        auc, mrr, ndcg, ndcg10 = prod_scoring(truth_file, submission_answer_file)
        output_file.write("AUC:{:.4f}\nMRR:{:.4f}\nnDCG@5:{:.4f}\nnDCG@10:{:.4f}".format(auc, mrr, ndcg, ndcg10))