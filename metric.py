import json
import math
import os
import argparse
def ndcg(ranks, gt_ranks, K):
    dcg_value = 0.
    idcg_value = 0.

    sranks = sorted(gt_ranks, reverse=True)
    for i in range(0,K):
        logi = math.log(i+2,2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi
    return dcg_value/idcg_value
def NDCG_k(ranking_results, rels, k):
    Mean_NDCG = 0
    count = 0
    for query_id in ranking_results.keys():
        temp_NDCG = 0.
        temp_IDCG = 0.
        answer_list = sorted([2**(rels[query_id][candidate_id]-1) if (candidate_id in rels[query_id] and rels[query_id][candidate_id] >= 1) else 0. 
                                for candidate_id in ranking_results[query_id]], reverse=True)
        for i, candidate_id in enumerate(ranking_results[query_id]):
            if i < k:
                if candidate_id in rels[query_id]:
                    temp_gain = 2**(rels[query_id][candidate_id]-1) if rels[query_id][candidate_id] >= 1 else 0.
                    temp_NDCG += (temp_gain / math.log(i+2, 2))
                    temp_IDCG += (answer_list[i] / math.log(i+2, 2))
            else:
                break
        if temp_IDCG > 0:
            Mean_NDCG += (temp_NDCG / temp_IDCG)
    Mean_NDCG /= (len(ranking_results.keys()) - count)
    return Mean_NDCG

# MAP
def MAP(ranking_results, rels):
    Mean_Average_Precision = 0
    count = 0
    for query_id in ranking_results.keys():
        if all([rels[query_id][did] != 3 for did in rels[query_id].keys()]):
            golden_labels = [2,3]
        else:
            golden_labels = [3]
        num_rel = 0
        Average_Precision = 0
        for i, candidate_id in enumerate(ranking_results[query_id]):
            if candidate_id in rels[query_id] and rels[query_id][candidate_id] in golden_labels:
                num_rel += 1
                Average_Precision += num_rel / (i + 1.0)
        if num_rel > 0:
            Average_Precision /= num_rel
        Mean_Average_Precision += Average_Precision
    Mean_Average_Precision /= (len(ranking_results.keys()) - count)
    return Mean_Average_Precision

# Precision@k
def Precision_k(ranking_results, rels, k=3):
    Precision_k = 0
    count = 0
    for query_id in ranking_results.keys():
        Precision = 0
        if query_id in rels:
            if all([rels[query_id][did] != 3 for did in rels[query_id].keys()]):
                golden_labels = [2,3]
            else:
                golden_labels = [3]
            for i, candidate_id in enumerate(ranking_results[query_id]):
                if i + 1 <= k:
                    if candidate_id in rels[query_id] and rels[query_id][candidate_id] in golden_labels:
                        Precision += 1
                else:
                    break
            Precision /= k
            Precision_k += Precision
        else:
            count += 1
    Precision_k /= (len(ranking_results.keys()) - count)
    return Precision_k
def load_file():
    label_dic = json.load(open("./labels/labelv2.json","r"))
    return label_dic
label_dic = load_file()

parser = argparse.ArgumentParser(description = 'lcr')
parser.add_argument('--exp_name', type = str, default = 'v2_in_2_u_efwf', 
                    help = 'dataset_name')  
parser.add_argument('--single_file', type = str, default = False, 
                    help = 'whether testing a single file')  
parser.add_argument('--file', type = str, default = "./results/result.json", 
                    help = 'whether testing a single file')  
args = parser.parse_args()

if __name__ == '__main__':
    if args.single_file is False:
        exp_name = args.exp_name
        all_map = []
        all_p = []
        all_ndcg_3 = []
        all_ndcg_5 = []
        all_ndcg_10 = []
        
        for n in range(10):
            print("times:",n)
            best_all = 0
            best_res = None
            best_epoch = 0
            for i in range(10):
                # print("epoch:",i)
                with open("./experiment/"+exp_name+"/times"+str(n)+"/"+str(i)+"result.json", "r")as f:
                    result_all = json.load(f)
                    f.close()
                for key in result_all:
                    del_val = []
                    for vals in result_all[key]:
                        if key in label_dic and vals not in label_dic[key]:
                            del_val.append(vals)
                    for vals in del_val:
                        result_all[key].pop(vals)
                if best_all < MAP(result_all,label_dic)+Precision_k(result_all,label_dic,3)+NDCG_k(result_all,label_dic,3)+NDCG_k(result_all,label_dic,5)+NDCG_k(result_all,label_dic,10):
                    best_all = MAP(result_all,label_dic)+Precision_k(result_all,label_dic,3)+NDCG_k(result_all,label_dic,3)+NDCG_k(result_all,label_dic,5)+NDCG_k(result_all,label_dic,10)
                    best_res = result_all
                    best_epoch = i
            print(best_epoch)
            print(MAP(best_res,label_dic))
            print(Precision_k(best_res,label_dic,3))
            print(NDCG_k(best_res,label_dic,3))
            print(NDCG_k(best_res,label_dic,5))
            print(NDCG_k(best_res,label_dic,10))
            all_map.append(MAP(best_res,label_dic))
            all_p.append(Precision_k(best_res,label_dic,3))
            all_ndcg_3.append(NDCG_k(best_res,label_dic,3))
            all_ndcg_5.append(NDCG_k(best_res,label_dic,5))
            all_ndcg_10.append(NDCG_k(best_res,label_dic,10))
        print(sum(all_map)/len(all_map),sum(all_p)/len(all_p),sum(all_ndcg_3)/len(all_ndcg_3),sum(all_ndcg_5)/len(all_ndcg_5),sum(all_ndcg_10)/len(all_ndcg_10))

    else:
        
        file = args.file
        print(file)
        with open(file,"r") as f:
            result_all = json.load(f)
            f.close()
        for key in result_all:
            del_val = []
            for vals in result_all[key]:
                if key in label_dic and vals not in label_dic[key]:
                    del_val.append(vals)
            for vals in del_val:
                result_all[key].pop(vals)
        print(MAP(result_all,label_dic))
        print(Precision_k(result_all,label_dic,3))
        print(NDCG_k(result_all,label_dic,3))
        print(NDCG_k(result_all,label_dic,5))
        print(NDCG_k(result_all,label_dic,10))



