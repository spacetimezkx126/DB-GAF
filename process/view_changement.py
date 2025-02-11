import json
import os
import re
import ast
import random
import numpy as np
import matplotlib.pyplot as plt
def MAP(ranking_results, rels):
    if type(list(ranking_results.values())[0]) == dict:
        ranking_results = {k: v.keys() for k, v in ranking_results.items()}
    Mean_Average_Precision = 0
    count =0
    for query_id in ranking_results.keys():
        if all([rels[query_id][did] != 3 for did in rels[query_id].keys()]):
            golden_labels = [2,3]
        else:
            golden_labels = [3]
        num_rel = 0
        Average_Precision = 0
        for i, candidate_id in enumerate(ranking_results[query_id]):
            if candidate_id in rels[query_id].keys() and rels[query_id][candidate_id] in golden_labels:
                num_rel += 1
                Average_Precision += num_rel / (i + 1.0)
        if num_rel > 0:
            Average_Precision /= num_rel
        Mean_Average_Precision += Average_Precision
    Mean_Average_Precision /= (len(ranking_results.keys()))
    return Mean_Average_Precision
def load_file():
    label_dic = json.load(open("./labels/labelv2.json","r"))
    return label_dic

if __name__ == '__main__':
    exp_name = "db_contrast_graph"
    log_path = "./log/" + exp_name
    result_path = "./experiment/" + exp_name
    label_dic = load_file()
    keys = label_dic.keys()
    epoch_regex = "epoch: (\d+)"
    count = 0
    epoch = 0
    times = 0
    count_result = 0
    count_all = 0
    data_all = {}
    with open(log_path,"r",encoding='utf-8')as f:
        datas = f.readlines()
        f.close()
    for data in datas:
        if re.search(epoch_regex,data) is not None:
            epoch = re.search(epoch_regex,data).group(1)
        
        if data.startswith("{("):
            chosen = {}
            not_chosen = {}
            crime_attribute = {}
            res = json.loads(data.replace("tensor(","").replace(".])","").replace("),",",").replace(".,",",").replace(", device='cuda:0', grad_fn=<DivBackward0>)","").replace("(",'"(').replace(")",')"').replace('"(查获)"',"").replace('"(边)"',""))
            for key in res: 
                converted_tuple = ast.literal_eval(key)
                if converted_tuple[0] not in crime_attribute:
                    crime_attribute[converted_tuple[0]]=[]
                crime_attribute[converted_tuple[0]].append([converted_tuple[1],res[key][0],res[key][1]])
            for k in crime_attribute.keys():
                crime_attribute[k].sort(key=lambda x: -x[1])
                chosen[k] = []
                not_chosen[k] = []
                for values in crime_attribute[k]:
                    chosen[k].append([values[0],values[1]])

            if (times,epoch) not in data_all:
                data_all[(times,epoch)] = []
            if count_result<6:
                data_all[(times,epoch)].append(chosen)
            
            # We can compare with the inherent alpha of GATv2.
            count_result +=1
            count_all+=1
            if count_result == 6:
                count_result = 0
            
            if count_all%6==0 and epoch=="9":
                times+=1
    

    x = []
    y1 = []
    y2 = []

    for key1 in data_all:
        list1 = data_all[key1]
        json_all = [data for data in list1]
        idx = 0
        n = 0
        count1 = 0
        count2 = 0
        all_change = {}
        for json1 in json_all:
            value_count = {}
            for key in json1:
                all_res = {}
                old_all_res = {}
                for data in json1[key]:
                    
                    if len(data[0].split("_"))==4:
                        import1 = data[0].split("_")[-2]
                        attr = data[0].split("_")[-1]
                        crime = data[0].split("_")[-3]
                        if crime not in all_res:
                            all_res[crime] = {}
                            all_res[crime]["important"] = {}
                            all_res[crime]["unimportant"] = {}
                        all_res[crime][import1][attr] = float(data[1])
                    if len(data[0].split("_"))==5:
                        import1 = data[0].split("_")[-3]
                        attr = data[0].split("_")[-1]
                        crime = data[0].split("_")[-4]
                        if crime not in all_res:
                            all_res[crime] = {}
                            all_res[crime]["important"] = {}
                            all_res[crime]["unimportant"] = {}
                        all_res[crime][import1][attr] = float(data[1])
                for crime in all_res:
                    for attr in all_res[crime]["important"]:
                        if (crime,attr) not in old_all_res:
                            old_all_res[(crime,attr)] = {"important":"","unimportant":"","value":""}
                        if crime not in all_change:
                            all_change[crime] = {}
                        if attr not in all_change[crime]:
                            all_change[crime][attr] = {"change":[]}

                        old_all_res[(crime,attr)]["value"] = all_res[crime]["important"][attr] - all_res[crime]["unimportant"][attr]
                        old_all_res[(crime,attr)]["important"] = all_res[crime]["important"][attr]

                        old_all_res[(crime,attr)]["unimportant"] = all_res[crime]["unimportant"][attr]
                        all_change[crime][attr]["change"].append(old_all_res[(crime,attr)]["value"])
                        value_count[all_res[crime]["important"][attr] - all_res[crime]["unimportant"][attr]]=value_count[all_res[crime]["important"][attr] - all_res[crime]["unimportant"][attr]]+1 if all_res[crime]["important"][attr] - all_res[crime]["unimportant"][attr] in value_count else 1    
                    idx+=1  
            n+=1
            count1+=sum([value_count[key] for key in value_count if abs(key)>1])
            count2+=sum([value_count[key] for key in value_count if abs(key)==0])
        change_all ={}
        sum_all = 0
        for crime in all_change:
            change_all[crime] = {}
            for attr in all_change[crime]:
                abs1 = [abs(s) for s in all_change[crime][attr]["change"]]
                rela1 = [s for s in all_change[crime][attr]["change"]]
                change_all[crime][attr]={"abs":0,"rela":0}
                change_all[crime][attr]["abs"] = sum(abs1)/len(abs1)
                sum_all+= sum(abs1)
                change_all[crime][attr]["rela"] = sum(all_change[crime][attr]["change"])/len(all_change[crime][attr]["change"])
        
        print(key1,sum_all)
        with open(result_path+"/times"+str(key1[0])+"/"+key1[1]+"result.json", "r")as f:
            data_json = json.load(f)
            f.close()
        
        map1 = {}
        index = 0
        for key in data_json:
            map1[key] = []
            del_val = []
            for vals in data_json[key]:
                if key in label_dic and vals not in label_dic[key]:
                    del_val.append(vals)
                map1[key].append(int(vals))

            for vals in del_val:
                data_json[key].pop(vals)
            index += 1

        # map_list = []
        dics = [map1]
        best_case = {}
        for dic in dics:
            smap = 0.0
            count = 0
            for key in keys:
                if key in dic:
                    values = [value for key1, value in label_dic[key].items()]
                    count+=1
                    ranks = [i for i in dic[key] if (str(i) in label_dic[key])] 
                    rels = [ranks.index(i) for i in ranks if label_dic[key][str(i)]==3]
                    tem_map = 0.0
                    for rel_rank in rels:
                        tem_map += float(len([j for j in ranks[:rel_rank+1] if label_dic[key][str(j)] ==3])/(rel_rank+1))
                    if len(rels) > 0:
                        smap += tem_map / len(rels)
                        best_case[key] = tem_map/len(rels)
            # map_list.append(MAP(data_json,label_dic))
        x.append(int(key1[1]))
        y1.append(sum_all)
        y2.append(MAP(data_json,label_dic))
        if len(x)==10:
            small_values = np.array(y2) 
            large_values = np.array(y1)  

            small_min, small_max = small_values.min(), small_values.max()
            large_min, large_max = large_values.min(), large_values.max()

            scaled_values = (small_values - small_min) / (small_max - small_min) * (large_max - large_min) + large_min
            plt.figure(figsize=(10, 10))
            plt.plot(x[:], y1[:], marker='o', linestyle='-', color='b', label='instability degree')
            plt.plot(x[:], scaled_values[:], marker='s', linestyle='--', color='g', label='MAP')
            plt.title('Change Tendency', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Sum of Difference', fontsize=12)
            plt.legend(loc='upper right',fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig("./images/"+str(key1[0])+"_comparison.png")
            plt.clf()
            x = []
            y1 = []
            y2 = []