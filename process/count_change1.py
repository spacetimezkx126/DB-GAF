import os
import json
import ast
import re
path ="/root/autodl-tmp/formal/DB-GAF/log1"
result_path = "/root/autodl-tmp/formal/DB-GAF/experiment/db_contrast_graph_coliee"
with open(path,"r",encoding='utf-8')as f:
    datas = f.readlines()
    f.close()
i = 0
best_map = {'1': ['55', '2']}
epoch_regex = "epoch: (\d+)"
fold_regex = "Fold (\d+)"
count = 0
epoch = 0
times = 0
count_result = 0
count_all = 0
print_count = 0
fold = 0
for data in datas:
    if re.search(epoch_regex,data) is not None:
        epoch = re.search(epoch_regex,data).group(1)
    if data.startswith("{("):
        count_all +=1
        chosen = {}
        not_chosen = {}
        crime_attribute = {}
        data = data.replace("'s","")
        test4 = json.loads(data.replace('"',"'").replace("tensor(","").replace(".])","").replace("),",",").replace(".,",",").replace(", device='cuda:0', grad_fn=<DivBackward0>)","").replace("(",'"(').replace(")",')"').replace('"(查获)"',"").replace('"(边)"',""))
        for key in test4: 
            converted_tuple = ast.literal_eval(key)
            if converted_tuple[0] not in crime_attribute:
                crime_attribute[converted_tuple[0]]=[]
            crime_attribute[converted_tuple[0]].append([converted_tuple[1],test4[key][0],test4[key][1]])
        # print()
        for k in crime_attribute.keys():
            crime_attribute[k].sort(key=lambda x: -x[1])
            chosen[k] = []
            not_chosen[k] = []
            for values in crime_attribute[k]:
                chosen[k].append([values[0],values[1]])
        i+=1
        count +=1
        if str(times) in best_map:
            if best_map[str(times)][0] == epoch:
                if count_result%2==1:
                    print(chosen)
                    print_count+=1
                count_result +=1
                if count_result == 6:
                    count_result = 0
        if epoch=="299" and count_all%6==0:
            times+=1

