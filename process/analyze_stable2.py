import os
import json
path1 = "./coliee_train1"
path2 = "./coliee_val1"
with open(path1,"r",encoding='utf-8')as f:
    datas = f.readlines()
    f.close()
with open(path2,"r",encoding='utf-8')as f:
    datas1 = f.readlines()
    f.close()
json_all = [json.loads(data.replace("\n","").replace("'",'"').strip()) for data in datas+datas1]
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
            if "unimportant" in data[0]:
                import1 = "unimportant"
            else:
                import1 = "important"
            import_ind = data[0].find(import1)
            attr = data[0][import_ind+len(import1):][1:]
            crime = data[0][4:import_ind][:-1]
            if import_ind!=-1:
                if crime not in all_res:
                    all_res[crime] = {}
                    all_res[crime]["important"] = {}
                    all_res[crime]["unimportant"] = {}
                all_res[crime][import1][attr] = float(data[1])
        for crime in all_res:
            for attr in all_res[crime]["important"]:
                if (crime,attr) not in old_all_res:
                    old_all_res[(crime,attr)] = {"important":"","unimportant":"","value":""}
                # if (crime,attr) not in all_change:
                #     all_change[(crime,attr)] = {"change":[]}
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
print(count1,count2)
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

for crime in change_all:
    attributes = change_all[crime].items()

    # 按照abs值从高到低排序
    sorted_attributes = sorted(attributes, key=lambda x: x[1]['abs'], reverse=True)
    change_all[crime] = sorted_attributes
with open("im_coliee_new1.json","w",encoding='utf-8')as f:
    json.dump(change_all,f,ensure_ascii=False, indent=4)
print(sum_all)