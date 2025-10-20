import json
import os
path1 = "/root/autodl-tmp/formal/DB-GAF/process/im_coliee_new.json"
# 找到最优的划分点
content = "./../dataset/coliee"

with open(path1,"r",encoding='utf-8')as f:
    data_json = json.load(f)
    f.close()

files = os.listdir(content)

count_res = {}
for file in files:
    path2 = os.path.join(content,file)
    with open(path2,"r",encoding='utf-8')as f:
        data_json3 = json.load(f)
        f.close()
    for case in data_json3:
        for crime in case["Crimes"]:
            # print(crime)
            if type(crime["Crime_Type"])==list:

                crime_name = crime["Crime_Type"][0]
            else:
                crime_name = crime["Crime_Type"]
            # print(crime_name)
            if crime_name not in count_res:
                count_res[crime_name] = {}
            for item in crime:
                # if crime_name in choose1:
                if (type(crime[item])==str or type(crime[item])==list) and len(crime[item])!=0:
                    if type(crime[item])==str:
                        count_res[crime_name][item]=count_res[crime_name][item]+1 if item in count_res[crime_name] else 1
                    else:
                        count_res[crime_name][item]=count_res[crime_name][item]+len(crime[item]) if item in count_res[crime_name] else len(crime[item])
low_res = {}
high_res = {}
for crime in data_json:
    group1 = data_json[crime]
    # print(group1)
    high_res[crime] = []
    low_res[crime] = []
    all_value = [b[1]['abs'] for b in group1]
    all_value.sort()
    print(all_value)
    high_3 = all_value[2*len(all_value)//3]
    low_3 = all_value[len(all_value)//3]
    print(high_3,low_3)
    for b in group1:
        if b[1]['abs']>=high_3 or b[1]['abs']<=low_3:
            # b[1]['label'] = 1
            high_res[crime].append(b[0])
        # if b[1]['abs']<=low_3:
        #     low_res[crime].append(b[0])

with open("high_low_res.json","w",encoding='utf-8')as f:
    json.dump(high_res,f,ensure_ascii=False, indent=4)
# with open("low_res.json","w",encoding='utf-8')as f:
#     json.dump(low_res,f,ensure_ascii=False, indent=4)