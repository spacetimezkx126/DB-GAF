import numpy as np
import json
import os
def calculate_overlap(group1_front, group2_front, group1_back, group2_back):
    # 计算前半部分和后半部分的重叠度
    overlap_front = 0
    overlap_back = 0
    
    # 前半部分重叠：根据属性名匹配
    for item1 in group1_front:
        for item2 in group2_front:
            if item1[0] == item2[0]:  # 以属性名作为匹配标准
                overlap_front += 1
                break  # 找到匹配项后跳出内循环
    
    # 后半部分重叠：根据属性名匹配
    for item1 in group1_back:
        for item2 in group2_back:
            if item1[0] == item2[0]:
                overlap_back += 1
                break  # 找到匹配项后跳出内循环
    
    return overlap_front, overlap_back
def calculate_count_diff(group1_front, group1_back, group2_front, group2_back, count):
    # 计算左右两部分分配的count差异
    count_front_group1 = sum(count.get(item[0], 0) for item in group1_front)
    count_back_group1 = sum(count.get(item[0], 0) for item in group1_back)
    count_front_group2 = sum(count.get(item[0], 0) for item in group2_front)
    count_back_group2 = sum(count.get(item[0], 0) for item in group2_back)
    
    diff_group1 = count_front_group1 - count_back_group1
    diff_group2 = count_front_group2 - count_back_group2
    
    
    return diff_group1,diff_group2

def find_best_split(group1, group2, count):
    len_group1 = len(group1)
    len_group2 = len(group2)
    
    max_overlap = -1
    best_split = (0, 0)
    
    overlap_left_right1 = 0
    overlap_middle1 = 0
    min_count_middle_group1 = sum(count.get(item[0], 0) for item in group1)
    
    # 尝试不同的划分点
    for i in range(1, min(len_group1, len_group2)):
        # 按照新的划分方式：左端和右端作为一个部分，中间部分作为另一个部分
        # 左端和右端部分
        group1_left_right = group1[:i] + group1[len_group1-i:]
        group2_left_right = group2[:i] + group2[len_group2-i:]
        
        # 中间部分
        group1_middle = group1[i:len_group1-i]
        group2_middle = group2[i:len_group2-i]
        

        group3_left_right_size = len(group1_left_right)
        
        diff_left_right_group1, diff_left_right_group2 = calculate_count_diff(group1_left_right, group1_middle, group2_left_right, group2_middle, count)
        overlap_left_right, overlap_middle = calculate_overlap(group1_left_right, group2_left_right, group1_middle, group2_middle)
        
        total_overlap = overlap_left_right + overlap_middle
        
        # (diff_left_right_group2 < min_count_middle_group1 or diff_left_right_group2 < min_count_middle_group1 + 20)
        if total_overlap > max_overlap and abs(len(group1_left_right)-len(group1_middle))< 3 and abs(diff_left_right_group2) < min_count_middle_group1 + 20:
            max_overlap = total_overlap
            best_split = (i, len_group1 - i)
            overlap_left_right1 = overlap_left_right
            overlap_middle1 = overlap_middle
            min_count_middle_group1 = abs(diff_left_right_group2)

    return best_split, overlap_left_right1, overlap_middle1


def split1(group1,count):
    len_group1 = len(group1)
    min_count_middle_group1 = sum(count.get(item[0], 0) for item in group1)
    low_high = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # low_high = [0.15]
    res = []
    for i in range(len(low_high)):
        middle_ratio = 1 - low_high[i]
        lh = int(len_group1*low_high[i])
        group1_left = group1[:lh]
        group1_right = group1[len_group1-lh:]
        res.append([group1_left,group1_right])
    return res

path1 = "./split_files/im_coliee_new_lr1.json"
# 找到最优的划分点
content = "./dataset/coliee"

with open(path1,"r",encoding='utf-8')as f:
    data_json = json.load(f)
    f.close()
count1 = 0
count2 = 0
count3 = 0
count4 = 0
result1 = {}
result2 = {}
result3 = {}
result4 = {}
count_res = {}
files = os.listdir(content)
for file in files:
    path2 = os.path.join(content,file)
    with open(path2,"r",encoding='utf-8')as f:
        data_json3 = json.load(f)
        f.close()
    for case in data_json3:
        for crime in case["Crimes"]:
            # print(crime)
            crime_name = crime["Crime_Type"][0].replace("(边)","")
            if crime_name not in count_res:
                count_res[crime_name] = {}
            for item in crime:
                # if crime_name in choose1:
                if (type(crime[item])==str or type(crime[item])==list) and len(crime[item])!=0:
                    if type(crime[item])==str:
                        count_res[crime_name][item]=count_res[crime_name][item]+1 if item in count_res[crime_name] else 1
                    else:
                        count_res[crime_name][item]=count_res[crime_name][item]+len(crime[item]) if item in count_res[crime_name] else len(crime[item])
result = {}
for i in range(1):
    for j in range(9):
        result[(i,j,"l")] = {}
        result[(i,j,"h")] = {}
for crime in data_json:
    group1 = data_json[crime]
    group_all = [split1(group1,count_res[crime])]
    for i in range(len(group_all)):
        for j in range(len(group_all[i])):
            lh, mid = group_all[i][j][0],group_all[i][j][1]
            result[(i,j,"l")][crime] = [b[0] for b in lh]
            result[(i,j,"h")][crime] = [b[0] for b in mid]
            print(i,j,crime)
for i in range(1):
    for j in range(9):
        with open("./split_files/coliee_lr_new2_"+str(i)+"_"+str(j)+"_l.json","w",encoding='utf-8')as f:
            json.dump(result[(i,j,"l")],f,ensure_ascii=False, indent=4)
        with open("./split_files/coliee_lr_new2_"+str(i)+"_"+str(j)+"_h.json","w",encoding='utf-8')as f:
            json.dump(result[(i,j,"h")],f,ensure_ascii=False, indent=4)


