# -*- coding: UTF-8 -*- 
import re

#python data
fp = open("/mnt/share/classfication_inference/onnx_run_reslut.txt", "r")
#cpp data
fc = open("/mnt/share/classfication_inference/classfication/bin/onnx_run_reslut.txt", "r")
toltle_unsamiler = 0
idx = 1
list = []
for line1, line2 in zip(fp.readlines(), fc.readlines()):
    line1 = line1.strip('\n')
    line2 = line2.strip('\n')
    str_py = ''
    str_cpp = ''
    if line1.find('top1') != -1 and line1.find('top5') != -1:
        pattern_top1 = re.compile('top1\[(.*?)\]')
        top1 = pattern_top1.findall(line1)[0]
        pattern_top5 = re.compile('top5\[(.*?)\]')
        top5 = pattern_top5.findall(line1)[0]
        str_py = top1 + top5
        # print("top1:", top1)
        # print("top5:", top5)
        # print("str_py:", str_py)
    if line2.find('top1') != -1 and line2.find('top5') != -1:
        pattern_top1 = re.compile('top1\"\:\"(.*?)\"')
        top1 = pattern_top1.findall(line2)[0]
        pattern_top5 = re.compile('top5\"\:\"(.*?)\"')
        top5 = pattern_top5.findall(line2)[0]
        str_cpp = top1 + top5
        # print("top1:", top1)
        # print("top5:", top5)
        # print("str_cpp:", str_cpp)
    if str_py != str_cpp:
        toltle_unsamiler += 1
        list.append(idx)
    idx += 1

print("toltle_unsamiler: ", toltle_unsamiler)
print("list:", list)
fp.close()
fc.close()