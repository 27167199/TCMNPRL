#根据算法生成组方和组方分数评价
import csv

import formula_herb_importance as fhi
import Data_input as di
import Data_output as do
import herb_pairs_from_formula as hpff
import random as rd
import PPI_analyse as ppi
import pandas as pd
from DRL_GE import DRLHerbRecommendation
from RL_PG import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


low_formula_num = 1#方剂最小中药数目
up_formula_num = 20#方剂最小中药数目加1
jun_w = 2 #君药的权重，下同
chen_w = 1
zuo_w = 1
shi_w = 1

jun_p = 0.1 #君药比例 下同
chen_p =  0.2
zuo_p = 0.3
shi_p =0.4

#初始化方剂
def innit_formula_seed(herb_score_dict, row_num , col_num_list):#herb_score_dict是中药分数字典,row_num为要生成的方剂数目,col_num为方剂中的中药数
    formula_herb_list = [] #方剂列表，包含有可能的row_num个方剂组成
    herb_score_dict = sorted(herb_score_dict.items(),key = lambda x: x[1], reverse=False)

    for i in range(row_num):
        formula_herb_list_seed = []  # 方剂种子，初始生成的方剂
        while(len(formula_herb_list_seed) < col_num_list[i]):
            benchmark_num = 0
            r = rd.random()
            for (k,v) in herb_score_dict:
                if r < benchmark_num + float(v) and k not in formula_herb_list_seed:
                    formula_herb_list_seed.append(k)
                    break
                benchmark_num = benchmark_num + float(v)
        #print(formula_herb_list_seed)
        formula_herb_list_seed = sorted(formula_herb_list_seed)
        formula_herb_list.append(formula_herb_list_seed)
    return formula_herb_list

#计算方剂得分
def compute_formula_score(formula_list , herb_score_dict, pair_num_dict):
    #基于君臣佐使,第一种排序方法。
    dict_formula = {}
    for h in formula_list:
        dict_formula[h] = herb_score_dict[h]
    #基于君臣佐使,第一种排序方法。


    '''
    #基于君臣佐使,第二种排序方法。
    dict_formula = {}
    formula_score_tmp = 0.0
    formula_score_del = 0.0

    for h in range(0,len(formula_list)):
        for herb_i in range(len(formula_list) - 1):
            for herb_j in range(herb_i + 1, len(formula_list)):
                if (str(formula_list[herb_i]) + str(formula_list[herb_j])) in pair_num_dict or (str(formula_list[herb_j]) + str(formula_list[herb_i])) in pair_num_dict:
                    if str(formula_list[herb_i]) + str(formula_list[herb_j]) in pair_num_dict:
                        formula_score_tmp += herb_score_dict[formula_list[herb_i]] * herb_score_dict[
                            formula_list[herb_j]] * pair_num_dict[str(formula_list[herb_i]) + str(formula_list[herb_j])]
                    elif str(formula_list[herb_j]) + str(formula_list[herb_i]) in pair_num_dict:
                        formula_score_tmp += herb_score_dict[formula_list[herb_i]] * herb_score_dict[
                            formula_list[herb_j]] * pair_num_dict[str(formula_list[herb_j]) + str(formula_list[herb_i])]

                if (herb_i!=h) and (herb_j!=h):
                    if (str(formula_list[herb_i]) + str(formula_list[herb_j])) in pair_num_dict or (str(formula_list[herb_j]) + str(formula_list[herb_i])) in pair_num_dict:
                        if str(formula_list[herb_i]) + str(formula_list[herb_j]) in pair_num_dict:
                            formula_score_del += herb_score_dict[formula_list[herb_i]] * herb_score_dict[
                                formula_list[herb_j]] * pair_num_dict[str(formula_list[herb_i]) + str(formula_list[herb_j])]
                        elif str(formula_list[herb_j]) + str(formula_list[herb_i]) in pair_num_dict:
                            formula_score_del += herb_score_dict[formula_list[herb_i]] * herb_score_dict[
                                formula_list[herb_j]] * pair_num_dict[str(formula_list[herb_j]) + str(formula_list[herb_i])]

        dict_formula[formula_list[h]] = formula_score_tmp - formula_score_del
    # 基于君臣佐使,第二种排序方法。
    '''
    dict_formula = sorted(dict_formula.items(), key=lambda x: x[1], reverse=True)
    formula_list_sorted = []
    for (k,v) in dict_formula:
        formula_list_sorted.append(k)

    #针对君臣佐使加权
    jun_list = []
    chen_list = []
    zuo_list = []
    shi_list = []
    junchenzuoshi_dict = {}

    if len(formula_list_sorted) > 3:
        jun_list = formula_list_sorted[0:int(jun_p*len(formula_list_sorted))]
        chen_list = formula_list_sorted[int(jun_p*len(formula_list_sorted)):int((jun_p+chen_p)*len(formula_list_sorted))]
        zuo_list = formula_list_sorted[int((jun_p+chen_p)*len(formula_list_sorted)):int((jun_p+chen_p+zuo_p)*len(formula_list_sorted))]
        shi_list = formula_list_sorted[int((jun_p+chen_p+zuo_p)*len(formula_list_sorted)):]

        for jun in jun_list:
            junchenzuoshi_dict[jun] = jun_w
        for chen in chen_list:
            junchenzuoshi_dict[chen] = chen_w
        for zuo in zuo_list:
            junchenzuoshi_dict[zuo] = zuo_w
        for shi in shi_list:
            junchenzuoshi_dict[shi] = shi_w
    #针对君臣佐使加权
    else:
        for h in formula_list_sorted:
            junchenzuoshi_dict[h] = 1


    formula_score = 0.0
    for herb_i in range(len(formula_list)-1):
        for herb_j in range(herb_i + 1 , len(formula_list)):
            if (str(formula_list[herb_i])+str(formula_list[herb_j])) in pair_num_dict or (str(formula_list[herb_j])+str(formula_list[herb_i])) in pair_num_dict:
                if str(formula_list[herb_i])+str(formula_list[herb_j]) in pair_num_dict:
                    formula_score += junchenzuoshi_dict[formula_list[herb_i]]*junchenzuoshi_dict[formula_list[herb_j]]*herb_score_dict[formula_list[herb_i]] * herb_score_dict[formula_list[herb_j]] * pair_num_dict[str(formula_list[herb_i])+str(formula_list[herb_j])]
                elif str(formula_list[herb_j])+str(formula_list[herb_i]) in pair_num_dict:
                    formula_score += junchenzuoshi_dict[formula_list[herb_i]]*junchenzuoshi_dict[formula_list[herb_j]]*herb_score_dict[formula_list[herb_i]] * herb_score_dict[formula_list[herb_j]] * pair_num_dict[str(formula_list[herb_j])+str(formula_list[herb_i])]
            else:
                formula_score = formula_score #- herb_score_dict[formula_list[herb_i]] * herb_score_dict[formula_list[herb_j]]
    return formula_score/len(formula_list)#(len(formula_list)*len(formula_list))

def is_same_herb_in_formula(formulalist):#判断方剂里面是否有重复的中药
    if len(formulalist) == len(set(formulalist)):
        return True
    else:
        return False


def variation(herbs_list,p,herb_score_dict):#变异
    herb_score_dict = sorted(herb_score_dict.items(),key = lambda x: x[1], reverse=False)
    herbs_list_new = []
    for herbi in herbs_list:
        r = rd.random()
        if r > p:
            herbs_list_new.append(herbi)
        else:
            benchmark_num = 0.0
            d = rd.random()
            for (k,v) in herb_score_dict:
                if  d < benchmark_num + float(v): #and k not in herbs_list_new:
                    herbs_list_new.append(k)
                    break
                benchmark_num = benchmark_num + float(v)

    return herbs_list_new

def generate_formula_list(rows_num):#生成1-15的方剂列表
    rows_num_list = []
    for i in range(rows_num):
        num = rd.randint(low_formula_num,up_formula_num)#从2味药到15味药
        rows_num_list.append(num)
    return rows_num_list

#根据方剂得分,删除后一半得分较低的方剂，然后，交叉、变异，生成新的方剂组
def Genetic_Algorithm(formulas_score_dict,herb_score_dict,herb_pair_from_data,formula_nums):
    formulas_score_dict = sorted(formulas_score_dict.items(),key = lambda x: x[0], reverse=True)#排序
    print(len(formulas_score_dict),formulas_score_dict)
    #保留一半得分高的方剂
    new_formulas_score_list1 = []
    new_formulas_score_list2 = []

    for i in range(int(len(formulas_score_dict)/2)):
        new_formulas_score_list1.append(formulas_score_dict[i])
    formulas_num = len(new_formulas_score_list1)
    rd.shuffle(new_formulas_score_list1)
    for i in range(0,formulas_num,2):
        (scorei,formulai) = new_formulas_score_list1[i]
        (scorej,formulaj) = new_formulas_score_list1[i+1]

        formulak = []
        formulak.extend(formulai)
        formulak.extend(formulaj)
        rd.shuffle(formulak)
        #组成新的方子,交叉变异
        formula1 = formulak[0:int(len(formulak)/2)]
        formula1 = variation(formula1,0.1,herb_score_dict)#变异，变异系数0.1
        formula1 = sorted(formula1)#相同的方剂序列一致
        formula1_list = (compute_formula_score(formula1, herb_score_dict, herb_pair_from_data)* 10000,formula1)

        formula2 = formulak[int(len(formulak)/2):len(formulak)]
        formula2 = variation(formula2,0.1,herb_score_dict)#变异，变异系数0.1
        formula2 = sorted(formula2)
        formula2_list = (compute_formula_score(formula2, herb_score_dict, herb_pair_from_data) *10000,formula2)
        if is_same_herb_in_formula(formula1):
            new_formulas_score_list2.append(formula1_list)
        if is_same_herb_in_formula(formula2):
            new_formulas_score_list2.append(formula2_list)

    #最大数值方剂给予一定概率变异
    variation_new_formulas_score_list1 = []
    for (score,fm) in new_formulas_score_list1:
        formula1 = variation(fm, 0.01, herb_score_dict)  # 变异，变异系数0.01
        formula1 = sorted(formula1)  # 相同的方剂序列一致
        formula1_list = (compute_formula_score(formula1, herb_score_dict, herb_pair_from_data) * 10000, formula1)
        if is_same_herb_in_formula(formula1):
            variation_new_formulas_score_list1.append(formula1_list)

    variation_new_formulas_score_list1.extend(new_formulas_score_list2)
    new_formulas_score_dict = {key:value for (key,value) in variation_new_formulas_score_list1}
    while (len(new_formulas_score_dict.keys()) != formula_nums):#有时候会有相同的方剂，使得总体数量减少，重新生成，补齐
        f_h_l = innit_formula_seed(herb_score_dict, 1, [rd.randint(low_formula_num,up_formula_num)])
        f_score = compute_formula_score(f_h_l[0], herb_score_dict, herb_pair_from_data) * 10000
        new_formulas_score_dict[f_score] = f_h_l[0]
    return  new_formulas_score_dict

#抽取从生成方剂的中number数量的方剂,用于排序和比较,number默认1000
def formula_sorted_num(formula_all_dict,number=500):
    formula_all_list = sorted(formula_all_dict.items(), key=lambda x: x[0], reverse=True)  # 排序
    formula_num_sample = []
    (max_fmapscore,max_formula) = formula_all_list[0]
    (min_fmapscore,min_formula) = formula_all_list[len(formula_all_list) -1]

    #将分数统一为0到100
    #1保留最大的前50个数值
    for i in range(50):
        (fmapscore,formula) = formula_all_list[i]
        std_fmapscore = (fmapscore - min_fmapscore)/(max_fmapscore-min_fmapscore)*100
        formula_num_sample.append((fmapscore,formula))
    formula_all_list = formula_all_list[4:]
    bin_pri = [80,60,40,20,0]
    ix = 0
    tmp_bin_formula_list = []
    for bin in formula_all_list:
        (fmapscore,formula) = bin
        std_fmapscore = (fmapscore - min_fmapscore)/(max_fmapscore-min_fmapscore)*100
        if std_fmapscore > bin_pri[ix] - 0.1:
            tmp_bin_formula_list.append((fmapscore,formula))
        else:
            print(len(tmp_bin_formula_list))
            rs = rd.sample(tmp_bin_formula_list,int((number-50) / 500))
            formula_num_sample.extend(rs)
            ix = ix + 1
            tmp_bin_formula_list = []
    rs = rd.sample(tmp_bin_formula_list, int((number - 50) / 500))
    formula_num_sample.extend(rs)
    formula_num_sample_dict = {}
    for (k,v) in formula_num_sample:
        formula_num_sample_dict[k] = v
    return formula_num_sample_dict

if __name__ == '__main__':
    filepath = 'data/'
    filename = 'TCMSP_DB_加工.xlsx'
    '''
    herbs = pd.read_excel("./herb_list_499.xlsx")
    herbs['herb_INDEX'] = [i for i in range(len(herbs))]
    herbs.to_csv('herb.dictionary_499.tsv', index=False)
    
    herbs = pd.read_csv("./herb.dictionary_446.tsv")
    herbs['herb_INDEX'] = [i for i in range(len(herbs))]
    herbs.to_csv('herb.dictionary_446.tsv', index=False)
    '''
    target_molecule = di.target_mol(filepath, filename, tar='0')#获取疾病对应的靶点和成分，tar=0 表示指定疾病的靶点和成分
    herb_mols =  di.herb_molecules(filepath, filename) #中药对应的成分
    targets_mol_herb = di.targets_mol_herb(filepath,filename)
    print(targets_mol_herb)
    #从PPI网络中获取节点重要性
    degree,pagerank,eigenvector,closeness,betweenness = ppi.symbol_sore_from_PPI()
    #importance_list = [degree,pagerank,eigenvector,closeness,betweenness]
    importance_list = [pagerank]

    #importance_list_name = ['degree','pagerank','eigenvector','closeness','betweenness']
    importance_list_name = ['pagerank']

    for importance_ix in range(len(importance_list)):
        '''
        herb_score = fhi.herb_walk_score_interation(targets_mol_herb,importance_list[importance_ix])#计算对应的药物分数
        herb_score = herb_score[['herb_cn_name','walk_score']].drop_duplicates()
        
        herb_score['walk_score'] = herb_score.apply(lambda x: x['walk_score']/herb_score['walk_score'].sum(),axis=1)#归一化
        herb_score_dict = {key: values for key, values in zip(herb_score['herb_cn_name'], herb_score['walk_score'])}

        '''
        from Hscore import Hscore
        herb_score = Hscore(targets_mol_herb,importance_list[importance_ix])
        herb_score['hscore'] = herb_score.apply(lambda x: x['hscore'] / herb_score['hscore'].sum(),axis=1)  # 归一化
        herb_score_dict = {key:values for key, values in zip(herb_score['herb_name'], herb_score['hscore'])}#转换为字典结构


        #fname = 'all_libing_hsocre.csv'
        #do.writedicttodata(fname, herb_score_dict)

        '''
        pair_score = 'herb_herb_mol_jaccard_gini'
        pair_s = di.data_from_excel_sheet(filepath + filename, pair_score)
        p_s = pair_s[['herb1','herb2','cos_mol']]
        p_s_dict = {(key1 ,key2):values for key1, key2 ,values in zip(p_s['herb1'], p_s['herb2'], p_s['cos_mol'])}#转换为字典结构
        '''

        formula_nums = 1000
        filepath = 'data/'
        #filename = '叶天士新.csv'
        #filename = '第一批100首-药物组成.csv'
        #filename = '中成药数据库.csv'
        filename = '伤寒金匮.csv'
        #filename = '瘟疫温病条辨.csv'
        #filename = '各地新冠方剂.csv'

        rows_list = generate_formula_list(formula_nums)#随机生成方剂中中药数目
        herb_pair_from_data = hpff.herb_pair_score_from_data(filepath,filename,herb_mols)

        #Sab数值替换配伍得分
        #filepath = 'D:/formula_result/1算法设计/2药物配伍得分和SAB等指标的关系/'
        #filename = '2药物配伍得分和SAB等指标的关系.xlsx'
        #herb_pair_from_data = hpff.herb_pair_score_from_Sab(filepath, filename, herb_mols)
        #Sab数值替换配伍得分

        input_size = len(herb_score)
        output_size = len(herb_score)
        model = PolicyNetwork(input_size, output_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(model, optimizer, episodes=3000, herbs=herb_score['herb_name'].values.tolist(), evaluate_fitness=compute_formula_score, herb_score_dict=herb_score_dict, herb_pair_from_data=herb_pair_from_data)

        num_prescriptions = 1000  # 生成1000个方剂
        generated_prescriptions = generate_prescriptions(model, herb_score['herb_name'].values.tolist(), num_prescriptions)
        scores_with_prescriptions = []
        for i, prescription in enumerate(generated_prescriptions):
            score = compute_formula_score(prescription, herb_score_dict, herb_pair_from_data)
            scores_with_prescriptions.append((score, prescription))
            #print(f"Prescription {i + 1}: {prescription}, score: {score}")

        # Sort by score in descending order
        scores_with_prescriptions.sort(reverse=True, key=lambda x: x[0])
        i=0
        # Print sorted scores with corresponding prescriptions
        for score, prescription in scores_with_prescriptions:
            i=i+1
            print(f"Rank：{i},Score: {score}, Prescription: {prescription}")

        # 只取前50个元素
        first_50 = scores_with_prescriptions[:50]

        # 指定要保存的CSV文件名
        csv_file = 'output_top50.csv'

        # 将前50个元素写入CSV文件
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in first_50:
                writer.writerow([item])  # 将每个元素写入新行

        print(f"前50个元素已保存到 {csv_file} 文件中。")


