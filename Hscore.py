#此程序为数据输入接口，处理各种格式的数据并且转换为矩阵或者图格式
#
import numpy as np
import pandas as pd
import networkx as nx
import random as rd
import copy as cp

filepath = 'data/'
filename = 'TCMSP_DB_加工.xlsx'
disease_t = 'v_Targets_Diseases'
disease_file = 'diseasename/diseasename_Alzheimer.csv'
direct_target = 'diseasename/jinshi.csv'
neighbor_num = 0
direct_flag = 0 #1表示直接使用靶点,0表示入口为疾病名称

def datafromcsv(fileapath):
    df = pd.DataFrame(fileapath)
    return df

def data_from_excel_sheet(filepath, st_name):
    df = pd.read_excel(filepath, sheet_name = st_name)  # 可以通过sheet_name来指定读取的表单
    return df

def disease_target(filepath , filename):#根据疾病名称读取疾病，靶点矩阵,疾病名称放在diseasename.csv里面
    disease_targ = data_from_excel_sheet(filepath + filename, disease_t)  # 计算中药对应的成分

    disease_list = pd.read_csv(disease_file, sep = '#')
    disease_tar = disease_targ[disease_targ['disease_name'].isin(list(disease_list['disease_name']))]
    disease_tar = pd.DataFrame(disease_tar['TARGET_ID'])
    disease_tar['TARGET_INDEX'] = list(range(len(disease_tar)))
    disease_tar =disease_tar.reset_index()
    disease_tar=disease_tar.drop(['index'], axis=1)
    disease_tar.to_csv('target.dictionary.tsv', sep='\t')
    return disease_tar
#a = disease_target(filepath , filename)
def Hscore(targets_mol_herb, importance_score):

    target = pd.read_csv('target.dictionary.tsv', sep='\t')
    mol = pd.read_csv('mol.dictionary.tsv', sep='\t')
    herb = pd.read_csv('herb.dictionary_446.tsv',sep=',')
    targets_mol_herb=targets_mol_herb
    target_indexs=[]
    mol_indexs=[]
    herb_indexs=[]
    for i in range(len(targets_mol_herb)):
        target_index = target.loc[target['TARGET_ID'] == targets_mol_herb.iloc[i]['TARGET_ID'], 'TARGET_INDEX'].values[0]
        mol_index = mol.loc[mol['MOL_ID'] == targets_mol_herb.iloc[i]['MOL_ID'], 'MOL_INDEX'].values[0]
        herb_index = herb.loc[herb['herb_cn_name'] == targets_mol_herb.iloc[i]['herb_cn_name'], 'herb_INDEX'].values[0]

        target_indexs.append(target_index)
        mol_indexs.append(mol_index+len(target))
        herb_indexs.append(herb_index+len(target)+len(mol))
    #targets_mol_herb = pd.DataFrame(target_index,mol_indexs,herb_indexs, columns=['target', 'mol', 'herb'])
    targets_mol_herb = pd.DataFrame({'target': target_indexs, 'mol':mol_indexs,  'herb': herb_indexs})
    A = np.zeros((len(target)+len(mol)+len(herb), len(target)+len(mol)+len(herb)))
    #t_m['walk_score'] = t_m['TARGET_ID'].apply(lambda x: importance_score[x] * 1 if x in importance_score else 0)

    for i in range(len(targets_mol_herb)):
         A[targets_mol_herb.iloc[i]['target'], targets_mol_herb.iloc[i]['mol']]=1
         A[targets_mol_herb.iloc[i]['mol'], targets_mol_herb.iloc[i]['herb']]=1
    A = A+A.T
    W = np.zeros_like(A, dtype=float)

    # 对于每一行
    for i in range(A.shape[0]):
        # 计算分母部分的和
        denominator_sum = np.sum(A[i])
        # 对于每一列
        for j in range(A.shape[1]):
            # 计算权重矩阵中的每个元素
            W[i, j] = A[i, j] / denominator_sum
    score = A * (W ** 2)
    Hscore = score[-446:]
    Hscore = np.sum(Hscore, axis=1)
    Hscore = pd.DataFrame({'herb_name':herb['herb_cn_name'], 'hscore': Hscore})
    return Hscore




