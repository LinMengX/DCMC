import difflib
import math
import os
import re
import torch

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import median_survival_times
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal, chi2_contingency


def p_normalize(x, p=2):
    return x / (torch.norm(x, p=p, dim=1, keepdim=True) + 1e-6)

def clinical_enrichement(label,clinical):
    cnt = 0
    stat, p_value_age = kruskal(np.array(clinical["age"]), np.array(label))
    if p_value_age < 0.05:
        cnt += 1
        #print(f"---age---p-value: {p_value_age}")
    stat_names = ["gender","T","M","N","stage"]
    for stat_name in stat_names:
        if stat_name in clinical:
            c_table = pd.crosstab(clinical[stat_name],label,margins = True)
            stat, p_value_other, dof, expected = chi2_contingency(c_table)
            #print(f"---{stat_name}---p-value: {p_value_other}")
            if p_value_other < 0.05:
                cnt += 1
                #print(f"---{stat_name}---p-value: {p_value_other}")
    return cnt


def log_rank(df):
    res = dict()
    results = multivariate_logrank_test(df['Survival'], df['label'], df['Death'])
    res['p'] = results.summary['p'].item()
    res['log10p'] = -math.log10(results.summary['p'].item())
    res['log2p'] = -math.log2(results.summary['p'].item())
    return res

def get_clinical(path,survival,cancer_type):
    clinical = pd.read_csv(f"{path}/{cancer_type}",sep="\t")
    if cancer_type == 'kidney':
        replace = {'gender.demographic': 'gender','submitter_id.samples': 'sampleID'}
        clinical = clinical.rename(columns=replace)
        clinical["sampleID"] = [re.sub("A", "", x) for x in clinical["sampleID"].str.upper()]
    clinical["sampleID"] = [re.sub("-", ".", x) for x in clinical["sampleID"].str.upper()]

    survival['age'] = pd.NA
    survival['gender'] = pd.NA
    if 'pathologic_T' in clinical.columns:
        survival['T'] = pd.NA
    if 'pathologic_M' in clinical.columns:
        survival['M'] = pd.NA
    if 'pathologic_N' in clinical.columns:
        survival['N'] = pd.NA
    if 'tumor_stage.diagnoses' in clinical.columns:
        survival['stage'] = pd.NA
    i = 0

    for name in survival['PatientID']:
        # print(name)
        flag = difflib.get_close_matches(name,list(clinical["sampleID"]),1,cutoff=0.6)
        if flag:
            idx = list(clinical["sampleID"]).index(flag[0])
            survival['age'][i] = clinical['age_at_initial_pathologic_diagnosis'][idx]
            survival['gender'][i] = clinical['gender'][idx]
            if 'pathologic_T' in clinical.columns:
                survival['T'][i] = clinical['pathologic_T'][idx]
            if 'pathologic_M' in clinical.columns:
                survival['M'][i] = clinical['pathologic_M'][idx]
            if 'pathologic_N' in clinical.columns:
                survival['N'][i] = clinical['pathologic_N'][idx]
            if 'tumor_stage.diagnoses' in clinical.columns:
                survival['stage'][i] = clinical['tumor_stage.diagnoses'][idx]
        else: print(name)
        i = i + 1
    return survival.dropna(axis=0, how='any')
