import numpy as np
from scipy.stats import rankdata, friedmanchisquare
from scipy.stats import wilcoxon
import pandas as pd

# 读取Excel文件
path = "D:\\001__studyMetarial\\05__mypaper\\ranksum.xlsx"
excel_data = pd.read_excel(path)

column_SSA      = np.array(excel_data['SSA'])
column_HSSA     = np.array(excel_data['HSSA'])
column_SCA_CSSA = np.array(excel_data["SCA-CSSA"])
column_CIW_SSA  = np.array(excel_data['CIW-SSA'])


# statistic: 弗里德曼统计量，度量了各组之间的差异程度
# p_value: 用于判断是否拒绝零假设
# 执行Wilcoxon秩和检验
statistic, p_value = wilcoxon(column_CIW_SSA, column_SSA)
print(statistic, "%.4E" % p_value)
statistic, p_value = wilcoxon(column_CIW_SSA, column_HSSA)
print(statistic, "%.4E" % p_value)
statistic, p_value = wilcoxon(column_CIW_SSA, column_SCA_CSSA)
print(statistic, "%.4E" % p_value)


get_rank = np.array([column_SSA, column_HSSA, column_SCA_CSSA, column_CIW_SSA]).T
ranks = [rankdata(func_rank) for func_rank in get_rank]

average_ranks = np.array(ranks).T


mean_SSA = average_ranks[0].mean()
mean_HSSA = average_ranks[1].mean()
mean_SCA_CSSA = average_ranks[2].mean()
mean_CIW_SSA = average_ranks[3].mean()

print(mean_SSA, mean_HSSA, mean_SCA_CSSA, mean_CIW_SSA)


