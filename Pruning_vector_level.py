'''
Author: ccbao 1689940525@qq.com
Date: 2023-06-25 23:14:38
LastEditors: ccbao 1689940525@qq.com
LastEditTime: 2023-06-26 07:36:29
FilePath: /Pruning/Pruning_vector_level.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import matplotlib.pyplot as plt

def vector_pruning(matrix, idx):
    row,col = idx
    pruned_matrix = matrix.copy()
    pruned_matrix[row,:] = 0
    pruned_matrix[:,col] = 0
    return pruned_matrix

matrix = np.random.randn(3,4)
idx = (1,2)
pruned_matrix = vector_pruning(matrix, idx)
print(f"matrix:  {matrix}")
print(f"pruned_matrix: {pruned_matrix}")