'''
Author: ccbao 1689940525@qq.com
Date: 2023-06-28 14:19:38
LastEditors: ccbao 1689940525@qq.com
LastEditTime: 2023-06-28 14:19:45
FilePath: /Pruning/Pruning_dropout_dropoutconnect.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import numpy as np

def dropout_layer(x, dropout_rate):
    dropout_mask = np.random.randn(*x.shape) > dropout_rate
    # dropout_mask = np.random.binomial(1, 1-dropout_rate, size=x.shape)
    return x * dropout_mask / (1 - dropout_rate)

def dropconnect_layer(weights, input_data, dropconnect_rate):
    dropconnect_mask = np.random.randn(*weights.shape) > dropconnect_rate
    masked_weights = weights * dropconnect_mask
    return input_data @ masked_weights

# Example usage for dropout
input_data = np.array([[0.1, 0.5, 0.2],
                       [0.8, 0.6, 0.7],
                       [0.9, 0.3, 0.4]])

dropout_rate = 0.5
output_data_dropout = dropout_layer(input_data, dropout_rate)
print(output_data_dropout)

# Example usage for dropconnect
dropconnect_rate = 0.5
weights = np.random.randn(3,4)
output_data_dropconnect = dropconnect_layer(weights, input_data, dropout_rate)
print(output_data_dropconnect)
