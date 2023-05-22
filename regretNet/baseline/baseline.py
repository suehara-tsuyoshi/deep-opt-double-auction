from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class MDProtcol:
    def __init__(self, config, data_b, data_s):
        self.config = config
        self.data_b = data_b
        self.data_s = data_s
    
    def opt_rev(self):
        num_buyers = self.config.num_buyers
        num_sellers = self.config.num_sellers
        num_instances = self.data_b.shape[0]

        revenue = 0.0
        for i in range(num_instances):

            data_b_sorted = np.sort(self.data_b[i])[::-1]
            data_s_sorted = np.sort(self.data_s[i])

            idx = 0
            max_idx = min(len(data_b_sorted), len(data_s_sorted))

            for j in range(max_idx):
                if data_b_sorted[j] >= data_s_sorted[j]:
                    idx += 1
                else:
                    break

            if idx == max_idx:
                price = (data_b_sorted[idx-1]+data_s_sorted[idx-1])/2.0
                revenue += price * idx
            elif idx != 0:
                price = (data_b_sorted[idx]+data_s_sorted[idx])/2.0
                if data_b_sorted[idx-1] >= price and data_s_sorted[idx-1] <= price:
                    revenue += price * idx
                else:
                    revenue += (data_b_sorted[idx-1]) * (idx-1)
            
            
        revenue = revenue/num_instances
        return(revenue)
    
