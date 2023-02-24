#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from FINDER import FINDER
from tqdm import tqdm


def main():
    dqn = FINDER()
    data_test_path = '/home/cpy/FINDER-master/data/synthetic/synthetic/degree_cost/'
#     data_test_name = ['30-50', '50-100', '100-200', '200-300', '300-400', '400-500']
    data_test_name = ['30-50', '50-100', '100-200', '200-300', '300-400', '400-500']
    file_path = '/home/cpy/FINDER-master/results/FINDER_ND/synthetic'
    #model_file = '/home/cpy/FINDER-origin/FINDER-master/code/FINDER_ND/models/nrange_30_50_iter_78000.ckpt'
    #model_file = '/home/cpy/FINDER-master/code/FINDER_ND/models/Model_0213_barabasi_albert/nrange_30_50_iter_54010.ckpt'
    if not os.path.exists('/home/cpy/FINDER-master/results/FINDER_ND'):
        os.mkdir('/home/cpy/FINDER-master/results/FINDER_ND')
    if not os.path.exists('/home/cpy/FINDER-master/results/FINDER_ND/synthetic'):
        os.mkdir('/home/cpy/FINDER-master/results/FINDER_ND/synthetic')
        
    with open('%s/result_link_0.2.txt'%file_path, 'w') as fout:
        for i in tqdm(range(len(data_test_name))):
            data_test = data_test_path + data_test_name[i]
            time_mean, time_std, w_mean, w_std = dqn.link_removal_syn(data_test)
            #time_mean, time_std, w_mean, w_std = dqn.Evaluate_HXA(data_test, 'HCA')
            fout.write('%s: w: %.2f±%.2f, time: %.2f±%.2f\n' % (data_test_name[i], w_mean, w_std, time_mean, time_std))
            fout.flush()
            print('\ndata_test_%s has been tested!' % data_test_name[i])   
        


if __name__=="__main__":
    main()
