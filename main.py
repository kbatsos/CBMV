from __future__ import division
import sys
import math
import random

import scipy
from sklearn.feature_extraction import image

import os
import numpy as np
import matplotlib.pyplot as plt

import argparse

#sys.path.insert(0,'./cpp/rectification/Debug')
sys.path.insert(0,'.')

#import librectification as rect 
from train import Training as cbmvtrain
from test import Testing as cbmvtest

parser = argparse.ArgumentParser(description='CBMV.')
parser.add_argument('--data_path', dest='data_path', default="")
parser.add_argument('--train_set', dest='train_set', default='')
parser.add_argument('--train_add', dest='train_add', default='')
parser.add_argument('--test_set', dest='test_set', default='')
parser.add_argument('--train', dest='train', action='store_true');
parser.add_argument('--dtn', dest='dtn', default='mb');
parser.add_argument('--model',dest='model',default='')
parser.add_argument('--prob_save_path',dest='prob_save_path',default='./results')
parser.add_argument('--disp_save_path',dest='disp_save_path',default='./results/disp.pfm') #
parser.add_argument('--l',dest='lim',default='')
parser.add_argument('--r',dest='rim',default='')
parser.add_argument('--calib',dest='calib',default='')
parser.add_argument('--w',dest='w',default=0)
parser.add_argument('--h',dest='h',default=0)
parser.add_argument('--d',dest='d',default=0)
parser.add_argument('--n_jobs',dest='n_jobs',type=int,default=6) # The number of jobs to run in parallel for both fit and predict in random forest;

# added by CCJ;
parser.add_argument('--isLocalExp',dest='isLocalExp', action='store_true')
parser.add_argument('--saveCostVolume', dest='saveCostVolume', action='store_true');
parser.add_argument('--loadCostVolume', dest='loadCostVolume', action='store_true');

args = parser.parse_args()

print 'args = ', args

if(args.train):
	t_phase = cbmvtrain( args.train_set, args.data_path,args.train_add)
	t_phase.train_model(args.model, args.n_jobs)
else:
	test_p = cbmvtest(args.test_set, args.data_path,args.lim,args.rim,args.calib,args.w,args.h,args.d)
        
        # checking cost volume for loading;
        cost_file = args.prob_save_path+args.test_set+".prob.npy"
        if args.loadCostVolume and os.path.exists(cost_file):
            print ('Loading cost volume from file : {}'.format(cost_file))
	    proba = np.load(cost_file)
        else:
            # run random forest model for testing;
            print ('Cost volume: {} not exists!'.format(cost_file))
            proba = test_p.test_model(args.model,args.prob_save_path)
            if args.saveCostVolume:
                # save cost volume
                np.save(cost_file,proba)
                print ('Saved cost volume {}'.format(cost_file))
        
	# Do post
	test_p.eval_prob(proba.astype(np.float64),args.disp_save_path,display=False,interpolate=False,isLocalExp=args.isLocalExp)
