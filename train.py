from __future__ import division
import sys
import math
import random

import scipy
from sklearn.feature_extraction import image
from sklearn.ensemble import RandomForestClassifier
import cPickle;

import os
import numpy as np
import math
import matplotlib.pyplot as plt

#sys.path.insert(0,'./cpp/rectification/Debug')
sys.path.insert(0,'.')
sys.path.insert(0,'./pylibs')
#sys.path.insert(0,'./cpp/matchers/Debug')
#sys.path.insert(0,'./cpp/featextract/Debug')
sys.path.insert(0,'./cpp/lib')

import librectification as rect
import pfmutil as pfm 
import libmatchers as mtc
import libfeatextract as fte

class Training(object):

	def __init__(self,t_set, data_path,ad_set):

		if t_set != '':
			trainfile = open(t_set,"r")
			trainset =  trainfile.read().rstrip().split(",")
			trainfile.close()
		else:
			print "You must specify a file containing the names of the image set to train on!"
			sys.exit()

		if ad_set != '':
			trainfile = open(ad_set,"r")
			addtrainset =  trainfile.read().rstrip().split(",")
			trainset = trainset+addtrainset
			trainfile.close()

		if(len(trainset) == 0):
			print "Train set is empty quiting!"
			sys.exit()

		self.__trainset = trainset
		self.__data_path = data_path
		self.__training_samples = np.empty((0,21))
		self.__censw = 11
		self.__nccw = 3
		self.__sadw =5
		self.__cens_sigma=2*8**2
		self.__ncc_sigma = 0.02
		self.__sad_sigma = 2*100**2


	def __fix_rectification(self,index):
		imgl = scipy.misc.imread(self.__data_path + self.__trainset[index]+"/im0.png",mode='L' )
		imgr = scipy.misc.imread( self.__data_path + self.__trainset[index]+"/im1.png",mode='L' )
                if os.path.exists(self.__data_path +self.__trainset[index]+"/im1L.png"): 
                    imgrL = scipy.misc.imread( self.__data_path +self.__trainset[index]+"/im1L.png",mode='L' )
                else:
                    imgrL = None
			
		h_l = np.zeros((3,3)).astype(np.float32)
		h_r = np.zeros((3,3)).astype(np.float32)

		# returns rectified images and homographies in place
		rect.fixrectification(imgl,imgr,h_l,h_r,)
                if imgrL is not None:
                    imgrLw = rect.warp(img=imgrL,homography=h_r,invert=False,option=False)
                else:
                    imgrLw = None
		return imgl,imgr,imgrLw				
		#return imgl,imgr

	def __read_calib(self,index):
		w=0;
		h=0;
		d=0;
		with open(self.__data_path + self.__trainset[index]+"/calib.txt") as f:
			lines = f.readlines();
			for i in range(0,len(lines)):
				line =  lines[i].strip("\n").split("=")
				if line[0] == "width":
					w = line[1]
				elif line[0] == "height":
					h=line[1]
				elif line[0] == "ndisp":
					d = line[1]
		return int(w),int(h),int(d)


	def __create_samples_mem(self,iml,imr,index):	
		w,h,ndisp = self.__read_calib(index)
		
		gt = pfm.load( self.__data_path + self.__trainset[index]+"/disp0GT.pfm" )[0]

		gt = np.reshape(gt, [gt.shape[0]*gt.shape[1],1])
		infs = np.concatenate( ( np.argwhere(gt == np.inf),np.argwhere(gt < 0)  ), axis=0) 
		# infs = np.empty((0,2))
		gt = np.delete(gt, infs[:,0],axis=0)
		gt = np.round(gt)
		gt = gt.astype(np.int32)	
                #print ("loading gt ... gt shape = {}".format(gt.shape))

		random_samples = fte.generate_d_indices(gt,ndisp,1)
                assert random_samples.shape[1] == 3 # here : 3 means 1 positive sample + 2 negative ones;
		samples = np.empty((random_samples.shape[0]*random_samples.shape[1],21))
                #print ("samples shape = {}".format(samples.shape))
                #print ("staring census ...")

		################## Census compute ##########################################################
                #print ('w = {}, h = {}, ndisp = {}, censW = {}'.format(w, h, ndisp, self.__censw))
                #print ('last iml = {}, last imr = {}'.format(iml[h-1,w-1], imr[h-1,w-1]))
		costcensus = mtc.census(iml,imr,ndisp,self.__censw ).astype(np.float64)
                #print ('costcensus shape = {}'.format(costcensus.shape))
		costcensusR = fte.get_right_cost(costcensus)
		costcensus = np.reshape(costcensus, [ costcensus.shape[0]*costcensus.shape[1],costcensus.shape[2] ])
		costcensusR = np.reshape(costcensusR, [ costcensusR.shape[0]*costcensusR.shape[1],costcensusR.shape[2] ])

		costcensus = np.delete(costcensus, infs[:,0],axis=0)

		samples[:,0] = fte.get_samples(costcensus , random_samples )
		samples[:,4] = fte.extract_ratio( costcensus,random_samples,.01  )
		samples[:,8] = fte.extract_likelihood( costcensus,random_samples,self.__cens_sigma )
		del costcensus
                #print ("census done!")

		r_pkrn = fte.extract_ratio(costcensusR,.01)

		r_pkrn = np.reshape(r_pkrn,[h,w,ndisp])

		r_pkrn = fte.get_left_cost(r_pkrn)
		r_pkrn = np.reshape(r_pkrn, [ r_pkrn.shape[0]*r_pkrn.shape[1],r_pkrn.shape[2] ])
		r_pkrn = np.delete(r_pkrn, infs[:,0],axis=0)

		samples[:,12] = fte.get_samples( r_pkrn,random_samples  )
		del r_pkrn

		r_aml = fte.extract_likelihood(costcensusR,self.__cens_sigma )
		r_aml = np.reshape(r_aml,[h,w,ndisp])
		r_aml = fte.get_left_cost(r_aml)
		r_aml = np.reshape(r_aml, [ r_aml.shape[0]*r_aml.shape[1],r_aml.shape[2] ])
		r_aml = np.delete(r_aml, infs[:,0],axis=0)
		samples[:,16] = fte.get_samples( r_aml,random_samples)

		del r_aml
		del costcensusR

		######################################################################################
		############################### NCC compute ##########################################

		costncc = mtc.nccNister(iml,imr,ndisp,self.__nccw)
		costncc = fte.swap_axes(costncc)	
		costnccR = fte.get_right_cost(costncc)
		costncc = np.reshape(costncc, [ costncc.shape[0]*costncc.shape[1],costncc.shape[2] ])
		costnccR = np.reshape(costnccR, [ costnccR.shape[0]*costnccR.shape[1],costnccR.shape[2] ])
		

		costncc = np.delete(costncc, infs[:,0],axis=0)
		
		samples[:,1] = fte.get_samples(costncc , random_samples )
		samples[:,5] = fte.extract_ratio( costncc,random_samples,1.01  )
		samples[:,9] = fte.extract_likelihood( costncc,random_samples,self.__ncc_sigma )
		del costncc


		r_pkrn = fte.extract_ratio(costnccR,1.01)
		r_pkrn = np.reshape(r_pkrn,[h,w,ndisp])
		r_pkrn = fte.get_left_cost(r_pkrn)
		r_pkrn = np.reshape(r_pkrn, [ r_pkrn.shape[0]*r_pkrn.shape[1],r_pkrn.shape[2] ])
		r_pkrn = np.delete(r_pkrn, infs[:,0],axis=0)
		samples[:,13] = fte.get_samples( r_pkrn,random_samples  )
		del r_pkrn

		r_aml = fte.extract_likelihood(costnccR,self.__ncc_sigma )
		r_aml = np.reshape(r_aml,[h,w,ndisp])
		r_aml = fte.get_left_cost(r_aml)
		r_aml = np.reshape(r_aml, [ r_aml.shape[0]*r_aml.shape[1],r_aml.shape[2] ])
		r_aml = np.delete(r_aml, infs[:,0],axis=0)
		samples[:,17] = fte.get_samples( r_aml,random_samples)

		del r_aml
		del costnccR	



		######################################################################################
		############################### Sob compute ##########################################

		sobl = mtc.sobel(iml)
		sobr = mtc.sobel(imr)

		costsob = mtc.sadsob(sobl,sobr,ndisp,5).astype(np.float64)
		costsob = fte.swap_axes(costsob)
		costsobR = fte.get_right_cost(costsob)
		costsob = np.reshape(costsob, [ costsob.shape[0]*costsob.shape[1],costsob.shape[2] ])
		costsobR = np.reshape(costsobR, [ costsobR.shape[0]*costsobR.shape[1],costsobR.shape[2] ])
		
		costsob = np.delete(costsob, infs[:,0],axis=0)

		
		samples[:,2] = fte.get_samples(costsob , random_samples )
		samples[:,6] = fte.extract_ratio( costsob,random_samples,.01  )
		samples[:,10] = fte.extract_likelihood( costsob,random_samples,self.__sad_sigma )
		del costsob


		r_pkrn = fte.extract_ratio(costsobR,.01)
		r_pkrn = np.reshape(r_pkrn,[h,w,ndisp])
		r_pkrn = fte.get_left_cost(r_pkrn)
		r_pkrn = np.reshape(r_pkrn, [ r_pkrn.shape[0]*r_pkrn.shape[1],r_pkrn.shape[2] ])
		r_pkrn = np.delete(r_pkrn, infs[:,0],axis=0)
		samples[:,14] = fte.get_samples( r_pkrn,random_samples  )
		del r_pkrn

		r_aml = fte.extract_likelihood(costsobR,self.__sad_sigma )
		r_aml = np.reshape(r_aml,[h,w,ndisp])
		r_aml = fte.get_left_cost(r_aml)
		r_aml = np.reshape(r_aml, [ r_aml.shape[0]*r_aml.shape[1],r_aml.shape[2] ])
		r_aml = np.delete(r_aml, infs[:,0],axis=0)
		samples[:,18] = fte.get_samples( r_aml,random_samples)

		del r_aml
		del costsobR			



		######################################################################################
		############################### Sad compute ##########################################

		costsad = mtc.zsad(iml,imr,ndisp,self.__sadw).astype(np.float64)
		costsad = fte.swap_axes(costsad)
		costsadR = fte.get_right_cost(costsad)
		costsad = np.reshape(costsad, [ costsad.shape[0]*costsad.shape[1],costsad.shape[2] ])
		costsadR = np.reshape(costsadR, [ costsadR.shape[0]*costsadR.shape[1],costsadR.shape[2] ])

		costsad = np.delete(costsad, infs[:,0],axis=0)
		
		samples[:,3] = fte.get_samples(costsad , random_samples )
		samples[:,7] = fte.extract_ratio( costsad,random_samples,.01  )
		samples[:,11] = fte.extract_likelihood( costsad,random_samples,self.__sad_sigma )
		del costsad


		r_pkrn = fte.extract_ratio(costsadR,.01)
		r_pkrn = np.reshape(r_pkrn,[h,w,ndisp])
		r_pkrn = fte.get_left_cost(r_pkrn)
		r_pkrn = np.reshape(r_pkrn, [ r_pkrn.shape[0]*r_pkrn.shape[1],r_pkrn.shape[2] ])
		r_pkrn = np.delete(r_pkrn, infs[:,0],axis=0)
		samples[:,15] = fte.get_samples( r_pkrn,random_samples  )
		del r_pkrn

		r_aml = fte.extract_likelihood(costsadR,self.__sad_sigma )
		r_aml = np.reshape(r_aml,[h,w,ndisp])
		r_aml = fte.get_left_cost(r_aml)
		r_aml = np.reshape(r_aml, [ r_aml.shape[0]*r_aml.shape[1],r_aml.shape[2] ])
		r_aml = np.delete(r_aml, infs[:,0],axis=0)
		samples[:,19] = fte.get_samples( r_aml,random_samples)

		del r_aml
		del costsadR


		samples[:,20] =fte.generate_labels(random_samples)
		return samples



	def train_model(self,save_model,n_jobs=6):
		print "Creating training bank..."
		for i in range(0,len(self.__trainset)): 
			print "sampling " + self.__trainset[i]
			iml,imr,_ = self.__fix_rectification(i)
			self.__training_samples = np.append(self.__training_samples, self.__create_samples_mem(iml,imr,i),axis=0 )
	

		print "Number of training samples: " + str(self.__training_samples.shape)
		print "Training model"
		forest = RandomForestClassifier(min_samples_leaf=700,oob_score=False,n_jobs=n_jobs);
		forest.set_params(n_estimators=50,verbose=3);
		forest.fit(self.__training_samples[:,0:self.__training_samples.shape[1]-1],self.__training_samples[:,self.__training_samples.shape[1]-1]);

		with open(save_model,'wb') as f:
				cPickle.dump(forest,f);	
                                print ("model is saved at {}".format(save_model))
