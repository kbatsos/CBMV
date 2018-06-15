from __future__ import division
import sys
import math
import random

import scipy
from sklearn.feature_extraction import image
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
import cPickle;

import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation

import time

sys.path.insert(0,'./cpp/rectification/Debug')
sys.path.insert(0,'.')
sys.path.insert(0,'./pylibs')
sys.path.insert(0,'./cpp/matchers/Debug')
sys.path.insert(0,'./cpp/featextract/Debug')
sys.path.insert(0,'./cpp/post')

import librectification as rect
import pfmutil as pfm 
import libmatchers as mtc
import libfeatextract as fte
import progressbar as pgb
import post

class Testing(object):

	def __init__(self,t_set="", data_path="", iml="",imr="",calib="",
					w=0,h=0,d=0,
					censw=11, nccw=3,sadw=5,pi1=1.3,pi2=10,sgm_q1=2.2,sgm_q2=3,alpha1=1.4,tau_so=.12,sgm_i=2,
					L1=6,tau1=.07,cbca_i1=0,cbca_i2=4,median_i=1,median_w=3,blur_sigma=2.2,blur_t=1.3):

		self.__censw = censw
		self.__nccw = nccw
		self.__sadw =sadw
		self.__testset = t_set
		self.__data_path = data_path
		self.__feature_vol = np.empty((0,16))
		self.__test_batch = 100000
		self.__pi1 = pi1 #0.8 #0.8 #2.6 #2.6 #1.3 #1.3
		self.__pi2 = pi2 #8.9 #8.9 #12.6 #13.9 #13.9
		self.__sgm_q1 = sgm_q1 #1.9#1.9 #4.5 #3
		self.__sgm_q2 = sgm_q2 #1.9#1.9 #2 
		self.__alpha1 = alpha1 #1.5 #0.6 #2.75 #1.1 
		self.__tau_so = tau_so #10 #0.14
		self.__sgm_i = sgm_i#2 #3		 
		self.__L1 = L1 #14 #14#14
		self.__tau1 = tau1 #10
		self.__cbca_i1 =cbca_i1
		self.__cbca_i2 = cbca_i2
		self.__median_i = median_i
		self.__median_w = median_w #5
		self.__blur_sigma = blur_sigma #0.9 #0.9 #1.8
		self.__blur_t = blur_t #4.7 #1.3	
		self.__cens_sigma=2*8**2
		self.__ncc_sigma = 0.02
		self.__sad_sigma = 2*100**2	
		if h==0 or w==0 or d==0:
			if calib=="":
				self.__calibfile = self.__data_path + self.__testset+"/calib.txt"
			else:
				self.__calibfile = self.__data_path+calib 
			self.w,self.h,self.d = self.__read_calib()
		else:
			self.w=int(w)
			self.h=int(h)
			self.d=int(d)	
		if iml=="" and imr=="":	
			self.imgl,self.imgr = self.__fix_rectification(self.__data_path + self.__testset+"/im0.png", self.__data_path + self.__testset+"/im1.png" )
		else:
			self.imgl,self.imgr = self.__fix_rectification(self.__data_path +iml, self.__data_path + imr )
		



	def __read_calib(self):
		w=0;
		h=0;
		d=0;
		with open(self.__calibfile) as f:
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

	def __gaussian(self,sigma):
		kr = math.ceil(sigma*3)
		ks = int(kr*2+1)
		k = np.zeros((ks,ks))
		for i in range(0,ks):
			for j in range(0,ks):
				y = (i-1)-kr
				x = (j-1)-kr
				k[i,j] = math.exp( - (x*x+y*y)/ (2*sigma*sigma) )

		return k.astype(np.float32)



	def __get_costs(self,iml,imr):
		costcensus = mtc.census(iml,imr,self.d,self.__censw).astype(np.float64)		

		costncc = mtc.nccNister(iml,imr,self.d,self.__nccw)
		costncc = fte.swap_axes(costncc)

		sobl = mtc.sobel(iml)
		sobr = mtc.sobel(imr)

		costsob = mtc.sadsob(sobl,sobr,self.d,5).astype(np.float64)
		costsob = fte.swap_axes(costsob)		
		
		costsad = mtc.zsad(iml,imr,self.d,self.__sadw).astype(np.float64)	
		costsad = fte.swap_axes(costsad)


		return costcensus, costncc, costsob, costsad	




	def eval_prob(self,prob,disp_save_path,display=False,interpolate=False):
		prob =  np.reshape(prob, [self.h,self.w,self.d,2])	
		prob = fte.get_cost(prob)	
		probr = fte.get_right_cost(prob)		

		self.do_post(prob,probr,disp_save_path,display,interpolate)



	def __postprocessing_mem_interp(self, imgl,imgr,cost,direction,display=False):

		
		r_cost = fte.get_right_cost(cost)
		post.InitCUDA()	
		start = time.time()

		cross_array_init = np.zeros((4,imgl.shape[0],imgl.shape[1])).astype(np.float32)
		imgl_d = post.CUDA_float_array(imgl)
		imgr_d = post.CUDA_float_array(imgr)
			
		rightc = np.zeros((4, imgr.shape[0], imgr.shape[1])).astype(np.float32)
		leftc_d = post.CUDA_float_array(cross_array_init)
		rightc_d = post.CUDA_float_array(cross_array_init)
		
		post.cross( imgl_d,leftc_d, self.__L1, self.__tau1 )
		post.cross( imgr_d, rightc_d,self.__L1, self.__tau1 )


		###compute right disparity 	
		r_cost_d = post.CUDA_double_array(r_cost)
		temp_cost_d = post.CUDA_double_array(cost)
		tmp = np.zeros((cost.shape[1],cost.shape[2]))
		tmp_d = post.CUDA_double_array(tmp)

		post.swap_axis_back(r_cost_d,temp_cost_d)
		cc = r_cost_d.get_array();		
		post.CUDA_copy_double(temp_cost_d,r_cost_d)	

		for i in range(0,self.__cbca_i1):
			post.cbca( rightc_d,leftc_d,r_cost_d, temp_cost_d,-1*direction )
			post.CUDA_copy_double(temp_cost_d,r_cost_d)

		if display :
			print "CBCA 1 right"
			cc = r_cost_d.get_array();
			ddisp = np.argmin(cc,axis =0)
			pfm.show(ddisp)			
		
		post.swap_axis(r_cost_d,temp_cost_d)	
		post.CUDA_copy_double(temp_cost_d,r_cost_d)
		

		for i in range(0,self.__sgm_i):
			temp_cost_d.set_to_zero()
			tmp_d.set_to_zero()
			post.sgm(imgl_d,imgr_d,r_cost_d,temp_cost_d,tmp_d, self.__pi1, self.__pi2, self.__tau_so, self.__alpha1, self.__sgm_q1, self.__sgm_q2, -1*direction)
			post.CUDA_divide_double(temp_cost_d,4)
			post.CUDA_copy_double(temp_cost_d,r_cost_d)	

		post.swap_axis_back(r_cost_d,temp_cost_d)
		post.CUDA_copy_double(temp_cost_d,r_cost_d)	

		if display :
			print "SGM right"
			cc = r_cost_d.get_array()
			ddisp = np.argmin(cc,axis =0)
			pfm.show(ddisp)			

		for i in range(0,self.__cbca_i2):
			post.cbca( rightc_d,leftc_d, r_cost_d, temp_cost_d ,-1*direction )

			post.CUDA_copy_double(temp_cost_d,r_cost_d)

		if display :
			print "CBCA2 right"
			cc = r_cost_d.get_array()
			ddisp = np.argmin(cc,axis =0)
			pfm.show(ddisp)			

		del temp_cost_d
		cc =r_cost_d.get_array()
		r_disp = np.argmin(cc,axis=0).astype(np.float32)
		del cc 
		del r_cost_d


		r_disp_d = post.CUDA_float_array(r_disp)	


		cost_d = post.CUDA_double_array(cost)	
		temp_cost_d = post.CUDA_double_array(cost)

		post.swap_axis_back(cost_d,temp_cost_d)
		post.CUDA_copy_double(temp_cost_d,cost_d)

		for i in range(0,self.__cbca_i1):
			post.cbca( leftc_d,rightc_d,cost_d, temp_cost_d,direction )
			post.CUDA_copy_double(temp_cost_d,cost_d)
		if display :
			print "CBCA 1"
			cc = cost_d.get_array();
			ddisp = np.argmin(cc,axis =0)
			pfm.show(ddisp)


		post.swap_axis(cost_d,temp_cost_d)	
		post.CUDA_copy_double(temp_cost_d,cost_d)

		tmp = np.zeros((cost.shape[1],cost.shape[2]))
		tmp_d = post.CUDA_double_array(tmp)

		for i in range(0,self.__sgm_i):
			temp_cost_d.set_to_zero()
			tmp_d.set_to_zero()
			post.sgm(imgl_d,imgr_d,cost_d,temp_cost_d,tmp_d, self.__pi1, self.__pi2, self.__tau_so, self.__alpha1, self.__sgm_q1, self.__sgm_q2, direction)
			post.CUDA_divide_double(temp_cost_d,4)
			post.CUDA_copy_double(temp_cost_d,cost_d)
		

		post.swap_axis_back(cost_d,temp_cost_d)
		post.CUDA_copy_double(temp_cost_d,cost_d)

		if display :
			print "SGM"
			cc = cost_d.get_array()
			ddisp = np.argmin(cc,axis =0)
			pfm.show(ddisp)


		for i in range(0,self.__cbca_i2):
			post.cbca( leftc_d,rightc_d, cost_d, temp_cost_d ,direction )

			post.CUDA_copy_double(temp_cost_d,cost_d)

		if display :
			print "CBCA2"
			cc = cost_d.get_array()
			ddisp = np.argmin(cc,axis =0)
			pfm.show(ddisp)


		cc = cost_d.get_array()	
		disp_l = np.argmin(cc,axis=0).astype(np.float32)
		out_s = cc.shape[0]
		del cc 

		disp_d = post.CUDA_float_array(disp_l)	
		disp_temp = post.CUDA_float_array(disp_l)

		post.subpixel_enchancement(  disp_d, cost_d,disp_temp )

		outlier = np.zeros((r_disp.shape[0],r_disp.shape[1])).astype(np.float32)
		outlier_d = post.CUDA_float_array(outlier)
		post.outlier_detection( disp_d,r_disp_d,outlier_d,out_s )

		post.interpolate_mismatch(disp_d,outlier_d,disp_temp)
		post.CUDA_copy_float(disp_temp,disp_d)

		post.interpolate_occlusion(disp_d,outlier_d,disp_temp)
		post.CUDA_copy_float(disp_temp,disp_d)


	
		post.CUDA_copy_float(disp_temp,disp_d)
		if display :
			disp = disp_d.get_array()
			print "Subpixel"
			pfm.show(disp)


		for i in range(0, self.__median_i):		
			disp = post.median2d(  disp_d,disp_temp, self.__median_w )
			post.CUDA_copy_float(disp_temp,disp_d)	

		
		if display :
			disp = disp_d.get_array()
			print "Median"
			pfm.show(disp)
	
		post.mean2d( disp_d,disp_temp , self.__gaussian( self.__blur_sigma ), self.__blur_t )
		post.CUDA_copy_float(disp_temp,disp_d)	
		
		if display :
			disp = disp_d.get_array()
			print "Bilateral"
			pfm.show(disp)

		disp = disp_d.get_array()	

		
		end = time.time()
		print(end - start)

		return disp


	def __postprocessing_mem(self, imgl,imgr,cost,direction,display=False):

		post.InitCUDA()	
		start = time.time()

		cross_array_init = np.zeros((4,imgl.shape[0],imgl.shape[1])).astype(np.float32)
		imgl_d = post.CUDA_float_array(imgl)
		imgr_d = post.CUDA_float_array(imgr)
			
		leftc_d = post.CUDA_float_array(cross_array_init)
		rightc_d = post.CUDA_float_array(cross_array_init)
		
		post.cross( imgl_d,leftc_d, self.__L1, self.__tau1 )
		post.cross( imgr_d, rightc_d,self.__L1, self.__tau1 )


		cost_d = post.CUDA_double_array(cost)	
		temp_cost_d = post.CUDA_double_array(cost)

		post.swap_axis_back(cost_d,temp_cost_d)
		post.CUDA_copy_double(temp_cost_d,cost_d)

		for i in range(0,self.__cbca_i1):
			post.cbca( leftc_d,rightc_d,cost_d, temp_cost_d,direction )
			post.CUDA_copy_double(temp_cost_d,cost_d)
		if display :
			print "CBCA 1"
			cc = cost_d.get_array();
			ddisp = np.argmin(cc,axis =0)
			pfm.show(ddisp)

		post.swap_axis(cost_d,temp_cost_d)	
		post.CUDA_copy_double(temp_cost_d,cost_d)

		tmp = np.zeros((cost.shape[1],cost.shape[2]))
		tmp_d = post.CUDA_double_array(tmp)

		for i in range(0,self.__sgm_i):
			temp_cost_d.set_to_zero()
			tmp_d.set_to_zero()
			post.sgm(imgl_d,imgr_d,cost_d,temp_cost_d,tmp_d, self.__pi1, self.__pi2, self.__tau_so, self.__alpha1, self.__sgm_q1, self.__sgm_q2, direction)
			post.CUDA_divide_double(temp_cost_d,4)
			post.CUDA_copy_double(temp_cost_d,cost_d)
		

		post.swap_axis_back(cost_d,temp_cost_d)
		post.CUDA_copy_double(temp_cost_d,cost_d)

		if display :
			print "SGM"
			cc = cost_d.get_array()
			ddisp = np.argmin(cc,axis =0)
			pfm.show(ddisp)


		for i in range(0,self.__cbca_i2):
			post.cbca( leftc_d,rightc_d, cost_d, temp_cost_d ,direction )

			post.CUDA_copy_double(temp_cost_d,cost_d)

		if display :
			print "CBCA2"
			cc = cost_d.get_array()
			ddisp = np.argmin(cc,axis =0)
			pfm.show(ddisp)


		cc = cost_d.get_array()	
		disp_l = np.argmin(cc,axis=0).astype(np.float32)
		out_s = cc.shape[0]
		del cc 

		disp_d = post.CUDA_float_array(disp_l)	
		disp_temp = post.CUDA_float_array(disp_l)

		post.subpixel_enchancement(  disp_d, cost_d,disp_temp )
	
		post.CUDA_copy_float(disp_temp,disp_d)
		if display :
			disp = disp_d.get_array()
			print "Subpixel"
			pfm.show(disp)


		for i in range(0, self.__median_i):		
			disp = post.median2d(  disp_d,disp_temp, self.__median_w )
			post.CUDA_copy_float(disp_temp,disp_d)	

		
		if display :
			disp = disp_d.get_array()
			print "Median"
			pfm.show(disp)

		post.mean2d( disp_d,disp_temp , self.__gaussian( self.__blur_sigma ), self.__blur_t )
		post.CUDA_copy_float(disp_temp,disp_d)	
		
		if display :
			disp = disp_d.get_array()
			print "Bilateral"
			pfm.show(disp)

		disp = disp_d.get_array()	

		
		end = time.time()
		print(end - start)

		return disp




	def do_post(self,lcost,rcost,save_path,display,interpolate=False ):
		imgl = self.imgl.astype(np.float32)
		imgr = self.imgr.astype(np.float32)
		imgl = (imgl-np.mean(imgl))/np.std(imgl)
		imgr = (imgr-np.mean(imgr))/np.std(imgr)
		if(interpolate):
			displ = self.__postprocessing_mem_interp( imgl,imgr, lcost,-1,display)
		else:
			displ = self.__postprocessing_mem( imgl,imgr, lcost,-1,display)
		
		print "Saving"
		pfm.save( save_path,displ.astype(np.float32))

	def __fix_rectification(self,lpath,rpath):
		imgl = scipy.misc.imread( lpath,mode='L' );
		imgr = scipy.misc.imread( rpath,mode='L' );
			
		h_l = np.zeros((3,3)).astype(np.float32)
		h_r = np.zeros((3,3)).astype(np.float32)

		# returns rectified images and homographies in place
		rect.fixrectification(imgl,imgr,h_l,h_r,)
	
		return imgl,imgr


	def __extract_features_lr(self,census,ncc,sobel,sad):

		dims = census.shape

		censusr = fte.get_right_cost(census)
		census =  np.reshape(census, [ dims[0]*dims[1],dims[2] ])
		censusr =  np.reshape(censusr, [ dims[0]*dims[1],dims[2] ])

		nccr = fte.get_right_cost(ncc)
		ncc =  np.reshape(ncc, [ dims[0]*dims[1],dims[2] ])
		nccr =  np.reshape(nccr, [ dims[0]*dims[1],dims[2] ])	

		sobelr = fte.get_right_cost(sobel)
		sobel =  np.reshape(sobel, [ dims[0]*dims[1],dims[2] ])
		sobelr =  np.reshape(sobelr, [ dims[0]*dims[1],dims[2] ])

		sadr = fte.get_right_cost(sad)
		sad =  np.reshape(sad, [ dims[0]*dims[1],dims[2] ])
		sadr =  np.reshape(sadr, [ dims[0]*dims[1],dims[2] ])						

		features = np.empty((dims[0]*dims[1]*dims[2],20))

		features[:,0]=np.reshape(census,[dims[0]*dims[1]*dims[2]])
		features[:,1]=np.reshape(ncc,[dims[0]*dims[1]*dims[2]])
		features[:,2]=np.reshape(sobel,[dims[0]*dims[1]*dims[2]])
		features[:,3]=np.reshape(sad,[dims[0]*dims[1]*dims[2]])

		features[:,4]=np.reshape( fte.extract_ratio( census,.01  ),[ dims[0]*dims[1]*dims[2]  ] )
		features[:,5]=np.reshape( fte.extract_ratio( ncc,1.01  ),[ dims[0]*dims[1]*dims[2]  ] )
		features[:,6]=np.reshape( fte.extract_ratio( sobel,.01  ),[ dims[0]*dims[1]*dims[2]  ] )
		features[:,7]=np.reshape( fte.extract_ratio( sad,.01  ),[ dims[0]*dims[1]*dims[2]  ] )

		features[:,8]=np.reshape( fte.extract_likelihood( census,self.__cens_sigma ),[ dims[0]*dims[1]*dims[2]  ] )
		features[:,9]=np.reshape( fte.extract_likelihood( ncc,self.__ncc_sigma ),[ dims[0]*dims[1]*dims[2]  ] )
		features[:,10]=np.reshape( fte.extract_likelihood( sobel,self.__sad_sigma ),[ dims[0]*dims[1]*dims[2]  ] )
		features[:,11]=np.reshape( fte.extract_likelihood( sad,self.__sad_sigma ),[ dims[0]*dims[1]*dims[2]  ] )

		r_pkrn = fte.extract_ratio(censusr,.01)
		r_pkrn = np.reshape(r_pkrn,[dims[0],dims[1],dims[2]])
		features[:,12] = np.reshape( fte.get_left_cost(r_pkrn) ,[ dims[0]*dims[1]*dims[2]  ] )

		r_aml = fte.extract_likelihood(censusr,self.__cens_sigma )
		r_aml = np.reshape(r_aml,[dims[0],dims[1],dims[2]])
		features[:,16] = np.reshape( fte.get_left_cost(r_aml) ,[ dims[0]*dims[1]*dims[2]  ] )
		del censusr

		r_pkrn = fte.extract_ratio(nccr,1.01)
		r_pkrn = np.reshape(r_pkrn,[dims[0],dims[1],dims[2]])
		features[:,13] = np.reshape( fte.get_left_cost(r_pkrn) ,[ dims[0]*dims[1]*dims[2]  ] )

		r_aml = fte.extract_likelihood(nccr,self.__ncc_sigma )
		r_aml = np.reshape(r_aml,[dims[0],dims[1],dims[2]])
		features[:,17] = np.reshape( fte.get_left_cost(r_aml) ,[ dims[0]*dims[1]*dims[2]  ] )
		del nccr
		
		r_pkrn = fte.extract_ratio(sobelr,.01)
		r_pkrn = np.reshape(r_pkrn,[dims[0],dims[1],dims[2]])
		features[:,14] = np.reshape( fte.get_left_cost(r_pkrn) ,[ dims[0]*dims[1]*dims[2]  ] )

		r_aml = fte.extract_likelihood(sobelr,self.__sad_sigma )
		r_aml = np.reshape(r_aml,[dims[0],dims[1],dims[2]])
		features[:,18] = np.reshape(  fte.get_left_cost(r_aml) ,[ dims[0]*dims[1]*dims[2]  ] )
		del sobelr


		r_pkrn = fte.extract_ratio(sadr,.01)
		r_pkrn = np.reshape(r_pkrn,[dims[0],dims[1],dims[2]])
		features[:,15] = np.reshape( fte.get_left_cost(r_pkrn) ,[ dims[0]*dims[1]*dims[2]  ] )

		r_aml = fte.extract_likelihood(sadr,self.__sad_sigma )
		r_aml = np.reshape(r_aml,[dims[0],dims[1],dims[2]])
		features[:,19] = np.reshape( fte.get_left_cost(r_aml) ,[ dims[0]*dims[1]*dims[2]  ] )
		del sadr

		return features
	
	def test_model(self,saved_model,prob_save_path):
		census,ncc,sobel,sad = self.__get_costs(self.imgl,self.imgr)

		print "Batch testing"
		proba = np.empty((0,2))

		with open(saved_model,'rb') as f:
			rf = cPickle.load(f);

		rf.set_params(verbose=0)

		r_s = int(math.floor(self.__test_batch/self.w))

		batch_index = 0
		print "Iterations: " + str( math.ceil(self.h/r_s) )
		pgb.printProgressBar(0,math.ceil(self.h/r_s), prefix='Progress', suffix='Complete', length=50 )
		while batch_index <self.h :

			features = self.__extract_features_lr(census[batch_index:batch_index+r_s,:,:],
												  ncc[batch_index:batch_index+r_s,:,:],
												  sobel[batch_index:batch_index+r_s,:,:],
												  sad[batch_index:batch_index+r_s,:,:])
			
			batch_proba = rf.predict_proba(features);
			batch_index += r_s
			proba = np.append(proba,batch_proba,axis=0)
			pgb.printProgressBar(batch_index/r_s,math.ceil(self.h/r_s), prefix='Progress', suffix='Complete', length=50 )	
		
		rf =[]
		return proba		
		


				
		
		

