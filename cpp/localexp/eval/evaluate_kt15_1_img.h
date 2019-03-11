//This file is added by CCJ;
#pragma once

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <vector>

#include "io_disp.h"
#include "io_integer.h"
#include "utils.h"
//added by CCJ
#include "pfm_rw.h"

using namespace std;

//****************
// declaration:
//****************
bool eval_kt2015 ( const string & imgDir, float * p_disp, float * err_result, 
		 int viewMode = 0 // left view = 0, right view = 1
		 );


std::vector<float> disparityErrorsOutlier_kt15 (DisparityImage &D_gt, 
		DisparityImage &D_orig,
		DisparityImage &D_ipol,
		IntegerImage &O_map //object map (0:background, >0:foreground)
		);

void pfm2uint16PNG(	string pfm_result_file, string disp_suf, 	string png_result_file,	int imgNum); 
