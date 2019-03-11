#include "evaluate_kt15_1_img.h"

#define ABS_THRESH 3.0
#define REL_THRESH 0.05

//**************************************************
//*** disparity from pfm to png, kt 2012/2015 ******
//**************************************************
void pfm2uint16PNG(
		string baseDir, 
		string pfm_result_file,// e.g., == "kitti2012-pfm-submit01/"
		string disp_suf, // e.g., == "_post_disp0PKLS"
		string png_result_file,// e.g., == "kitti2012-png-submit01/"
		int imgNum
		){ 
  
  string pfm_result_dir =  baseDir + "results/" + pfm_result_file;
  string png_result_dir =  baseDir + "results/" + png_result_file;
  PFM pfmIO;
  // for all test files do
  for (int32_t i=0; i < imgNum; i++){
    // file name
    char prefix[256];
    sprintf(prefix,"%06d_10",i);
    
		string pfm_image_file = pfm_result_dir + "/" + prefix + disp_suf + ".pfm";
		float * p_disp = pfmIO.read_pfm<float>(pfm_image_file);
		//cout << "Done ! reading pfm disparity: " << prefix + disp_suffix + ".pfm\n";

		// construct disparity image from data
		DisparityImage D_pfm( p_disp, pfmIO.getWidth(), pfmIO.getHeight());
		string png_image_file = png_result_dir + "/" + prefix + ".png";
		D_pfm.write(png_image_file);
		delete[] p_disp;
	}
}



//*****************************
//*** disparity, kt 2015 ******
//*****************************
vector<float> disparityErrorsOutlier_kt15 (
		DisparityImage &D_gt,
		DisparityImage &D_orig,
		DisparityImage &D_ipol,
		IntegerImage &O_map //object map (0:background, >0:foreground)
		){

  // check file size
  if (D_gt.width()!=D_orig.width() || D_gt.height()!=D_orig.height()) {
    cout << "ERROR: Wrong file size!" << endl;
    throw 1;
  }

  // extract width and height
  int32_t width  = D_gt.width();
  int32_t height = D_gt.height();

  // init errors
  vector<float> errors;
  int32_t num_errors_bg = 0; // nubmer of outliers in bg
  int32_t num_pixels_bg = 0;// number of pixels in bg
  int32_t num_errors_bg_result = 0;
  int32_t num_pixels_bg_result = 0;
  int32_t num_errors_fg = 0;
  int32_t num_pixels_fg = 0;
  int32_t num_errors_fg_result = 0;
  int32_t num_pixels_fg_result = 0;
  int32_t num_errors_all = 0;
  int32_t num_pixels_all = 0;
  int32_t num_errors_all_result = 0;
  int32_t num_pixels_all_result = 0;

  // for all pixels do
  for (int32_t u=0; u<width; u++) {
    for (int32_t v=0; v<height; v++) {
      if (D_gt.isValid(u,v)) {
        float d_gt  = D_gt.getDisp(u,v);
        float d_est = D_ipol.getDisp(u,v);
				// both absolute error and relative error;
        bool  d_err = fabs(d_gt-d_est)>ABS_THRESH && fabs(d_gt-d_est)/fabs(d_gt)>REL_THRESH;
        

				if (O_map.getValue(u,v)==0) {// 0: background (bg);
          if (d_err)
            num_errors_bg++;
          num_pixels_bg++;
          if (D_orig.isValid(u,v)) {// your calculated disparity
            if (d_err)
              num_errors_bg_result++;
            num_pixels_bg_result++;
          }
        }// end of bg;
				
				else { // > 0: foreground (fg);
          if (d_err)
            num_errors_fg++;
          num_pixels_fg++;
          if (D_orig.isValid(u,v)) {
            if (d_err)
              num_errors_fg_result++;
            num_pixels_fg_result++;
          }
        }

        if (d_err)
          num_errors_all++;
        num_pixels_all++;
        if (D_orig.isValid(u,v)){
          if (d_err)
            num_errors_all_result++;
          num_pixels_all_result++;
        }

      }// end of valid pixels with ground truth
    }
  }
  // push back errors and pixel count
  errors.push_back(num_errors_bg);
  errors.push_back(num_pixels_bg);
  errors.push_back(num_errors_bg_result);
  errors.push_back(num_pixels_bg_result);
  errors.push_back(num_errors_fg);
  errors.push_back(num_pixels_fg);
  errors.push_back(num_errors_fg_result);
  errors.push_back(num_pixels_fg_result);
  errors.push_back(num_errors_all);
  errors.push_back(num_pixels_all);
  errors.push_back(num_errors_all_result);
  errors.push_back(num_pixels_all_result);

  // push back density
  errors.push_back((float)num_pixels_all_result/max((float)num_pixels_all,1.0f));

  // return errors
  return errors;
}


//***********************************
//*** evaluate stereo, kt 2015 ******
//***********************************
bool eval_kt2015 (
		const string & imgDir, // e.g., == "/media/ccjData2/GCNet/datasets/KT15/training/000000_10/" 
		//const int & heiht, // image height;
		//const int & width, // image width;
		float * p_disp, // the estimated disparity; 
		float * err_result,
	  int viewMode // left view = 0, right view = 1	
		){

  // ground truth and result directories
	// left_viw : "im0.png"; right_view : "im1.png";
  string gt_img = imgDir + "im" + std::to_string(viewMode) + ".png";
  string gt_obj_map = imgDir + "obj_map.png";
  string gt_disp_noc_0 = imgDir + "disp" + std::to_string(viewMode) + "GT_noc.pfm";
  string gt_disp_occ_0 = imgDir + "disp" + std::to_string(viewMode) + "GT_occ.pfm";

#if 0
  std::cout << "gt_img = " << gt_img << "\n"
            << "gt_obj_map = " << gt_obj_map << "\n"
            << "gt_disp_noc_0 = " << gt_disp_noc_0 << "\n"
            << "gt_disp_occ_0 = " << gt_disp_occ_0 << "\n";
#endif      	
	//PFM pfmIO;
	// declaration of global data structures
	DisparityImage D_gt_noc_0, D_gt_occ_0;
	// load object map (0:background, >0:foreground)
	IntegerImage O_map = IntegerImage(gt_obj_map);
	// construct disparity image from data
  PFM pfmIO;
	float * p_disp_gt_tmp = pfmIO.read_pfm<float>(gt_disp_noc_0);
  const int width = pfmIO.getWidth(); 
  const int height = pfmIO.getHeight(); 
	D_gt_noc_0 = DisparityImage(p_disp_gt_tmp, width, height);
	p_disp_gt_tmp = pfmIO.read_pfm<float>(gt_disp_occ_0);
	D_gt_occ_0 = DisparityImage(p_disp_gt_tmp, width, height);
	
	// load submitted result and interpolate missing values
	//DisparityImage D_orig_0(p_disp, pfmIO.getWidth(), pfmIO.getHeight());
	DisparityImage D_orig_0(p_disp, width, height);
	DisparityImage D_ipol_0 = DisparityImage(D_orig_0);
	D_ipol_0.interpolateBackground();

	// calculate disparity errors
	vector<float> errors_disp_noc_0 = disparityErrorsOutlier_kt15(D_gt_noc_0, D_orig_0, D_ipol_0, O_map);
	vector<float> errors_disp_occ_0 = disparityErrorsOutlier_kt15(D_gt_occ_0, D_orig_0, D_ipol_0, O_map);

	// saving result.
	/*
	char * notes[] = {
		"noc-bg  (all       pixels)",
		"noc-bg  (estimated pixels)",
		"noc-fg  (all       pixels)",
		"noc-fg  (estimated pixels)",
		"noc-all (all       pixels)",
		"noc-all (estimated pixels)",
		"noc-all (         density)",
		
		"occ-bg  (all       pixels)",
		"occ-bg  (estimated pixels)",
		"occ-fg  (all       pixels)",
		"occ-fg  (estimated pixels)",
		"occ-all (all       pixels)",
		"occ-all (estimated pixels)",
		"ooc-all (         density)"
	};
	*/

	err_result[0] = errors_disp_noc_0[0]/max(errors_disp_noc_0[1],1.0f); // noc, bg (all       pixels)
	err_result[1] = errors_disp_noc_0[2]/max(errors_disp_noc_0[3],1.0f); // noc, bg (estimated pixels)
	err_result[2] = errors_disp_noc_0[4]/max(errors_disp_noc_0[5],1.0f); // noc, fg (all       pixels)
	err_result[3] = errors_disp_noc_0[6]/max(errors_disp_noc_0[7],1.0f); // noc, fg (estimated pixels)
	err_result[4] = errors_disp_noc_0[8]/max(errors_disp_noc_0[9],1.0f); // noc,all (all       pixels)
	err_result[5] = errors_disp_noc_0[10]/max(errors_disp_noc_0[11],1.0f);//noc,all (estimated pixels)
	err_result[6] = errors_disp_noc_0[11]/max(errors_disp_noc_0[9],1.0f); //noc,all (density)
	
	err_result[7] = errors_disp_occ_0[0]/ max(errors_disp_occ_0[1],1.0f); // occ, bg (all       pixels)
	err_result[8] = errors_disp_occ_0[2]/ max(errors_disp_occ_0[3],1.0f); // occ, bg (estimated pixels)
	err_result[9] = errors_disp_occ_0[4]/ max(errors_disp_occ_0[5],1.0f); // occ, fg (all       pixels)
	err_result[10]= errors_disp_occ_0[6]/ max(errors_disp_occ_0[7],1.0f); // occ, fg (estimated pixels)
	err_result[11]= errors_disp_occ_0[8]/ max(errors_disp_occ_0[9],1.0f); // occ,all (all       pixels)
	err_result[12]= errors_disp_occ_0[10]/max(errors_disp_occ_0[11],1.0f);//occ,all (estimated pixels)
	err_result[13]= errors_disp_occ_0[11]/max(errors_disp_occ_0[9],1.0f); //occ,all (density)

  // success
	// printf("Successfully evaluated %d images!!\n", imgIdxEnd - imgIdxStart);
	return true;
}

