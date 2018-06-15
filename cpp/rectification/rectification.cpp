/*
 * rectification.cpp
 *
 *  Created on: Jun 5, 2017
 *      Author: kbatsos
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include <numpy/ndarrayobject.h>
#include <iostream>
#include <cstdio>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <sstream>
#include <fstream>
#include <dirent.h>
using namespace std;
using namespace cv;
using namespace boost::python;

typedef uint8_t uint8;


bool isEmptyModel(cv::Mat& M){

        if (M.rows == 3) {
                float a = M.at<float>(0,0);
                float b = M.at<float>(1,0);
                float c = M.at<float>(2,0);
                if (a == 0 && b == 0 && c == 0){
                        return true;
                }
                else{
                        return false;
                }
        }
        else{
                float a = M.at<float>(0,0);
                float b = M.at<float>(1,0);
                if (a == 0 && b == 0){
                        return true;
                }
                else{
                        return false;
                }
        }
}

int buildModel_ylinear(std::vector<Point2f>& pair1, std::vector<Point2f>& pair2, cv::Mat& M){

        float y1 = pair1[0].y;
        float y2 = pair2[0].y;

        if ((y1 - y2) < FLT_EPSILON){
                return 0;
        }

        float d1, d2;
        d1 = (pair1[0].y + pair1[1].y)/2.0; //y avg pair1
        d2 = (pair2[0].y + pair2[1].y)/2.0; //y avg pair2

        float m = (d1-d2)/(y1-y2);
        M.at<float>(0,0) = m;
        M.at<float>(1,0) = d1 - m*y1;

        return 1;
}

void findInlierFeaturePairs_ylinear(cv::Mat& M, std::vector<std::vector<Point2f> >& leftPts_rightPts,
                std::vector<std::vector<Point2f> >& inliersTmp, float t){
        //for each pair in leftPts_rightPts, check if residual < t
        std::vector<Point2f> tmp;
        float y_left, y_right, y_target, y_model;
        for (unsigned int i = 0; i < leftPts_rightPts.size(); i++){
                y_left = leftPts_rightPts[i][0].y;
                y_right = leftPts_rightPts[i][1].y;

                y_target = (y_left + y_right)/2.0;
                y_model = (M.at<float>(0,0)*y_left) + M.at<float>(1,0);

                if (fabs(y_target - y_model) <= t){ //within error tolerance
                        std::vector<Point2f> tmp;
                        tmp.push_back(leftPts_rightPts[i][0]);
                        tmp.push_back(leftPts_rightPts[i][1]);
                        inliersTmp.push_back(tmp);
                }
        }
}

void RANSACFitTransform_ylinear( std::vector<std::vector<Point2f> >& leftPts_rightPts,
                std::vector<std::vector<Point2f> >& inliers, cv::Mat& returnModel, unsigned int N, float t, bool isAdaptive){

        unsigned int maxTrials = 1000;
        unsigned int maxDataTrials = 1000;
        unsigned int numPoints = leftPts_rightPts.size();

        float p = 0.99;

        cv::Mat bestM = cv::Mat::zeros(2,1, CV_32F);
        cv::Mat M = cv::Mat::zeros(2,1, CV_32F);

        unsigned int trialCount = 0;
        unsigned int dataTrialCount = 0;
        unsigned int ninliers = 0;
        float bestScore = 0.0;
        float fracinilers = 0.0;
        float pNoOutliers = 0.0;

        int degenerate = 0; //default
        std::vector<Point2f> pair1;
        std::vector<Point2f> pair2;



        if (isAdaptive){
                N = 1; //dummy initialization for adaptive termination RANSAC
        }

        while (N > trialCount){

                degenerate = 0; //default- singular.
                dataTrialCount = 1;

                while (degenerate == 0){//

                        std::random_shuffle(leftPts_rightPts.begin(), leftPts_rightPts.end());

                        pair1 = leftPts_rightPts[0];
                        pair2 = leftPts_rightPts[1];

                        //return 1 if non-singular, OK model. return 0 if singular (degenerate model)

                        degenerate = buildModel_ylinear(pair1, pair2, M);

                        dataTrialCount++;
                        if ( dataTrialCount > maxDataTrials){
                                cout << "Unable to select a non-degenerate data set." << endl;
                                return;
                        }
                }

                //now we know that M contains some type of non-degenerate model,
                std::vector<std::vector<Point2f> > inliersTmp;
                findInlierFeaturePairs_ylinear(M, leftPts_rightPts, inliersTmp, t);
                ninliers = inliersTmp.size();

                if (ninliers > bestScore){
                        bestScore = ninliers;
                        inliers = inliersTmp;
                        bestM = M;
                        //if adaptive termination RANSAC, update estimate of N
                        if (isAdaptive){
                                fracinilers = float(ninliers)/float(numPoints);
                                pNoOutliers = 1 - pow(fracinilers,3); //3 for three points selected
                                pNoOutliers = max(FLT_EPSILON, pNoOutliers);
                                pNoOutliers = min(1-FLT_EPSILON, pNoOutliers);
                                N = (unsigned int) round(log(1-p)/log(pNoOutliers));
                        }
                }

                trialCount++;
                if (trialCount > maxTrials){
                        break;
                }
        }



        if (!isEmptyModel(bestM)){ //found a solution
                returnModel = bestM;
        }
        else{
                cout << "RANSAC was unable to find a useful solution." << endl;
        }
}


void compute_y_offset_ylinear(cv::Mat& leftImageG, cv::Mat& rightImageG, cv::Mat& left_warped, cv::Mat& right_warped,
                bool displayOn, double w, double h, cv::Mat& H_l, cv::Mat& H_r){

        std::vector<KeyPoint> keypoints_left, keypoints_right;
        std::vector<std::vector<Point2f> > leftPts_rightPts;
        std::vector<std::vector<Point2f> > inliers;




		int minHessian = 600;

		//CV_WRAP SURF(double hessianThreshold, int nOctaves=4, int nOctaveLayers=2, bool extended=true, bool upright=false);

		//Ptr<SURF> detector =  cv::xfeatures2d::SURF::create(minHessian, 4, 2, true, true);
		Ptr<xfeatures2d::SURF> detector=xfeatures2d::SURF::create(minHessian, 4, 2, true, true);
		//SurfFeatureDetector detector(minHessian, 4, 2, true, true);

		//KeyPoint objects have Point2f object pt

		detector->detect( leftImageG, keypoints_left);
		detector->detect( rightImageG, keypoints_right);


        //calculate descriptors (feature vectors)
        Ptr<xfeatures2d::SURF> extractor = xfeatures2d::SURF::create();
        //SurfDescriptorExtractor extractor;
        cv::Mat desc_left, desc_right;
        extractor->compute( leftImageG, keypoints_left, desc_left);
        extractor->compute( rightImageG, keypoints_right, desc_right);

        //compute pairwise y-distances mask to limit possible matches to +/- 6 pixels
        float y_tol = 6.0;
        float l_val, r_val;
        cv::Mat yDimMask = cv::Mat::zeros(keypoints_left.size(), keypoints_right.size(), CV_8U);
        for (unsigned int l = 0; l < keypoints_left.size(); l++){
                for (unsigned int r = 0; r < keypoints_right.size(); r++){
                        l_val = keypoints_left[l].pt.y;
                        r_val = keypoints_right[r].pt.y;
                        if (fabs(l_val-r_val) <= y_tol){
                            yDimMask.at<uchar>(l,r) = 1;
                        }
                }
        }
 
    //match keypoints
    BFMatcher matcher(NORM_L2);
    std::vector< DMatch > matches;
    matcher.match( desc_left, desc_right, matches, yDimMask);

    //Quick calculation of max and min distances between keypoints
    double max_dist = 0; double min_dist = 100;
    double dist;
    for (unsigned int i = 0; i < keypoints_left.size(); i++){
            dist = matches[i].distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
    }


    //-- Localize the object
    std::vector<Point2f> leftPts;
    std::vector<Point2f> rightPts;

    std::vector<char> good_matches;
    float kpd_thresh = 0.1;
    for(unsigned int i = 0; i < matches.size(); i++ ){
            //threshold mask for visualization of matching keypoints
            if( matches[i].distance < (kpd_thresh*max_dist) ){
                    good_matches.push_back(1);
                    leftPts.push_back( keypoints_left[ matches[i].queryIdx ].pt );
                    rightPts.push_back( keypoints_right[ matches[i].trainIdx ].pt );
            }
            else{
                    good_matches.push_back(0); //0 FOR  MASKING
            }
    }


    //RANSAC for inliers from feature pairs

    for (unsigned int i = 0; i < leftPts.size(); i++){
            std::vector<Point2f> tmp;
            tmp.push_back(leftPts[i]);
             tmp.push_back(rightPts[i]);
             leftPts_rightPts.push_back(tmp);
             tmp.clear();
     }


	 bool isAdaptive = true;
	 int N = 1000;
	 float t = 0.5;
	 cv::Mat Model;

     if(leftPts_rightPts.size() >2){

    	   RANSACFitTransform_ylinear(leftPts_rightPts, inliers, Model, N, t, isAdaptive );






         //build matrices for SVD solver to refit to all inliers
         Mat A_l = cv::Mat::ones(inliers.size(), 2, CV_32F);
         Mat A_r = cv::Mat::ones(inliers.size(), 2, CV_32F);
         Mat b_l = cv::Mat::ones(inliers.size(), 1, CV_32F);
         Mat b_r = cv::Mat::ones(inliers.size(), 1, CV_32F);
         Mat inlierAverageyBeforeSVD = cv::Mat::ones(inliers.size(), 1, CV_32F);
         Mat x_l, x_r;
         float y_avg, y_left, y_right;
         for (unsigned int i = 0; i < inliers.size(); i++){
                 y_left = inliers[i][0].y;
                 y_right = inliers[i][1].y;
                 y_avg = (y_left + y_right)/2.0;

                 A_l.at<float>(i,0) = inliers[i][0].y;
                 b_l.at<float>(i,0) = y_avg;

                 A_r.at<float>(i,0) = inliers[i][1].y;
                 b_r.at<float>(i,0) = y_avg;

                 inlierAverageyBeforeSVD.at<float>(i,0) = abs(y_left - y_right);
         }
         solve(A_l, b_l, x_l, DECOMP_SVD);
         solve(A_r, b_r, x_r, DECOMP_SVD);


         //build homography matrices
         //Mat H_l = cv::Mat::zeros(3,3, CV_32F);
         //Mat H_r = cv::Mat::zeros(3,3, CV_32F);
         H_l.at<float>(0,0) = 1;
         H_r.at<float>(0,0) = 1;
         H_l.at<float>(2,2) = 1;
         H_r.at<float>(2,2) = 1;

         H_l.at<float>(1,1) = x_l.at<float>(0,0);
         H_l.at<float>(1,2) = x_l.at<float>(1,0);

         H_r.at<float>(1,1) = x_r.at<float>(0,0);
         H_r.at<float>(1,2) = x_r.at<float>(1,0);

         //cout << imageName << " left: \n" << H_l << endl;
         //cout << imageName << " right: \n" << H_r << endl;

         // H_L = H_l;
         // H_R = H_r;

         Size dsize = Size(w, h);
         warpPerspective(leftImageG, left_warped, H_l, dsize, INTER_LINEAR, BORDER_CONSTANT, 0); //+ WARP_INVERSE_MAP
         warpPerspective(rightImageG, right_warped, H_r, dsize, INTER_LINEAR, BORDER_CONSTANT, 0); //+ WARP_INVERSE_MAP

    }else{
        left_warped = leftImageG;
        right_warped = rightImageG;

         //Mat H_l = cv::Mat::zeros(3,3, CV_32F);
         //Mat H_r = cv::Mat::zeros(3,3, CV_32F);
         H_l.at<float>(0,0) = 1;
         H_r.at<float>(0,0) = 1;
         H_l.at<float>(1,1) = 1;
         H_r.at<float>(1,1) = 1;
         H_l.at<float>(2,2) = 1;
         H_r.at<float>(2,2) = 1;


    }


     //store keypoints for use in plane fitting later
//     Mat tmpL1 = cv::Mat::ones(3, 1, CV_32F);
//     Mat inlierAverageyAfterSVD = cv::Mat::ones(inliers.size(), 1, CV_32F);
//     Mat modelL1;
//
//     float x_right;
//     float disp = 0.0;
//     for (int i = 0; i < A_l.rows; i++){ //for all inlier points
//             tmpL1.at<float>(0,0) = inliers[i][0].x; // keypoint left image, x coordinate
//             tmpL1.at<float>(1,0) = A_l.at<float>(i,0); // keypoint left image, y coordinate original
//
//             x_right = inliers[i][1].x; // keypoint right image, x coordinate
//             disp = tmpL1.at<float>(0,0) - x_right; // d = x_left - x_right
//
//             modelL1 = H_l*tmpL1; // use homography to compute new y location of point, 3 rows x 1 col
//
//             inlierAverageyAfterSVD.at<float>(i,0) = abs(inliers[i][1].y - modelL1.at<float>(1,0));
//
//             returnKeypoints.at<float>(i,0) = modelL1.at<float>(0,0); //x orig
//             returnKeypoints.at<float>(i,1) = modelL1.at<float>(1,0); //y after model
//             returnKeypoints.at<float>(i,2) = disp; //disparity from keypoints
//     }
//
//
//
//     //FileStorage file((string)output+"metrics/"+ (string)imageName +"_inlierAverageyAfterSVD.txt",FileStorage::WRITE);
//     //file << "inlierAverageyAfterSVD" << inlierAverageyAfterSVD;
//
//     Mat differencesComparison = cv::Mat::ones(inliers.size(), 2, CV_32F);
//     for(int i=0; i<differencesComparison.rows; i++){
//             differencesComparison.at<float>(i,0) = inlierAverageyBeforeSVD.at<float>(i,0);
//             differencesComparison.at<float>(i,1) = inlierAverageyAfterSVD.at<float>(i,0);
//     }

}

void apply_homography( cv::Mat& leftImageG, cv::Mat& rightImageG, cv::Mat& left_warped, cv::Mat& right_warped,double w, double h, cv::Mat& H_L, cv::Mat& H_R){


        Size dsize = Size(w, h);
        warpPerspective(leftImageG, left_warped, H_L, dsize, INTER_LINEAR , BORDER_CONSTANT, 0); //+ WARP_INVERSE_MAP
        warpPerspective(rightImageG, right_warped, H_R, dsize, INTER_LINEAR , BORDER_CONSTANT, 0); //+ WARP_INVERSE_MAP
}

//Option Inter-linear or inter-nearest
PyObject* warp( PyObject* img, PyObject *homography, bool invert=false, bool option =false ){

	PyArrayObject* imgA = reinterpret_cast<PyArrayObject*>(img);
	PyArrayObject* hA = reinterpret_cast<PyArrayObject*>(homography);

	uchar * imgp = reinterpret_cast<uchar*>(PyArray_DATA(imgA));
	float * hp = reinterpret_cast<float*>(PyArray_DATA(hA));

	npy_intp *shape = PyArray_DIMS(imgA);



	Mat Image = Mat( shape[0],shape[1],CV_8UC1,imgp );
	Mat H_l = Mat( 3,3,CV_32FC1, hp);
	Mat warpedImage;

	if(invert){
		if(option == false){
			warpPerspective(Image,warpedImage,H_l,Image.size(),INTER_LINEAR+WARP_INVERSE_MAP, BORDER_CONSTANT, 0);
		}
		else{
			warpPerspective(Image,warpedImage,H_l,Image.size(),INTER_NEAREST+WARP_INVERSE_MAP, BORDER_CONSTANT, 0);
		}
	}else{
		if(option == false){
			warpPerspective(Image,warpedImage,H_l,Image.size(),INTER_LINEAR, BORDER_CONSTANT, 0);
		}
		else{
			warpPerspective(Image,warpedImage,H_l,Image.size(),INTER_NEAREST, BORDER_CONSTANT, 0);
		}

	}


	PyObject* warped = PyArray_SimpleNew(2, shape, NPY_UINT8);

	uint8* warp_data = static_cast<uint8*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(warped)));

    for (int i=0; i<shape[0]; i++){
    	for(int j=0; j<shape[1]; j++){
    		warp_data[i*shape[1]+j] = warpedImage.at<uchar>(i,j);
    	}
    }


	return warped;

}




void fixrectification(PyObject* left, PyObject *right, PyObject *hl, PyObject *hr){
    // Cast to pointer to Python Array object
    PyArrayObject* leftA = reinterpret_cast<PyArrayObject*>(left);
    PyArrayObject* rightA = reinterpret_cast<PyArrayObject*>(right);

    PyArrayObject* hlA = reinterpret_cast<PyArrayObject*>(hl);
    PyArrayObject* hrA = reinterpret_cast<PyArrayObject*>(hr);

    //Get the pointer to the data
    uchar * leftp = reinterpret_cast<uchar*>(PyArray_DATA(leftA));
    uchar * rightp = reinterpret_cast<uchar*>(PyArray_DATA(rightA));

    float * hlp = reinterpret_cast<float*>(PyArray_DATA(hlA));
    float * hrp = reinterpret_cast<float*>(PyArray_DATA(hrA));

    npy_intp *shape = PyArray_DIMS(leftA);


    Mat leftImage = Mat( shape[0],shape[1],CV_8UC1,leftp );
    Mat rightImage = Mat( shape[0],shape[1], CV_8UC1, rightp );

    cv::Mat leftWarpedG, rightWarpedG, H_l=cv::Mat::zeros(3,3, CV_32F), H_r=cv::Mat::zeros(3,3, CV_32F);

    double h = double(leftImage.rows);
    double w = double(leftImage.cols);



    compute_y_offset_ylinear( leftImage, rightImage, leftWarpedG, rightWarpedG, false, w, h, H_l, H_r);


//    std::cout << H_l.type() << std::endl;
    for (int i=0; i<H_l.rows; i++){
    	for(int j=0; j<H_l.cols; j++){
    		hlp[i*H_l.cols+j] = H_l.at<float>(i,j);
    		hrp[i*H_l.cols+j] = H_r.at<float>(i,j);
//    		std::cout << H_l.at<double>(i,j) << std::endl;
    	}
    }

    for (int i=0; i<shape[0]; i++){
    	for(int j=0; j<shape[1]; j++){
			leftp[i*shape[1]+j] = leftWarpedG.at<uchar>(i,j);
			rightp[i*shape[1]+j] = rightWarpedG.at<uchar>(i,j);
    	}
    }

}


BOOST_PYTHON_MODULE(librectification) {

    numeric::array::set_module_and_type("numpy", "ndarray");


    def("fixrectification",fixrectification);
    def("warp",warp, ( boost::python::arg("img"), boost::python::arg("homography"),
    					boost::python::arg("invert")=false, boost::python::arg("option")=false  ));


    import_array();
}



