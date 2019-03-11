#pragma once

#include "TimeStamper.h"
#include <opencv2/opencv.hpp>
#include "StereoEnergy.h"

//added by CCJ
//#include "../KITTI-devkit/evaluate_kt15_1_img.h"
#include "evaluate_kt15_1_img.h"
//#include "pgm_pfm/pfm_rw.h"
#include "pfm_rw.h"
#include <string>
using namespace std;

class Evaluator
{
	
protected:
	TimeStamper timer;
	const float DISPARITY_FACTOR;
	const cv::Mat dispGT;
	const cv::Mat nonoccMask;
	cv::Mat occMask;
	cv::Mat validMask;
	std::string saveDir;
	//**************************
	// added by CCJ;
	const std::string imgDir;
	const std::string dataType;
	//**************************
	
	std::string header;
	int validPixels;
	int nonoccPixels;
	FILE *fp_energy;
	FILE *fp_output;
	float qprecision;
	float errorThreshold;
	bool showedOnce;

public:
	double lastAccuracy;
	bool showProgress;
	bool saveProgress;
	bool printProgress;
	//*****************
  // added by CCJ;
	std::string getImgDir() const {return imgDir;}
	std::string getDataType() const {return dataType;}
	//*****************

	std::string getSaveDirectory()
	{
		return saveDir;
	}

	Evaluator(
			cv::Mat dispGT, 
			cv::Mat nonoccMask, 
			float disparityFactor, 
			std::string header, // e.g., == "result";
			std::string saveDir, // e.g., == "./"; 
			std::string imgDir, // e.g., == "./";
			std::string dataType, // e.g., == "KT15", "MiddV3", "MiddV2";
			bool show = true, bool print = true, bool save = true)
		: dispGT(dispGT)
		, nonoccMask(nonoccMask)
		, header(header)
		, saveDir(saveDir)
		, imgDir (imgDir)
		, dataType(dataType)
		, DISPARITY_FACTOR(disparityFactor)
		, showProgress(show)
		, saveProgress(save)
		, printProgress(print)
		, fp_energy(nullptr)
		, fp_output(nullptr)
	{

		if (save)
		{
			//fp_energy = fopen((saveDir + "log_energy.txt").c_str(), "w");
			//if (fp_energy != nullptr)
			//{
			//	fprintf(fp_energy, "%s\t%s\t%s\t%s\t%s\t%s\n", "Time", "Eng", "Data", "Smooth", "all", "nonocc");
			//	fflush(fp_energy);
			//}

			fp_output = fopen((saveDir + "log_output.txt").c_str(), "w");
			if (fp_output != nullptr)
			{
				if (dataType == "KT15"){
					fprintf(fp_output, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", "Index","Time", "Energy", "DataCost", "SmoothCost", "all-err(%)", "nonocc-err(%)");
				}

				if (dataType == "ETH3D"){
					fprintf(fp_output, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", "Index","Time", "Energy", "DataCost", "SmoothCost", "all-bad1.0(%)", "nonocc-bad1.0(%)");
				}
	      if (this -> dataType == "MiddV2" or this -> dataType == "MiddV3"){
					fprintf(fp_output, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", "Index", "Time", "Energy", "DataCost", "SmoothCost", 
							"all-bad2.0F(%)", "nonocc-bad2.0F(%)");
				}
				
				fflush(fp_output);
			}
		}

		showedOnce = false;
		errorThreshold = 0.5f;
		qprecision = 1.0f / DISPARITY_FACTOR;

		validMask = (dispGT > 0.0f) & (dispGT != INFINITY);
		validPixels = cv::countNonZero(validMask);
		occMask = ~nonoccMask & validMask;
		nonoccPixels = cv::countNonZero(nonoccMask);
	}
	~Evaluator()
	{
		if (fp_energy != nullptr) fclose(fp_energy);
		if (fp_output != nullptr) fclose(fp_output);
	}

	void setPrecision(float precision)
	{
		qprecision = precision;
	}
	void setErrorThreshold(float t)
	{
		errorThreshold = t;
	}

	//void outputFiles(cv::Mat labeling, int index, const char *header = "result", bool showImage = true, bool justShow = false)
	//{
	//	cv::Mat disparityMap = cvutils::channelDot(labeling, coordinates) * (DisparityFactor / 255.0);
	//	cv::Mat normalMap = getNormalMap(labeling);

	//	if (justShow == false){
	//		char str[512];
	//		sprintf(str, "%sD%02d.png", header, index);
	//		cv::imwrite(saveDir + str, disparityMap * 255);
	//		sprintf(str, "%sN%02d.png", header, index);
	//		cv::imwrite(saveDir + str, normalMap * 255);
	//	}
	//}

	void quantize(cv::Mat m, float precision)
	{
		cv::Mat qm = cv::Mat(m.size(), CV_32S);
		m.convertTo(qm, CV_32S, 1.0 / precision);
		qm.convertTo(m, m.type(), precision);
	}



	void evaluate(
			cv::Mat labeling_m, 
			cv::Mat unaryCost2, 
			const StereoEnergy& energy2,
			bool show, 
			bool save, 
			bool print, 
			//*****************************
			// training dataset, we could do 
			// the evaluation due to the 
			// ground truth of disparity;
			// But for test data, there is no 
			// ground truth, hence no evaluation;
			bool isTestDataset, 
			int index, // iteration index;
			int mode, // veiw mode: 0 = left; 1 = right;
			// ****************
			// added by CCJ;
			// ****************
			const std::string outputDir // E.g., == "./results/some_dir/", for saving the disparity once each iteration is done;
      //const std::string & dataType // e.g., == "KT15", MiddV3, "MiddV2"
			){
		
		bool isTicking = timer.isTicking();
		stop();

		cv::Mat labeling = labeling_m(energy2.getRectWithoutMargin());
		double sc2 = energy2.computeSmoothnessCost(labeling_m);
		double dc2 = cv::sum(unaryCost2)[0];
		double eng2 = sc2 + dc2;

		cv::Mat disp = energy2.computeDisparities(labeling);
		if (qprecision > 0)
			quantize(disp, qprecision);

		cv::Mat disparityMapVis = disp * DISPARITY_FACTOR / 255;
		cv::Mat normalMapVis = energy2.computeNormalMap(labeling);
		//cv::Mat vdispMapVis;
		//cv::extractChannel(labeling, vdispMapVis, 3);
		//vdispMapVis = (vdispMapVis + 3.0) / 6.0;
   
		cv::Mat errorMap = cv::abs(disp - dispGT) <= errorThreshold;
		cv::Mat errorMapVis = errorMap | (~validMask);
		errorMapVis.setTo(cv::Scalar(200), occMask & (~errorMapVis));
    
		float all = -1.0, nonocc = -1.0;
		// For testing dataset, no evaluation;
		// For training dataset, do evaluation;
		if (!isTestDataset){	
			if (this -> dataType == "MiddV2" or 
					this -> dataType == "MiddV3" or 
					this -> dataType == "ETH3D"){
				//std::cout << "Processing data " << this -> dataType << "\n";
				all = 1.0 - (float)cv::countNonZero(errorMap & validMask) / validPixels;
				nonocc = 1.0 - (float)cv::countNonZero(errorMap & nonoccMask) / nonoccPixels;
				all *= 100.0;
				nonocc *= 100.0;
			}
		
			// ****************
			// added by CCJ;
			// ****************
			else if (this-> dataType == "KT15"){
				//std::cout << "Processing data " << this -> dataType << "\n";
				const int h = disp.rows;
				const int w = disp.cols;
				//std::cout << "h = " << h<< ", w = " << w << "\n";
				float * p_disp = new float[h*w];
				cv::Mat tmpMat;
				disp.convertTo(tmpMat, CV_32FC1);
				for (int i = 0; i < h; ++i){
					for (int j = 0; j < w; ++j){
						p_disp[i*w+j] = tmpMat.at<float>(i, j);
					}
				}

				float * err_result = new float [14];
				//*******************************
				// calling function eval_kt2015 defined in file src/KITTI-devkit/evaluate_stereo.cpp;
				//*******************************
				if(eval_kt2015( this-> imgDir	, p_disp, err_result, mode)){
					all = err_result[12];
					nonocc = err_result[5];
					all *= 100.0;
					nonocc *= 100.0;
					// release memory;
					delete [] p_disp;
					delete [] err_result;
				}
				else
					std::cout << "Failed in evaluating " <<this-> imgDir << "\n";
			}

			else{
				std::cout << "Wrong datatype provided! Ensure it be one of KT15, MiddV2, MiddV3.\n";
			}
		}

		if (mode == 0)
			lastAccuracy = all;

		if (this-> showProgress && show)
		{
			if (showedOnce == false)
			{
				showedOnce = true;
				cv::namedWindow(header + std::to_string(mode) + "V", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
				cv::namedWindow(header + std::to_string(mode) + "D", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
				cv::namedWindow(header + std::to_string(mode) + "N", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
				cv::namedWindow(header + std::to_string(mode) + "E", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
			}
			//cv::imshow(header + std::to_string(mode) + "V", vdispMapVis);
			cv::imshow(header + std::to_string(mode) + "D", disparityMapVis);
			cv::imshow(header + std::to_string(mode) + "N", normalMapVis);
			cv::imshow(header + std::to_string(mode) + "E", errorMapVis);
			cv::waitKey(10);
		}

		if (this->saveProgress && save){
			//std::cout << "Saving : " << saveDir +  header + cv::format("%dD%02d.png", mode, index) << "\n";
			cv::imwrite(saveDir + header + cv::format("%dD%02d.png", mode, index), disparityMapVis * 255);
			cv::imwrite(saveDir + header + cv::format("%dN%02d.png", mode, index), normalMapVis * 255);
			cv::imwrite(saveDir + header + cv::format("%dE%02d.png", mode, index), errorMapVis);

			if (fp_output != nullptr && mode == 0)
			{
				fprintf(fp_output, "%d\t%lf\t%lf\t%lf\t%lf\t%f\t%f\n", index, getCurrentTime(), eng2, dc2, sc2, all, nonocc);
				fflush(fp_output);
			}
		}

		// Output energy values in inner loops
		if (mode == 0 && fp_energy != nullptr && saveProgress)
		{
			fprintf(fp_energy, "%d\t%lf\t%lf\t%lf\t%lf\t%f\t%f\n", index, getCurrentTime(), eng2, dc2, sc2, all, nonocc);
			fflush(fp_energy);
		}

		if (printProgress && print) if ( mode == 0){
			if (this -> dataType == "KT15"){
				std::cout << cv::format("%2d   %6.1lf   %14.0lf   %14.0lf   %8.0lf   %5.2f(occ-3.0)   %5.2f(noc-3.0)", 
						index, getCurrentTime(), eng2, dc2, sc2, all, nonocc) << std::endl;
			}
	    if (this -> dataType == "MiddV2" or this -> dataType == "MiddV3"){
				std::cout << cv::format("%2d   %6.1lf   %14.0lf   %14.0lf   %8.0lf   %5.2f(all-B2.0F)   %5.2f(noc-B2.0F)", 
						index, getCurrentTime(), eng2, dc2, sc2, all, nonocc) << std::endl;
			}
	    
			if (this -> dataType == "ETH3D"){
				std::cout << cv::format("%2d   %6.1lf   %14.0lf   %14.0lf   %8.0lf   %5.2f(all-B1.0F)   %5.2f(noc-B1.0F)", 
						index, getCurrentTime(), eng2, dc2, sc2, all, nonocc) << std::endl;
			}
      
		}

		if (printProgress && print){
			// saving the temporary disparity results after each iteration;
		  std::string tmp_name = outputDir + "disp" +  std::to_string(mode) + "-iter-" + std:: to_string(index) + ".pfm";
		  cvutils::io::save_pfm_file(tmp_name, disp);
		  //std::cout << "Saved " << tmp_name << " ... Done!\n";
		
		}
		
		if (isTicking)
			start();
	}



	void start()
	{
		timer.start();
	}

	void stop()
	{
		timer.stop();
	}

	double getCurrentTime()
	{
		return timer.getCurrentTime();
	}

};// end of class definition;
