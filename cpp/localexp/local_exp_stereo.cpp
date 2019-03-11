//#define _CRT_SECURE_NO_WARNINGS

#include <omp.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "FastGCStereo.h"
#include "Evaluator.h"
#include "ArgsParser.h"
#include "CostVolumeEnergy.h"
#include "Utilities.hpp"

// a C/C++ header file provided by Microsoft Windows
//#include <direct.h>
#include <unistd.h>

//**************************************
//added by CCJ for Python + CPP coding;
//**************************************
#include <experimental/filesystem>
#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include <numpy/ndarrayobject.h>
using namespace boost::python;
namespace fs = std::experimental::filesystem;


struct Options
{
	std::string mode = "MiddV3"; // "ETH3D", "KT15" or "MiddV3"
	std::string outputDir = "";
	std::string targetDir = "";
  //std::string costvolumeLeftFile =  "";
	//std::string costvolumeRightFile = "";

	int iterations = 4;
	int pmIterations = 1;
	bool doDual = true;
	bool isTestData = false;// do evaluation or not;

	int ndisp = 0;
	float smooth_weight = 1.0;
	float mc_threshold = 0.8;
	int filterRadious = 20;

	int threadNum = -1;
	// cell width for paramter tuning;
	float cell_w1 = 0.01, cell_w2 = 0.03, cell_w3 = 0.09;
	// the seed of random value generator, 
	int seed = 1234;

	void loadOptionValuesViaDict(boost::python::dict my_args_dict){
		outputDir = extract<std::string> (my_args_dict["outputDir"]);
		targetDir = extract<std::string> (my_args_dict["targetDir"]);
		//costvolumeLeftFile = extract<std::string> (my_args_dict["costvolumeLeftFile"]);
	  //costvolumeRightFile = extract<std::string> (my_args_dict["costvolumeRightFile"]);
		//mode = extract<std::string> (my_args_dict["mode"]);
		smooth_weight = extract<float>(my_args_dict["smooth_weight"]);
		filterRadious = extract<int>(my_args_dict["filterRadious"]);
		//doDual = extract<int>(my_args_dict["doDual"]);
		isTestData = extract<int>(my_args_dict["isTestDataset"]);
		mc_threshold = extract<float>(my_args_dict["mc_threshold"]);
		threadNum = extract<int>(my_args_dict["threadNum"]);
		iterations = extract<int>(my_args_dict["iterations"]);
		pmIterations = extract<int>(my_args_dict["pmIterations"]);
		ndisp = extract<int>(my_args_dict["ndisp"]);
		
		// to control the cell size of local expansion;
	  cell_w1 = extract<float>(my_args_dict["cell_w1"]);
	  cell_w2 = extract<float>(my_args_dict["cell_w2"]);
	  cell_w3 = extract<float>(my_args_dict["cell_w3"]);
	  seed  = extract<int >(my_args_dict["seed"]); 
	}


	void printOptionValues(FILE * out = stdout)
	{
		fprintf(out, "----------- parameter settings -----------\n");
		fprintf(out, "mode           : %s\n", mode.c_str());
		fprintf(out, "outputDir      : %s\n", outputDir.c_str());
		fprintf(out, "targetDir      : %s\n", targetDir.c_str());
		//fprintf(out, "costRightFile  : %s\n", costvolumeRightFile.c_str());
		//fprintf(out, "costLeftFile   : %s\n", costvolumeLeftFile.c_str());

		fprintf(out, "threadNum      : %d\n", threadNum);
		fprintf(out, "doDual         : %d\n", (int)doDual);
		fprintf(out, "pmIterations   : %d\n", pmIterations);
		fprintf(out, "iterations     : %d\n", iterations);

		fprintf(out, "ndisp          : %d\n", ndisp);
		fprintf(out, "filterRadious  : %d\n", filterRadious);
		fprintf(out, "smooth_weight  : %f\n", smooth_weight);
		fprintf(out, "mc_threshold   : %f\n", mc_threshold);
		fprintf(out, "isTestDataset  : %d\n", (int)isTestData);

		fprintf(out, "cell_w1        : %f\n", cell_w1);
		fprintf(out, "cell_w2        : %f\n", cell_w2);
		fprintf(out, "cell_w3        : %f\n", cell_w3);
		if (seed > 0)
			fprintf(out, "input seed     : %d\n", seed);
	}
};/*end of Options Structure*/

/*
const Parameters paramsBF = Parameters(20, // lambda
	 	20, // windR 
	 	"BF", // filterName
	 	10 // filter_params = 10;
		);
const Parameters paramsGFfloat = Parameters(1.0f, 20, "GFfloat", 0.0001f); // Slightly faster
*/

const Parameters paramsGF = Parameters(1.0f, 20, "GF", 0.0001f);


void fillOutOfView(cv::Mat& volume, int mode, int margin = 0)
{
	int D = volume.size.p[0];
	int H = volume.size.p[1];
	int W = volume.size.p[2];

	if (mode == 0)
		for (int d = 0; d < D; d++)
			for (int y = 0; y < H; y++){
				auto p = volume.ptr<float>(d, y);
				auto q = p + d + margin;
				float v = *q;
				while (p != q){
					*p = v;
					p++;
				}
			}
	else
		for (int d = 0; d < D; d++)
			for (int y = 0; y < H; y++){
				auto q = volume.ptr<float>(d, y) + W;
				auto p = q - d - margin;
		    float v = p[-1];
				while (p != q){
					*p = v;
					p++;
				}
			}
}

cv::Mat convertVolumeL2R(cv::Mat& volSrc, int margin = 0)
{
	int D = volSrc.size[0];
	int H = volSrc.size[1];
	int W = volSrc.size[2];
	cv::Mat volDst = volSrc.clone();

	for (int d = 0; d < D; d++)
	{
		cv::Mat_<float> s0(H, W, volSrc.ptr<float>() + H*W*d);
		cv::Mat_<float> s1(H, W, volDst.ptr<float>() + H*W*d);
		s0(cv::Rect(d, 0, W - d, H)).copyTo(s1(cv::Rect(0, 0, W - d, H)));

		cv::Mat edge1 = s0(cv::Rect(W - 1 - margin, 0, 1, H)).clone();
		cv::Mat edge0 = s0(cv::Rect(d + margin, 0, 1, H)).clone();
		for (int x = W - 1 - d - margin; x < W; x++)
			edge1.copyTo(s1.col(x));
		for (int x = 0; x < margin; x++)
			edge0.copyTo(s1.col(x));
	}
	return volDst;
}

bool loadData(const std::string inputDir, cv::Mat& im0, cv::Mat& im1, cv::Mat& dispGT, cv::Mat& nonocc)
{
	im0 = cv::imread(inputDir + "imL.png");
	im1 = cv::imread(inputDir + "imR.png");
	if (im0.empty() || im1.empty())
	{
		im0 = cv::imread(inputDir + "im0.png");
		im1 = cv::imread(inputDir + "im1.png");

		if (im0.empty() || im1.empty())
		{
			printf("Image pairs (im0.png, im1.png) or (imL.png, imR.png) not found in\n");
			printf("%s\n", inputDir.c_str());
			return false;
		}
	}

	// cvutils is defined in file src/Utilities.hpp
	dispGT = cvutils::io::read_pfm_file(inputDir + "disp0GT.pfm");
	
	//*******************************************
	// for testing dataset, there is no 
	// any information about the ground truth;
	// ******************************************
	if (dispGT.empty())
		dispGT = cv::Mat_<float>::zeros(im0.size());

	nonocc = cv::imread(inputDir + "nonocc.png", cv::IMREAD_GRAYSCALE);
	if(nonocc.empty())
		nonocc = cv::imread(inputDir + "mask0nocc.png", cv::IMREAD_GRAYSCALE);

	if (!nonocc.empty()) 
		nonocc = nonocc == 255;
	//*******************************************
	// for testing dataset, there is no 
	// any information about the ground truth;
	// ******************************************
	else
		nonocc = cv::Mat_<uchar>(im0.size(), 255);

	return true;
}


inline void check_negative(cv::Mat & disp){
		for (int r = 0; r< disp.rows; ++r){
			for (int c = 0; c< disp.cols; ++c){
				disp.at<float>(r,c) = (disp.at<float>(r,c) < 0.0) ? 0.0 : disp.at<float>(r,c);
			}
		}
}

// to create a directory
// returns true if successful;
// otherwise, returns false;
inline bool MakeDir(const string & dir_path){
	fs::path p(dir_path);
	if (fs::exists(p) && fs::is_directory(p)){ // if directory p actually exist;
		std::cout << p << " already exists.\n";
		return true;
	}
	else{ // p actually does not exist;
		return fs::create_directory(p);
	}
}


//void MidV3(
std::vector<cv::Mat> MidV3(
		const std::string inputDir, 
	  //e.g., == "../data/MiddV3/trainingH/Adirondack/im0.acrt";
		//const std::string costvolumeLeftFile,
		//const std::string costvolumeRightFile, // it could be NULL;
		const float * volume0, // float volume0[ndisp][height][width], for left image;
		//const float * volume1, // float volume1[ndisp][height][width], for right image;
		const std::string outputDir, 
		const Options& options){

	cv::Mat imL, imR, dispGT, nonocc;
	//Calib calib;
	//calib.ndisp = options.ndisp; // ndisp of argument option has higher priority
	if (loadData(inputDir, imL, imR, dispGT, nonocc) == false)
		exit(-1);
	printf("ndisp = %d\n", options.ndisp);

	// For evaluation, set the threshold be 1.0f for Half Resolution image in Middlebury;
	float maxdisp = (float)options.ndisp - 1;
	float errorThresh = 1.0f;
	/*
	if (cvutils::contains(inputDir, "trainingQ") || cvutils::contains(inputDir, "testQ"))
		errorThresh = errorThresh / 2.0f;
	else if (cvutils::contains(inputDir, "trainingF") || cvutils::contains(inputDir, "testF"))
		errorThresh = errorThresh * 2.0f;
  */

	Parameters param = paramsGF;
	param.windR = options.filterRadious;
	param.lambda = options.smooth_weight;
	param.th_col = options.mc_threshold; // tau_CNN in the paper

	int sizes[] = { options.ndisp, imL.rows, imL.cols };
	cv::Mat volL = cv::Mat_<float>(3, sizes);
	if (volume0 != NULL){
		for (int d = 0; d < sizes[0]; ++d){
			int tmp_d = d * sizes[1]* sizes[2];
			for (int h = 0; h < sizes[1]; ++h){
				int tmp_h = h * sizes[2];
				for (int w = 0; w < sizes[2]; ++w){
					volL.at<float>(d,h,w) = volume0[tmp_d + tmp_h + w]; 
				}
			}
		}
		printf("loading left cost volume from memory ... Done!\n");
	}
	else{
		printf("Left cost volume %s not found\n");
		exit(-1);
	}
  
	int interp_margin = 0; // Disabled as the use of margin worsens the results...
	fillOutOfView(volL, 0, interp_margin);

	cv::Mat volR = cv::Mat_<float>(3, sizes);
	volR = convertVolumeL2R(volL, interp_margin);

	std::cout << "dir = " << outputDir + "debug" << "\n";
	MakeDir((outputDir + "debug"));
  
	printf("Running by %s mode.\n", options.mode.c_str());
	Evaluator *eval = new Evaluator(
			dispGT, 
			nonocc, 
			255.0f / (maxdisp), 
			"result", 
			outputDir + "debug/",
			inputDir, // added for KT15 data set evaluation, Middlebury will not use this parameter;
			options.mode // e.g. , == "MiddV3", "ETH3D", or "KT15";
			);


	eval->setPrecision(-1);
	eval->showProgress = false;
	eval->setErrorThreshold(errorThresh);
  
	// added by CCJ;
	bool isTestData = options.isTestData == 1 ? true : false;
	std::cout << "isTestData = " << isTestData << "\n";
	FastGCStereo stereo(imL, imR, param, outputDir, isTestData, maxdisp);
	stereo.setStereoEnergyCPU(std::make_unique<CostVolumeEnergy>(imL, imR, volL, volR, param, maxdisp));
	stereo.saveDir = outputDir + "debug/";
	stereo.setEvaluator(eval);

	int w = imL.cols;
	IProposer* prop1 = new ExpansionProposer(1);
	IProposer* prop2 = new RandomProposer(7, maxdisp);
	IProposer* prop3 = new ExpansionProposer(2);
	IProposer* prop4 = new RansacProposer(1);
	//stereo.addLayer(int(w * 0.01), { prop1, prop4, prop2 });
	//stereo.addLayer(int(w * 0.03), { prop3, prop4 });
	//stereo.addLayer(int(w * 0.09), { prop3, prop4 });
	int cell_w1 = options.cell_w1 < 1.0 ? int( options.cell_w1 * w) : int(options.cell_w1);
	int cell_w2 = options.cell_w2 < 1.0 ? int( options.cell_w2 * w) : int(options.cell_w2);
	int cell_w3 = options.cell_w3 < 1.0 ? int( options.cell_w3 * w) : int(options.cell_w3);
	// for negative values, use the default coeffcients: 0.01, 0.03, 0.09;
	cell_w1 = cell_w1 <= 0 ? int( w * 0.01) : cell_w1;
	cell_w2 = cell_w2 <= 0 ? int( w * 0.03) : cell_w2;
	cell_w3 = cell_w3 <= 0 ? int( w * 0.09) : cell_w3;

	std::cout << "3 cell grids : " << cell_w1 << ", " << cell_w2 << ", " << cell_w3 << "\n";
	stereo.addLayer( cell_w1, { prop1, prop4, prop2 });
	stereo.addLayer( cell_w2, { prop3, prop4 });
	stereo.addLayer( cell_w3, { prop3, prop4 });
  //exit(0);
	
	cv::Mat labeling, rawdisp, labeling1, rawdisp1;
	vector<cv::Mat> labelingLR = {labeling, labeling1};
	vector<cv::Mat> rawdispLR = {rawdisp, rawdisp1};
	
	if (options.doDual){
		stereo.run(options.iterations, { 0, 1 }, options.pmIterations, 
				labelingLR[0], rawdispLR[0], labelingLR[1], rawdispLR[1]);
	}
	else{
		stereo.run(options.iterations, { 0 }, options.pmIterations, labelingLR[0], rawdispLR[0]);
	}

	delete prop1;
	delete prop2;
	delete prop3;
	delete prop4;

	//cvutils::io::save_pfm_file(outputDir + "disp0.pfm", stereo.getEnergyInstance().computeDisparities(labeling));
	
	vector<cv::Mat> disps;
	
	if (options.doDual)
		disps = vector<cv::Mat>({stereo.getEnergyInstance().computeDisparities(labelingLR[0]), 
				stereo.getEnergyInstance().computeDisparities(labelingLR[1])});
	else
		disps = vector<cv::Mat>({stereo.getEnergyInstance().computeDisparities(labelingLR[0])});

	for (int i = 0; i < disps.size() ; ++i){
		check_negative(disps[i]);
		cvutils::io::save_pfm_file(outputDir + "disp" + std:: to_string(i) + ".pfm", disps[i]);
	}

	if(options.doDual){ 
		cv::Mat disp0raw = stereo.getEnergyInstance().computeDisparities(rawdispLR[0]);
		check_negative(disp0raw);
		cvutils::io::save_pfm_file(outputDir + "disp0raw.pfm", disp0raw);
	}
	{
		FILE *fp = fopen((outputDir + "time.txt").c_str(), "w");
		if (fp != nullptr){ fprintf(fp, "%lf\n", eval->getCurrentTime()); fclose(fp); }
	}

	delete eval;
	return disps;
}


PyObject * local_exp_stereo_cost_volume(
		boost::python::dict my_args_dict,
		// left cost volume, a 3-d array in shape of [ndisp, height, width]
		PyObject* cost_volume0
		){

	Options options;
	options.loadOptionValuesViaDict(my_args_dict);
	unsigned int seed = (unsigned int)time(NULL);
	if (options.seed > 0){
		seed = options.seed;
		//std::cout << "Random generator seed is FIXED as " << seed << "!!!\n";
	}
	else{
		//std::cout << "Random generator seed is RANDOM, as current time " << seed << "!!!\n";
	}

#if 0
	// For debugging
	//  1  99.4        262247  252693  9554    10.51   8.54
	options.targetDir = "../data/MiddV3/trainingH/Adirondack";
	options.costvolumeFile = "../data/MiddV3/trainingH/Adirondack/im0.acrt";
	options.outputDir = "../results/Adirondack";
	options.mode = "MiddV3";
	options.smooth_weight = 0.5;
	options.pmIterations = 2;
	//options.threadNum = 1;
	seed = 0;
#endif
	//std::cout << "calling printOptionValues() function:\n";
	options.printOptionValues();
  //exit(0);
	
	int nThread = omp_get_max_threads();
	#pragma omp parallel for
	for (int j = 0; j < nThread; j++)
	{
		srand(seed + j);
		cv::theRNG() = seed + j;
	}

	if (options.threadNum > 0)
		omp_set_num_threads(options.threadNum);

	
	if (options.outputDir.length()){
		//_mkdir((options.outputDir).c_str());
		std:: string tmpDir = options.outputDir;
		if (MakeDir(tmpDir))
			std::cout << "Make dir : " << tmpDir << "\n";
		else
			std::cout << "Failed in making dir : " << tmpDir << "\n";
	}
	printf("\n\n");
	
  
    PyArrayObject* cost_volume0A = reinterpret_cast<PyArrayObject*>(cost_volume0);
		npy_intp * size = PyArray_DIMS(cost_volume0A);
		printf("cost_volume0 shape = [%d, %d, %d]\n", size[0], size[1], size[2]);

    // 3-dim array is reinterpreted as 1-d array.
		float * volume0 = reinterpret_cast<float*>(PyArray_DATA(cost_volume0A));

		std::vector<cv::Mat> v_disps = MidV3(
					options.targetDir + "/", 
					//options.costvolumeLeftFile, 
					//options.costvolumeRightFile,
					volume0,
					//volume1,
					options.outputDir + "/", 
					options);
		
		npy_intp * sizes = new npy_intp[2];
		cv::Mat disp0= v_disps[0];
		sizes[0] = disp0.rows; 
		sizes[1] = disp0.cols;
		printf("sizes = (%d, %d)\n", sizes[0], sizes[1]);

	  PyObject* disp = PyArray_SimpleNew(2, sizes, NPY_FLOAT32);
		
		float * p_disp = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(disp)));

		for (int h = 0; h < sizes[0]; ++h){
			int tmp_h = h * sizes[1];
			for (int w = 0; w < sizes[1]; ++w){
				p_disp[tmp_h + w] = disp0.at<float>(h, w);

			}
		}

	  std::cout << "Left disparity ... Done!\n";
		return disp;
}


BOOST_PYTHON_MODULE(liblocal_exp_stereo){
	numeric::array::set_module_and_type("numpy", "ndarray");
	//def("run_local_exp_stereo", local_exp_stereo);
	def("run_local_exp_stereo", local_exp_stereo_cost_volume);
	import_array();
}
