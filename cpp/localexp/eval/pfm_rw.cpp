/*
 * pgm.cpp
 *
 *  Created on: Jun 9, 2016
 *      Author: kochoba
 */

#include "pfm_rw.h"
using namespace std;

unsigned char *read_ppm_image(const char *fname, const int &numChannel,
	int & nrows, int & ncols){

	unsigned char * dummy;
	int i_size, x, y;
  //int s;
	ifstream fin;
	fin.open(fname, ios::binary);
	if (!fin.is_open()) {
		cout << "File I/O error" << endl;
		exit(0);
	}

	char line[110];
	fin.getline(line, 100); //read P#

	fin.getline(line, 5, ' ');

	if (line[0] >= '0' && line[0] <= '9')  // read number
		x = atoi(line);
	else
	{
		fin.getline(line, 100);  // read through comment
		fin.getline(line, 5, ' ');
		x = atoi(line);
	}

	//fin >> y >> s; // >> t;
	fin.getline(line, 50);
	y = atoi(line);
	fin.getline(line, 50);
	//s = atoi(line);

	//cout << "x " << x << " y " << y << endl;
	//cout << "s " << s << endl; // " t " << t << endl;
	while ('\n' == fin.peek())
		fin.ignore(1);
	nrows = x; ncols = y;
	i_size = (nrows)*(ncols)*numChannel;
	dummy = new unsigned char[i_size];
	fin.read((char *)dummy, i_size);
	fin.close();
	return dummy;
}


int write_ppm_Uimage(const unsigned char *image, const char *fname,
	const int & nrows, const int & ncols, const int & numChannel){
	
	ofstream fout;
	//cout << "Writing " << nrows << "x" << ncols << " image." << endl;
	fout.open(fname, ios::binary);
	if (!fout.is_open()) {
		cout << "File I/O error" << endl;
		return 0;
	}

	fout << "P6" << endl; // P6 means binary (most practical).
	fout << "# Intermediate image file" << endl;
	fout << nrows << " " << ncols << endl; // rows, columns
	fout << "255" << endl; // maxVal

	fout.write((const char *)image, numChannel * nrows * ncols * 
		sizeof(unsigned char));
	fout.close();
	return 1;
}


void ppm2rgb(unsigned char *image, unsigned char** red,
	unsigned char** green, unsigned char** blue,
	int width, int height)
{
	int i, j, pt = 0;

	for (j = 0; j < height; j++)
		for (i = 0; i < width; i++)
		{
			red[i][j] = image[pt];
			pt++;
			green[i][j] = image[pt];
			pt++;
			blue[i][j] = image[pt];
			pt++;
		}

}

void png2ppm(const std::string & pngName, const std::string & ppmName,
	const int & gray){

	cv::Mat img = cv::imread(pngName, gray);// 0: gray; 1: color.
	if (img.empty()) {
		std::cout << "No image read for " << pngName << "\n";
		return;
	}
	int imgW = img.cols, imgH = img.rows, idx = 0;
	uchar * img_buf = new uchar[img.channels()*imgW *imgH];

	if (img.channels() == 3) {
		for (int r = 0; r < imgH; ++r) {
			for (int c = 0; c < imgW; ++c) {
				img_buf[idx] = img.ptr<cv::Vec3b>(r)[c][2];// red
				img_buf[idx + 1] = img.ptr<cv::Vec3b>(r)[c][1];//green
				img_buf[idx + 2] = img.ptr<cv::Vec3b>(r)[c][0];//blue
				idx += 3;
			}
		}
	}

	else {
		for (int r = 0; r < imgH; ++r) {
			for (int c = 0; c < imgW; ++c) {
				img_buf[idx] = img.ptr<uchar>(r)[c];// red
				idx++;
			}
		}
	}
	write_ppm_Uimage(img_buf, ppmName.c_str(), imgH, imgW, img.channels());
}


void ppm2png(const std::string & ppmName, const std::string & pngName,
	const int & numChannel)
{
	int imgW, imgH, idx = 0;
	unsigned char * img_buf = read_ppm_image(ppmName.c_str(), numChannel,
		imgH, imgW);

	cv::Mat img;
	if (numChannel == 3) {
		img = cv::Mat(imgH, imgW, CV_8UC3);
		for (int r = 0; r < imgH; ++r) {
			for (int c = 0; c < imgW; ++c) {
				img.ptr<cv::Vec3b>(r)[c][2] = img_buf[idx];// red
				img.ptr<cv::Vec3b>(r)[c][1] = img_buf[idx + 1];//green
				img.ptr<cv::Vec3b>(r)[c][0] = img_buf[idx + 2];//blue
				idx += 3;
			}
		}
	}

	else {
		img = cv::Mat(imgH, imgW, CV_8UC1);
		for (int r = 0; r < imgH; ++r) {
			for (int c = 0; c < imgW; ++c) {
				img.ptr<uchar>(r)[c] = img_buf[idx];// red
				idx++;
			}
		}
	}

	cv::imwrite(pngName, img);
}

PFM::PFM(){
	// TODO Auto-generated constructor stub
	this->height = 0;
	this->width=0;
	this->intensity= 255;
}
float PFM::getEndianess() {
	return this->endianess;
}

int PFM::getHeight(void){
	return this->height;
}

int PFM::getWidth(void){
	return this->width;
}

int PFM::getIntensity(void){
	return this->intensity;
}

void PFM::setHeight(const int & height){
	this->height=height;
}

void PFM::setWidth(const int & width){
	this->width=width;
}

PFM::~PFM() {
	// TODO Auto-generated destructor stub
}

// example to use this class.
#if 0
PFM pfm_rw;
string temp = dir + "img/Motorcycle/disp0GT.pfm";
float * p_disp_gt = pfm_rw.read_pfm<float>(temp);
int imgH = pfm_rw.getHeight();
int imgW = pfm_rw.getWidth();
float scale = pfm_rw.getEndianess();

string temp2 = dir + "result/Motorcycle/disp0GT_p1";
//pfm_rw.write_pfm<float>(temp2 + ".pfm", p_disp_gt, 1.0f);
delete[] p_disp_gt;

float * p_disp =
pfm_rw.read_pfm<float>(temp2 + ".pfm");
imgH = pfm_rw.getHeight();
imgW = pfm_rw.getWidth();
scale = pfm_rw.getEndianess();
Mat disp(imgH, imgW, CV_32FC1);
for (int i = 0, idx = 0; i < imgH; ++i) {
	for (int j = 0; j < imgW; ++j, ++idx) {
		disp.at<float>(i, j) = (p_disp[idx] > 10 * imgW) ? 0 : p_disp[idx];
	}
}
cout << "\n\n";
for (int i = 0; i < 1; ++i) {
	for (int j = 0; j < 20; ++j) {
		cout << disp.at<float>(i, j) << "\t";
	}
	cout << "\n";
}
#endif
