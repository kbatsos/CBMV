#ifndef _PGM_PFM_H_
#define _PGM_PFM_H_
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <bitset>/*std::bitset<32>*/
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

unsigned char * read_ppm_image(const char *fname, const int &numChannel,
	int & nrows, int & ncols);
// U: uchar
int write_ppm_Uimage(const unsigned char *image, const char *fname,
	const int & nrows, const int & ncols, const int & numChannel);

void png2ppm(const std::string & pngName, const std::string & ppmName,
	const int & gray = 0);
void ppm2png(const std::string & ppmName, const std::string & pngName,
	const int & numChannel);

// Info about pfm file:
// see pfm file format, http://netpbm.sourceforge.net/doc/pfm.html.
// ** 1) Identifier Line: The identifier line contains the characters 
// "PF" or "Pf". PF means it's a color PFM. 
// Pf means it's a gray scale PFM.
// ** 2) Dimensions Line:
// The dimensions line contains two positive decimal integers, 
// separated by a blank. The first is the width of the image; 
// the second is the height. Both are in pixels.
// ** 3) Scale Factor / Endianness:
// The Scale Factor / Endianness line is a queer line that jams 
// endianness information into an otherwise sane description 
// of a scale. The line consists of a nonzero decimal number, 
// not necessarily an integer. If the number is negative, that 
// means the PFM raster is little endian. Otherwise, it is big 
// endian. The absolute value of the number is the scale 
// factor for the image.
// The scale factor tells the units of the samples in the raster.
// You use somehow it along with some separately understood unit 
// information to turn a sample value into something meaningful, 
// such as watts per square meter.

/* PFM raster:
* The raster is a sequence of pixels, packed one after another,
* with no delimiters of any kind. They are grouped by row,
* with the pixels in each row ordered left to right and
* the rows ordered bottom to top.
* Each pixel consists of 1 or 3 samples, packed one after another,
* with no delimiters of any kind. 1 sample for a grayscale PFM
* and 3 for a color PFM (see the Identifier Line of the PFM header).
* Each sample consists of 4 consecutive bytes. The bytes represent
* a 32 bit string, in either big endian or little endian format,
* as determined by the Scale Factor / Endianness line of the PFM
* header. That string is an IEEE 32 bit floating point number code.
* Since that's the same format that most CPUs and compiler use,
* you can usually just make a program use the bytes directly
* as a floating point number, after taking care of the
* endianness variation.
*/
enum PFM_endianness { BIG, LITTLE, ERROR };

class PFM {
public:
	PFM();
	inline bool is_little_big_endianness_swap() {
		if (this->endianess == 0.f) {
			std::cout << "Error! this-> endianness is not assigned yet!\n";
			exit(0);
		}
		else {
			uint32_t endianness = 0xdeadbeef;
			//std::cout << "\n" << std::bitset<32>(endianness) << std::endl;
			unsigned char * temp = (unsigned char *)&endianness;
			//std::cout << std::bitset<8>(*temp) << std::endl;
			PFM_endianness endianType_ = (((*temp) ^ 0xef) == 0 ?
				LITTLE : ((*temp) ^ (0xde)) == 0 ? BIG : ERROR);
			// ".pfm" format file specifies that: 
			// positive scale means big endianess;
			// negative scale means little endianess.
			return  ((BIG == endianType_) && (this->endianess < 0.f))
				|| ((LITTLE == endianType_) && (this->endianess > 0.f));
		}
	}

	template<typename T>
	T* read_pgm(const char* filename) {
		std::ifstream ifs(filename, std::ifstream::binary);
		std::string c;

		if (ifs) {
			ifs >> c;
			if (!c.compare("P2") || !c.compare("P5")) {
				ifs >> c;
				this->width = atoi(c.c_str());
				ifs >> c;
				this->height = atoi(c.c_str());
				ifs >> c;
				this->intensity = atoi(c.c_str());
				char b;
				int length_ = this->width * this->height;
				ifs.read(&b, 1);
				unsigned char * buffer = new unsigned char[length_];
				ifs.read((char *)buffer, length_);
				T * img = new T[length_];

				for (int i = 0; i < length_; i++)
					img[i] = (T)buffer[i];

				delete[] buffer;
				if (ifs) {
					return img;
				}
				else {
					std::cout << "Error only " << ifs.gcount()
						<< " was read..." << std::endl;
					exit(0);
				}
			}
			else {
				std::cout << "Invalid magic number!" << std::endl;
				ifs.close();
				exit(0);
			}
		}
		else {
			std::cout << "File " << filename << " does not exist!" << std::endl;
			ifs.close();
			exit(0);
		}
		ifs.close();
	}

	void write_pgm(char* filename, unsigned char* imgbuffer) {
		std::ofstream ofs(filename, std::ifstream::binary);
		ofs << "P5\n" << this->width << " " << this->height << "\n"
			<< this->intensity << "\n";
		ofs.write((char *)imgbuffer, this->width*this->height);
		ofs.close();
	}

	template<typename T>
	T * read_pfm(const std::string & filename) {
		FILE * pFile;
		pFile = fopen(filename.c_str(), "rb");
		char c[100];
		if (pFile != NULL) {
			fscanf(pFile, "%s", c);
			// strcmp() returns 0 if they are equal.
			if (!strcmp(c, "Pf")) {
				// format: "%s": String of characters. This will read 
				// subsequent characters until a whitespace is found 
				// (whitespace characters are considered to 
				//  be blank, newline and tab).
				fscanf(pFile, "%s", c);
				this->width = atoi(c);// atoi: ASCII to integer.
				fscanf(pFile, "%s", c);
				this->height = atoi(c);
				int length_ = this->width * this->height;
				fscanf(pFile, "%s", c);
				this->endianess = (float) atof(c);

				fseek(pFile, 0, SEEK_END);
				long lSize = ftell(pFile);
				long pos = lSize - this->width*this->height * sizeof(T);
				fseek(pFile, pos, SEEK_SET);

				T* img = new T[length_];
				//cout << "sizeof(T) = " << sizeof(T);
				fread(img, sizeof(T), length_, pFile);
				fclose(pFile);

				/* The raster is a sequence of pixels, packed one after another,
				 * with no delimiters of any kind. They are grouped by row,
				 * with the pixels in each row ordered left to right and
				 * the rows ordered BOTTOM to TOP.
				 */
				T* tbimg = (T *)malloc(length_ * sizeof(T));// top-to-bottom.
				//PFM SPEC image stored bottom -> top reversing image											
				for (int i = 0; i < this->height; i++) {
					memcpy(&tbimg[(this->height - i - 1)*(this->width)],
						&img[(i*(this->width))],
						(this->width) * sizeof(T));
				}


				if (this->is_little_big_endianness_swap()) {
					//std::cout << "little-big endianness transformation is needed.\n";
					// little-big endianness transformation is needed.
					union {
						T f;
						unsigned char u8[sizeof(T)];
					} source, dest;

					for (int i = 0; i < length_; ++i) {
						source.f = tbimg[i];
						for (unsigned int k = 0, s_T = sizeof(T); k < s_T; k++)
							dest.u8[k] = source.u8[s_T - k - 1];
						tbimg[i] = dest.f;
						//cout << dest.f << ", ";
					}
				}
				delete[] img;
				return tbimg;

			}
			else {
				std::cout << "Invalid magic number!"
					<< " No Pf (meaning gray scale pfm) is missing!!\n";
				fclose(pFile);
				exit(0);
			}

		}
		else {
			std::cout << "Cannot open file " << filename
				<< ", or it does not exist!\n";
			fclose(pFile);
			exit(0);
		}

	}


	
	template<typename T>
	void write_pfm(const std::string & filename, T* imgbuffer,
		const float & endianess_) {
		std::ofstream ofs(filename.c_str(), std::ifstream::binary);
		
		ofs << "Pf\n"
			<< this->width << " " << this->height << "\n"
			<< endianess_ << "\n";
		
		int length_ = this->width*this->height;
		this->endianess = endianess_;

		if (this->is_little_big_endianness_swap()) {
			std::cout << "little-big endianness transformation is needed.\n";
			// little-big endianness transformation is needed.
			union {
				T f;
				unsigned char u8[sizeof(T)];
			} source, dest;

			for (int i = 0; i < length_; ++i) {
				// change invalid pixels (which seem to be represented as -10)
				// to INF.
				// INFINITY: defined in header <math.h>.
				source.f = (imgbuffer[i] < 0) ? INFINITY : imgbuffer[i];
				for (size_t k = 0, s_T = sizeof(T); k < s_T; k++)
					dest.u8[k] = source.u8[s_T - k - 1];
				imgbuffer[i] = dest.f;
				//cout << dest.f << ", ";
			}
		}

		for (int i = height - 1; i >= 0; i--) {
			T * ptr = imgbuffer + i * this->width;
			ofs.write((char *)ptr, this->width * sizeof(T));
		}
		ofs.close();
	}

	float getEndianess();
	int getHeight(void);
	int getWidth(void);
	int getIntensity(void);
	void setHeight(const int & height);
	void setWidth(const int & width);
	virtual ~PFM();
private:
	int height;
	int width;
	int intensity;
	float endianess;
};

#endif /* PGM_H_ */
