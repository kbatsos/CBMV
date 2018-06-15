#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <iostream>
#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include <numpy/ndarrayobject.h>
#include <algorithm>
#include <vector>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <omp.h>
#include <math.h> 
#include <string>

#include <assert.h>
#include <math_constants.h>
#include <unistd.h>
using namespace boost::python;

#define TB 128
#define BLOCK_D_SIZE 64
#define MAX_THREADS 1024

#define DISP_MAX 256



void checkCudaError() {
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}

void checkCudaError(std::string s ) {
     cudaError_t err = cudaGetLastError();
     std::cout << s << std::endl;
     if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}


#define COLOR_DIFF(x, i, j) (abs(x[i] - x[j]))


__device__ void sort(float *x, int n)
{
	for (int i = 0; i < n - 1; i++) {
		int min = i;
		for (int j = i + 1; j < n; j++) {
			if (x[j] < x[min]) {
				min = j;
			}
		}
		float tmp = x[min];
		x[min] = x[i];
		x[i] = tmp;
	}
}



class  CUDA_double_array{
	public:
		CUDA_double_array(PyObject* in){
			PyArrayObject* inA = reinterpret_cast<PyArrayObject*>(in);

			this->ndims = PyArray_NDIM(inA);
			this->desc = PyArray_DESCR(inA);
			npy_intp *sizes = PyArray_DIMS(inA);
			this->ins = new npy_intp[this->ndims];
			for (int i=0; i< this->ndims; i++)
				this->ins[i] = sizes[i];
			assert(this->desc->type_num==12);

			double * inp = reinterpret_cast<double*>(PyArray_DATA(inA));
			this->bytes=this->ins[0];
			for(int i=1; i< this->ndims;i++){
				this->bytes *= this->ins[i];
				
			}
			cudaMalloc(&this->array_d, this->bytes*sizeof(double));
    		cudaMemcpy( this->array_d, inp, this->bytes*sizeof(double), cudaMemcpyHostToDevice);
    		checkCudaError();
		}

		~CUDA_double_array(){
			cudaFree(this->array_d);
		}

		void set_to_zero(){
				cudaMemset(this->array_d,0 , this->bytes*sizeof(double));
				checkCudaError();
		}

		PyObject* get_array(){

			PyObject* array_h = PyArray_SimpleNew(this->ndims, this->ins, NPY_FLOAT64);
			double* array_data = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array_h)));
			cudaMemcpy( array_data, this->array_d, this->bytes*sizeof(double), cudaMemcpyDeviceToHost );
			checkCudaError();
			return array_h;

		}

		void free_array(){
			cudaFree(this->array_d);
		}
		double* array_d;
		int ndims;
		npy_intp *ins ;
		size_t bytes;
	private: 
		PyArray_Descr* desc;
		
};

class  CUDA_float_array{
	public:
		CUDA_float_array(PyObject* in){
			PyArrayObject* inA = reinterpret_cast<PyArrayObject*>(in);
			this->ndims = PyArray_NDIM(inA);
			this->desc = PyArray_DESCR(inA);
			this->ins = new npy_intp[this->ndims];
			npy_intp *sizes = PyArray_DIMS(inA);
			for (int i=0; i< this->ndims; i++)
				this->ins[i] = sizes[i];
			assert(this->desc->type_num==11);

			float * inp = reinterpret_cast<float*>(PyArray_DATA(inA));
			this->bytes=this->ins[0];
			for(int i=1; i< this->ndims;i++){
				this->bytes *= this->ins[i];
			}
			cudaMalloc(&this->array_d, this->bytes*sizeof(float));
    		cudaMemcpy( this->array_d, inp, this->bytes*sizeof(float), cudaMemcpyHostToDevice);
    		checkCudaError();

		}

		~CUDA_float_array(){
			cudaFree(this->array_d);
		}

		void set_to_zero(){
				cudaMemset(this->array_d,0 , this->bytes*sizeof(float));
				checkCudaError();
		}		

		PyObject* get_array(){

			PyObject* array_h = PyArray_SimpleNew(this->ndims, this->ins, NPY_FLOAT32);
			float* array_data = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array_h)));
			cudaMemcpy( array_data, this->array_d, this->bytes*sizeof(float), cudaMemcpyDeviceToHost );
			checkCudaError();
			return array_h;

		}

		void free_array(){
			cudaFree(this->array_d);
		}
		float* array_d;
		int ndims;
		npy_intp *ins ;
		size_t bytes;
	private: 
		PyArray_Descr* desc;
		
};


#define INDEX(dim0, dim1, dim2, dim3) \
	assert((dim1) >= 0 && (dim1) < size1 && (dim2) >= 0 && (dim2) < size2 && (dim3) >= 0 && (dim3) < size3), \
	((((dim0) * size1 + (dim1)) * size2 + (dim2)) * size3 + dim3)


template <int sgm_direction>
__global__ void sgm2(float *x0, float *x1, double *input, double *output, double *tmp, float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction, int size1, int size2, int size3, int step)
{
	int x, y, dx, dy;
	int d = threadIdx.x;

	if (sgm_direction == 0) {
		/* right */
		x = step;
		y = blockIdx.x;
		dx = 1;
		dy = 0;
	} else if (sgm_direction == 1) {
		/* left */
		x = size2 - 1 - step;
		y = blockIdx.x;
		dx = -1;
		dy = 0;
	} else if (sgm_direction == 2) {
		/* down */
		x = blockIdx.x;
		y = step;
		dx = 0;
		dy = 1;
	} else if (sgm_direction == 3) {
		/* up */
		x = blockIdx.x;
		y = size1 - 1 - step;
		dx = 0;
		dy = -1;
	}

	if (y - dy < 0 || y - dy >= size1 || x - dx < 0 || x - dx >= size2) {
		float val = input[INDEX(0, y, x, d)];
		output[INDEX(0, y, x, d)] += val;
		tmp[d * size2 + blockIdx.x] = val;
		return;
	}

	__shared__ float output_s[400], output_min[400];

	output_s[d] = output_min[d] = tmp[d * size2 + blockIdx.x];
	__syncthreads();

	for (int i = 256; i > 0; i /= 2) {
		if (d < i && d + i < size3 && output_min[d + i] < output_min[d]) {
			output_min[d] = output_min[d + i];
		}
		__syncthreads();
	}

	int ind2 = y * size2 + x;
	float D1 = COLOR_DIFF(x0, ind2, ind2 - dy * size2 - dx);
	float D2;
	int xx = x + d * direction;
	if (xx < 0 || xx >= size2 || xx - dx < 0 || xx - dx >= size2) {
		D2 = 10;
	} else {
		D2 = COLOR_DIFF(x1, ind2 + d * direction, ind2 + d * direction - dy * size2 - dx);
	}
	float P1, P2;
	if (D1 < tau_so && D2 < tau_so) {
		P1 = pi1;
		P2 = pi2;
	} else if (D1 > tau_so && D2 > tau_so) {
		P1 = pi1 / (sgm_q1 * sgm_q2);
		P2 = pi2 / (sgm_q1 * sgm_q2);
	} else {
		P1 = pi1 / sgm_q1;
		P2 = pi2 / sgm_q1;
	}

	float cost = min(output_s[d], output_min[0] + P2);
	if (d - 1 >= 0) {
		cost = min(cost, output_s[d - 1] + (sgm_direction == 2 ? P1 / alpha1 : P1));
	}
	if (d + 1 < size3) {
		cost = min(cost, output_s[d + 1] + (sgm_direction == 3 ? P1 / alpha1 : P1));
	}

	double val = (input[INDEX(0, y, x, d)] + cost - output_min[0]);
	output[INDEX(0, y, x, d)] += val;
	tmp[d * size2 + blockIdx.x] = val;
}

void sgm_w(CUDA_float_array* left, CUDA_float_array *right,CUDA_double_array* in_cost, CUDA_double_array *out_cost,CUDA_double_array* tmp, 
		float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction )
{	

	int size1 = out_cost->ins[0]*out_cost->ins[2];
	int size2 = out_cost->ins[1]*out_cost->ins[2];
	int disp_max = out_cost->ins[2];



	for (int step = 0; step < in_cost->ins[1]; step++) {
		sgm2<0><<<(size1 - 1) / disp_max + 1, disp_max>>>(
			left->array_d,
			right->array_d,
			in_cost->array_d,
			out_cost->array_d,
			tmp->array_d,
			pi1, pi2, tau_so, alpha1, sgm_q1, sgm_q2, direction,
			in_cost->ins[0],
			in_cost->ins[1],
			in_cost->ins[2],
			step);
	}	

	for (int step = 0; step < in_cost->ins[1]; step++) {
		sgm2<1><<<(size1 - 1) / disp_max + 1, disp_max>>>(
			left->array_d,
			right->array_d,
			in_cost->array_d,
			out_cost->array_d,
			tmp->array_d,
			pi1, pi2, tau_so, alpha1, sgm_q1, sgm_q2, direction,
			in_cost->ins[0],
			in_cost->ins[1],
			in_cost->ins[2],
			step);
	}

	


	for (int step = 0; step < in_cost->ins[0]; step++) {
		sgm2<2><<<(size2 - 1) / disp_max + 1, disp_max>>>(
			left->array_d,
			right->array_d,
			in_cost->array_d,
			out_cost->array_d,
			tmp->array_d,
			pi1, pi2, tau_so, alpha1, sgm_q1, sgm_q2, direction,
			in_cost->ins[0],
			in_cost->ins[1],
			in_cost->ins[2],
			step);
	}

	for (int step = 0; step < in_cost->ins[0]; step++) {
		sgm2<3><<<(size2 - 1) / disp_max + 1, disp_max>>>(
			left->array_d,
			right->array_d,
			in_cost->array_d,
			out_cost->array_d,
			tmp->array_d,
			pi1, pi2, tau_so, alpha1, sgm_q1, sgm_q2, direction,
			in_cost->ins[0],
			in_cost->ins[1],
			in_cost->ins[2],
			step);
	}

   	checkCudaError();
}	


__global__ void cross(float *x0, float *out, int size, int dim2, int dim3, int L1, float tau1)
{

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int dir = id;
		int x = dir % dim3;
		dir /= dim3;
		int y = dir % dim2;
		dir /= dim2;

		int dx = 0;
		int dy = 0;
		if (dir == 0) {
			dx = -1;
		} else if (dir == 1) {
			dx = 1;
		} else if (dir == 2) {
			dy = -1;
		} else if (dir == 3) {
			dy = 1;
		} else {
			assert(0);
		}

		int xx, yy, ind1, ind2, dist;
		ind1 = y * dim3 + x;
		for (xx = x + dx, yy = y + dy;;xx += dx, yy += dy) {
			if (xx < 0 || xx >= dim3 || yy < 0 || yy >= dim2) break;

			
			dist = max(abs(xx - x), abs(yy - y));

			if (dist == 1) continue;

			ind2 = yy * dim3 + xx;

			/* rule 1 */
			if (COLOR_DIFF(x0, ind1, ind2) >= tau1) break;

			/* rule 2 */
			if (dist >= L1) break;
		}
		out[id] = dir <= 1 ? xx : yy;

	}
}

void cross_w(CUDA_float_array* in, CUDA_float_array *out, int L1, float tau1) {


	cross<<<(out->ins[0]*out->ins[1]*out->ins[2] -1 ) / TB + 1, TB>>>(
		in->array_d,
		out->array_d,
		out->ins[0]*out->ins[1]*out->ins[2],
		out->ins[1],
		out->ins[2],
		L1, tau1);

	checkCudaError();
}


__global__ void cbca(float *x0c, float *x1c, double *vol, double *out, int size, int dim2, int dim3, int direction)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int d = id;
		int x = d % dim3;
		d /= dim3;
		int y = d % dim2;
		d /= dim2;

		if (x + d * direction < 0 || x + d * direction >= dim3) {
			out[id] = vol[id];
		} else {
			float sum = 0;
			int cnt = 0;

			int yy_s = max(x0c[(2 * dim2 + y) * dim3 + x], x1c[(2 * dim2 + y) * dim3 + x + d * direction]);
			int yy_t = min(x0c[(3 * dim2 + y) * dim3 + x], x1c[(3 * dim2 + y) * dim3 + x + d * direction]);
			for (int yy = yy_s + 1; yy < yy_t; yy++) {
				int xx_s = max(x0c[(0 * dim2 + yy) * dim3 + x], x1c[(0 * dim2 + yy) * dim3 + x + d * direction] - d * direction);
				int xx_t = min(x0c[(1 * dim2 + yy) * dim3 + x], x1c[(1 * dim2 + yy) * dim3 + x + d * direction] - d * direction);
				for (int xx = xx_s + 1; xx < xx_t; xx++) {
					float val = vol[(d * dim2 + yy) * dim3 + xx];
					assert(!isnan(val));
					sum += val;
					cnt++;
				}
			}

			assert(cnt > 0);
			out[id] = sum / cnt;
			assert(!isnan(out[id]));
		}
	}
}

void cbca_W(CUDA_float_array* leftc, CUDA_float_array *rightc,CUDA_double_array *vol, CUDA_double_array *vol_out, int direction )
{

	assert(direction == -1 or direction == 1);
	cbca<<<(vol->ins[0]*vol->ins[1]*vol->ins[2] - 1) / TB + 1, TB>>>(
		leftc->array_d,
		rightc->array_d,
		vol->array_d,
		vol_out->array_d,
		vol->ins[0]*vol->ins[1]*vol->ins[2],
		vol->ins[1],
		vol->ins[2],
		direction);
	checkCudaError();

}


__global__ void subpixel_enchancement(float *d0, double *c2, float *out, int size, int dim23, int disp_max) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int d = d0[id];
		out[id] = d;
		if (1 <= d && d < disp_max - 1) {
			float cn = c2[(d - 1) * dim23 + id];
			float cz = c2[d * dim23 + id];
			float cp = c2[(d + 1) * dim23 + id];
			float denom = 2 * (cp + cn - 2 * cz);
			if (denom > 1e-5) {
				out[id] = d - min(1.0, max(-1.0, (cp - cn) / denom));
			}
		}
	}
}

void subpixel_enchancement_w(CUDA_float_array* d, CUDA_double_array *cost,CUDA_float_array* out) {

	int disp_max = cost->ins[0];

	subpixel_enchancement<<<(d->ins[0]*d->ins[1] - 1) / TB + 1, TB>>>(
		d->array_d,
		cost->array_d,
		out->array_d,
		d->ins[0]*d->ins[1],
		d->ins[0]*d->ins[1],
		disp_max);

	checkCudaError();	
}


__global__ void median2d(float *img, float *out, int size, int dim2, int dim3, int kernel_radius)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int y = id / dim3;

		float xs[11 * 11];
		int xs_size = 0;
		for (int xx = x - kernel_radius; xx <= x + kernel_radius; xx++) {
			for (int yy = y - kernel_radius; yy <= y + kernel_radius; yy++) {
				if (0 <= xx && xx < dim3 && 0 <= yy && yy < dim2) {
					xs[xs_size++] = img[yy * dim3 + xx];
				}
			}
		}
		sort(xs, xs_size);
		out[id] = xs[xs_size / 2];
	}
}

void median2d_w(CUDA_float_array* in, CUDA_float_array* out , int kernel_size) {

	median2d<<<(in->ins[0]*in->ins[1] - 1) / TB + 1, TB>>>(
		in->array_d,
		out->array_d,
		in->ins[0]*in->ins[1],
		in->ins[0],
		in->ins[1],
		kernel_size / 2);
	checkCudaError();

}

__global__ void mean2d(float *img, float *kernel, float *out, int size, int kernel_radius, int dim2, int dim3, float alpha2)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int y = id / dim3;

		float sum = 0;
		float cnt = 0;
		int i = 0;
		for (int xx = x - kernel_radius; xx <= x + kernel_radius; xx++) {
			for (int yy = y - kernel_radius; yy <= y + kernel_radius; yy++, i++) {
				if (0 <= xx && xx < dim3 && 0 <= yy && yy < dim2 && abs(img[yy * dim3 + xx] - img[y * dim3 + x]) < alpha2) {
					sum += img[yy * dim3 + xx] * kernel[i];
					cnt += kernel[i];
				}
			}
		}
		out[id] = sum / cnt;
	}
}

void mean2d_w(CUDA_float_array* in,CUDA_float_array* out, PyObject* kernel  ,float  alpha2) {



    PyArrayObject* kernelA = reinterpret_cast<PyArrayObject*>(kernel);
	float * kernelp = reinterpret_cast<float*>(PyArray_DATA(kernelA));
	npy_intp *kernels = PyArray_DIMS(kernelA);
	float *kernel_d;
	assert(kernels[0] %2 ==1);

	cudaMalloc(&kernel_d, kernels[0]*kernels[1]*sizeof(float));
    cudaMemcpy( kernel_d, kernelp, kernels[0]*kernels[1]*sizeof(float), cudaMemcpyHostToDevice);

	mean2d<<<(in->ins[0]*in->ins[1] - 1) / TB + 1, TB>>>(
		in->array_d,
		kernel_d,
		out->array_d,
		in->ins[0]*in->ins[1],
		kernels[0] / 2,
		in->ins[0],
		in->ins[1],
		alpha2);
	checkCudaError();
    cudaFree(kernel_d);
}


__global__ void swap_axis(const double* __restrict__ cost, double* temp_cost, const int rows, const int cols, const int ndisp ){

	//const int Row =blockIdx.y*blockDim.y+ threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x; 
    int Row = blockIdx.y*BLOCK_D_SIZE + threadIdx.y; 

    __shared__ double tile[BLOCK_D_SIZE][BLOCK_D_SIZE+1];



    if( Col< cols*rows ){
    	#pragma unroll
    	for(int d=0; d<BLOCK_D_SIZE; d+=16){
    		if((Row+d)<ndisp)
    			tile[threadIdx.y+d][threadIdx.x] = cost [(Row+d)*rows*cols+Col ];
    	}	
    }


    	__syncthreads();

    	Col = blockIdx.x*blockDim.x+threadIdx.y;
    	Row = blockIdx.y*BLOCK_D_SIZE+threadIdx.x; 
    
    	#pragma unroll
    	for(int d=0; d<BLOCK_D_SIZE; d+=16){
    		if((Col+d) < cols*rows && Row<ndisp)
    			temp_cost[ (Col+d)*ndisp+Row ] = tile[threadIdx.x][threadIdx.y+d];
	   	}
	    
	


}

void swap_axis_w(CUDA_double_array* in,CUDA_double_array* out) {


    dim3 swapBlock(BLOCK_D_SIZE,16,1);
    dim3 swapGrid(ceil((float)in->ins[2]*in->ins[1]/BLOCK_D_SIZE),ceil((float) in->ins[0]/BLOCK_D_SIZE ));	

    swap_axis<<< swapGrid, swapBlock >>>( in->array_d, out->array_d,in->ins[1],in->ins[2],in->ins[0] );

    out->ins[0]=in->ins[1];
    out->ins[1]=in->ins[2];
    out->ins[2]=in->ins[0];
	checkCudaError();
}


__global__ void swap_axis_back(const double* __restrict__ cost, double* temp_cost, const int rows, const int cols, const int ndisp ){

    int Col = blockIdx.x*blockDim.x + threadIdx.y;
    int Row = blockIdx.y*BLOCK_D_SIZE+threadIdx.x; 

    __shared__ double tile[BLOCK_D_SIZE][BLOCK_D_SIZE+1];



    if( Col< cols*rows && Row<ndisp){
    	#pragma unroll
    	for(int d=0; d<BLOCK_D_SIZE; d+=16){

    		tile[threadIdx.y+d][threadIdx.x] = cost [(Col+d)*ndisp+Row  ];
    	}
    }


    	__syncthreads();


    	Col = blockIdx.x*blockDim.x + threadIdx.x; 
    	Row = blockIdx.y*BLOCK_D_SIZE + threadIdx.y; 
    
    	#pragma unroll
    	for(int d=0; d<BLOCK_D_SIZE; d+=16){
    		if((Col+d) < cols*rows && (Row+d)<ndisp)
    			temp_cost[ (Row+d)*rows*cols+Col ] = tile[threadIdx.x][threadIdx.y+d];
	   	}
	    
	


}

void swap_axis_back_w(CUDA_double_array* in,CUDA_double_array* out) {



    dim3 swapBlock(BLOCK_D_SIZE,16,1);
    dim3 swapGrid(ceil((float)in->ins[1]*in->ins[0]/BLOCK_D_SIZE),ceil((float) in->ins[2]/BLOCK_D_SIZE ));	

    swap_axis_back<<< swapGrid, swapBlock >>>( in->array_d, out->array_d,in->ins[0],in->ins[1],in->ins[2] );

    out->ins[0]=in->ins[2];
    out->ins[1]=in->ins[0];
    out->ins[2]=in->ins[1];
	checkCudaError();
}

template <typename T>
__global__ void divide_kernel( T* array, double val , const int els ){
	const int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < els)
		array[index] /= val; 

}

void CUDA_divide_double(CUDA_double_array* in, double val){

	dim3 Block(MAX_THREADS);
    dim3 Grid(ceil((float)in->bytes/MAX_THREADS));	

    divide_kernel<<< Grid,Block>>>( in->array_d, val, in->bytes );

}


template <typename T>
__global__ void  copy_kernel(const T* __restrict__ in, T* out, const int els ){

	const int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < els)
		out[index] = in[index]; 
}

void CUDA_copy_double(CUDA_double_array* in, CUDA_double_array* out){

	dim3 Block(MAX_THREADS);
    dim3 Grid(ceil((float)in->bytes/MAX_THREADS));	

    copy_kernel<<< Grid,Block>>>( in->array_d,out->array_d, in->bytes );

    out->ins[0] = in->ins[0];
    out->ins[1] = in->ins[1];
    out->ins[2] = in->ins[2];

}

void CUDA_copy_float(CUDA_float_array* in, CUDA_float_array* out){

	dim3 Block(MAX_THREADS);
    dim3 Grid(ceil((float)in->bytes/MAX_THREADS));	

    copy_kernel<<< Grid,Block>>>( in->array_d,out->array_d, in->bytes );

}

__global__ void outlier_detection(float *d0, float *d1, float *outlier, int size, int dim3, int disp_max)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int d0i = d0[id];
		if (x - d0i < 0) {
			//assert(0);
			outlier[id] = 1;
		}
		 else if (abs(d0[id] - d1[id - d0i]) < 1.1) {
			outlier[id] = 0; /* match */
		} else {
			outlier[id] = 1; /* occlusion */
			for (int d = 0; d < disp_max; d++) {
				if (x - d >= 0 && abs(d - d1[id - d]) < 1.1) {
					outlier[id] = 2; /* mismatch */
					break;
				}
			}
		}
	}
}

void outlier_detection_w(CUDA_float_array* d0, CUDA_float_array* d1, CUDA_float_array* outlier, int disp_max)
{
	

	outlier_detection<<<(d0->ins[0]*d0->ins[1] - 1) / TB + 1, TB>>>(
		d0->array_d,
		d1->array_d,
		outlier->array_d,
		d0->ins[0]*d0->ins[1],
		d0->ins[1],
		disp_max);
	checkCudaError();

}


__global__ void interpolate_mismatch(float *d0, float *outlier, float *out, int size, int dim2, int dim3)
{
	const float dir[] = {
		0	,  1,
		-0.5,  1,
		-1	,  1,
		-1	,  0.5,
		-1	,  0,
		-1	, -0.5,
		-1	, -1,
		-0.5, -1,
		0	, -1,
		0.5 , -1,
		1	, -1,
		1	, -0.5,
		1	,  0,
		1	,  0.5,
		1	,  1,
		0.5 ,  1
	};

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		if (outlier[id] != 2) {
			out[id] = d0[id];
			return;
		}

		float vals[16];
		int vals_size = 0;

		int x = id % dim3;
		int y = id / dim3;
		for (int d = 0; d < 16; d++) {
			float dx = dir[2 * d];
			float dy = dir[2 * d + 1];
			float xx = x;
			float yy = y;
			int xx_i = round(xx);
			int yy_i = round(yy);
			while (0 <= yy_i && yy_i < dim2 && 0 <= xx_i && xx_i < dim3 && outlier[yy_i * dim3 + xx_i] == 2) {
				xx += dx;
				yy += dy;
				xx_i = round(xx);
				yy_i = round(yy);
			}

			int ind = yy_i * dim3 + xx_i;
			if (0 <= yy_i && yy_i < dim2 && 0 <= xx_i && xx_i < dim3) {
				assert(outlier[ind] != 2);
				vals[vals_size++] = d0[ind];
			}
		}
		assert(vals_size > 0);
		sort(vals, vals_size);
		out[id] = vals[vals_size / 2];
	}
}

void interpolate_mismatch_w(CUDA_float_array* d0,CUDA_float_array* outlier,CUDA_float_array* out)
{

	interpolate_mismatch<<<(d0->ins[0]*d0->ins[1] - 1) / TB + 1, TB>>>(
		d0->array_d,
		outlier->array_d,
		out->array_d,
		d0->ins[0]*d0->ins[1],
		d0->ins[0],
		d0->ins[1]);
	checkCudaError();
}

__global__ void interpolate_occlusion(float *d0, float *outlier, float *out, int size, int dim3)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		if (outlier[id] != 1) {
			out[id] = d0[id];
			return;
		}
		int x = id % dim3;

		int dx = 0;
		while (x + dx >= 0 && outlier[id + dx] != 0) {
			dx--;
		}
		if (x + dx < 0) {
			dx = 0;
			while (x + dx < dim3 && outlier[id + dx] != 0) {
				dx++;
			}
		}
		if (x + dx < dim3) {
			out[id] = d0[id + dx];
		} else {
			out[id] = d0[id];
		}
	}
}

void interpolate_occlusion_w(CUDA_float_array* d0,CUDA_float_array* outlier,CUDA_float_array* out)
{

	interpolate_occlusion<<<(d0->ins[0]*d0->ins[1] - 1) / TB + 1, TB>>>(
		d0->array_d,
		outlier->array_d,
		out->array_d,
		d0->ins[0]*d0->ins[1],
		d0->ins[1]
	);

	checkCudaError();
}




bool InitCUDA(void)
{
    int count = 0;
    int i = 0;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }
    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        return false;
    }
    cudaSetDevice(i);

    printf("CUDA initialized.\n");
    return true;
}



BOOST_PYTHON_MODULE(post) {
    omp_set_num_threads(1);
    numeric::array::set_module_and_type("numpy", "ndarray");
    def("InitCUDA",InitCUDA);
    def("sgm",sgm_w);
    def("cross",cross_w);
    def("cbca",cbca_W);
    def("subpixel_enchancement",subpixel_enchancement_w);
    def("median2d",median2d_w);
    def("mean2d",mean2d_w);
    def("swap_axis",swap_axis_w);
    def("swap_axis_back",swap_axis_back_w);
    def("CUDA_copy_double",CUDA_copy_double);
    def("CUDA_copy_float",CUDA_copy_float);
    def("CUDA_divide_double",CUDA_divide_double);
    def("outlier_detection",outlier_detection_w);
    def("interpolate_mismatch",interpolate_mismatch_w);
    def("interpolate_occlusion",interpolate_occlusion_w);
    class_< CUDA_double_array>("CUDA_double_array", init<PyObject*>())
    	.def("set_to_zero",&CUDA_double_array::set_to_zero)
    	.def("get_array",&CUDA_double_array::get_array)
    	.def("free_array",&CUDA_double_array::free_array);
    class_< CUDA_float_array>("CUDA_float_array", init<PyObject*>())
    	.def("set_to_zero",&CUDA_float_array::set_to_zero)
    	.def("get_array",&CUDA_float_array::get_array)
    	.def("free_array",&CUDA_float_array::free_array);    	

    import_array();
}


