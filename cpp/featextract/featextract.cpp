#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <iostream>
#include <stdlib.h>
#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include <numpy/ndarrayobject.h>
#include <algorithm>
#include <vector>
#include <stdint.h>
#include <assert.h>
#include <immintrin.h>
#include <omp.h>
#include <sstream>
//added by CCJ: for macro THREADS_NUM_USED
#include "../paramSetting.hpp"

using namespace std;
using namespace boost::python;
typedef uint8_t uint8;
typedef int16_t int16;
typedef unsigned long long int uint64;

PyObject* swap_axes(PyObject* cost ){


    PyArrayObject* costA = reinterpret_cast<PyArrayObject*>(cost);
    double * costp = reinterpret_cast<double*>(PyArray_DATA(costA));
    npy_intp *shape = PyArray_DIMS(costA);

    npy_intp *res_shape = new npy_intp[3];
    res_shape[0] = shape[1]; res_shape[1] = shape[2], res_shape[2] = shape[0] ;


	PyObject* res = PyArray_SimpleNew(3, res_shape, NPY_FLOAT64);

	double* res_data = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(res)));

//#pragma omp parallel num_threads(12)
// added by CCJ for using THREADS_NUM_USED
#pragma omp parallel num_threads(THREADS_NUM_USED)
	{
		#pragma omp for
		for(int i=0; i<res_shape[0]*res_shape[1]; i++){
			for(int j=0; j<res_shape[2]; j++){

				res_data[ i*res_shape[2] +j ] = costp[ j*res_shape[0]*res_shape[1] +i];
			}
		}
	}

	return res;
}

PyObject* swap_axes_back(PyObject* cost ){


    PyArrayObject* costA = reinterpret_cast<PyArrayObject*>(cost);
    double * costp = reinterpret_cast<double*>(PyArray_DATA(costA));
    npy_intp *shape = PyArray_DIMS(costA);

    npy_intp *res_shape = new npy_intp[3];
    res_shape[0] = shape[2]; res_shape[1] = shape[0], res_shape[2] = shape[1] ;


	PyObject* res = PyArray_SimpleNew(3, res_shape, NPY_FLOAT64);

	double* res_data = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(res)));

//#pragma omp parallel num_threads(12)
#pragma omp parallel num_threads(THREADS_NUM_USED)
	{
		#pragma omp for
		for(int i=0; i<res_shape[1]*res_shape[2]; i++){
			for(int j=0; j<res_shape[0]; j++){

				res_data[  j*res_shape[1]*res_shape[2] +i  ] = costp[i*res_shape[0] +j];
			}
		}
	}

	return res;
}

PyObject* get_cost(PyObject* cost ){


    PyArrayObject* costA = reinterpret_cast<PyArrayObject*>(cost);
    double * costp = reinterpret_cast<double*>(PyArray_DATA(costA));
    npy_intp *shape = PyArray_DIMS(costA);

    npy_intp *res_shape = new npy_intp[3];
    res_shape[0] = shape[0]; res_shape[1] = shape[1], res_shape[2] = shape[2] ;


	PyObject* res = PyArray_SimpleNew(3, res_shape, NPY_FLOAT64);

	double* res_data = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(res)));

//#pragma omp parallel num_threads(12)
#pragma omp parallel num_threads(THREADS_NUM_USED)
	{
		#pragma omp for
		for(int i=0; i<shape[0]*shape[1]; i++){
			for(int j=0; j<shape[2]; j++){

				res_data[ i*shape[2] +j ] = costp[ i*shape[2]*shape[3] +j*shape[3]];
			}
		}
	}

	return res;
}

PyObject* get_right_cost(PyObject* cost ){


    PyArrayObject* costA = reinterpret_cast<PyArrayObject*>(cost);
    double * costp = reinterpret_cast<double*>(PyArray_DATA(costA));
    npy_intp *shape = PyArray_DIMS(costA);

    npy_intp *res_shape = new npy_intp[3];
    res_shape[0] = shape[0]; res_shape[1] = shape[1], res_shape[2] = shape[2] ;


	PyObject* res = PyArray_SimpleNew(3, res_shape, NPY_FLOAT64);

	double* res_data = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(res)));
	int fill_size = shape[0]*shape[1]*shape[2];
	std::fill_n(res_data,fill_size, costp[0]);

//#pragma omp parallel num_threads(12)
#pragma omp parallel num_threads(THREADS_NUM_USED)
	{
		#pragma omp for
		for(int d=0; d<shape[2]; d++){
			for(int i=0; i<shape[0]; i++){

				for(int j=0; j<(shape[1]-d); j++){

					res_data[ i*shape[1]*shape[2] + j*shape[2] +d ] = costp[  i*shape[1]*shape[2] +(j+d)*shape[2] + d ] ;

					//res_data[ d*shape[1]*shape[2] +i*shape[2] + j ] = costp[  d*shape[1]*shape[2] +i*shape[2] + (j+d) ] ;

				}

			}
		}
	}

	return res;
}

PyObject* generate_d_indices(
		PyObject* gt,
		int maxd,
		int dispThresh
		){
	PyArrayObject* gtA = reinterpret_cast<PyArrayObject*>(gt);
	int * gtp = reinterpret_cast<int*>(PyArray_DATA(gtA));

	npy_intp *samples = PyArray_DIMS(gtA);
	npy_intp *sshape = new npy_intp[2];
	sshape[0] = samples[0];sshape[1]=3;

	PyObject* rs= PyArray_SimpleNew(2, sshape, NPY_INT32);
	int* rsd = static_cast<int*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(rs)));
	std::fill_n(rsd, sshape[0]*sshape[1], 0);
	std::cout << samples[0] << std::endl;

//#pragma omp parallel num_threads(12)
#pragma omp parallel num_threads(THREADS_NUM_USED)
		{
    #pragma omp for
		for(long int s=0; s<samples[0]; s++){
//			if(gtp[s] > maxd || gtp[s]<0)
//				continue;
			assert(gtp[s] <= maxd);
			assert(gtp[s]>=0);



			rsd[3*s] = gtp[s];
			bool low = true;
			bool high = true;

			if(gtp[s]- dispThresh <=0 )
				low = false;

			if(gtp[s]+ dispThresh >= maxd)
				high = false;
			int d1 = 0;
			int d2 = 0;

			if( low ){
				d1 = rand()%(gtp[s] - dispThresh);
			}
			else{
				d1 = rand()%(maxd - (gtp[s] + dispThresh+ 1)) + (gtp[s] + dispThresh+1);
			}
			if( high){
				d2 = rand()%(maxd - (gtp[s] + dispThresh +1)) + (gtp[s] + dispThresh+1);
			}
			else{
				d2 = rand()%(gtp[s] - dispThresh);
			}

			rsd[3*s+1] = d1;
			rsd[3*s+2] = d2;

		}
	}

	return rs;
}

PyObject* get_samples(
		PyObject* vol,
		PyObject* r_samp
		){
	PyArrayObject* volA = reinterpret_cast<PyArrayObject*>(vol);
	double * volp = reinterpret_cast<double*>(PyArray_DATA(volA));

	PyArrayObject* r_sampA = reinterpret_cast<PyArrayObject*>(r_samp);
	int * r_sampp = reinterpret_cast<int*>(PyArray_DATA(r_sampA));


	npy_intp *vol_shape = PyArray_DIMS(volA);
	npy_intp *samples = PyArray_DIMS(r_sampA);
	npy_intp *sshape = new npy_intp[1];
	sshape[0] = samples[0]*samples[1];

	PyObject* res= PyArray_SimpleNew(1, sshape, NPY_FLOAT64);
	double* resd = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(res)));

//#pragma omp parallel num_threads(12)
#pragma omp parallel num_threads(THREADS_NUM_USED)
		{
    #pragma omp for
		for(long int s=0; s<samples[0]; s++){


			resd[3*s] = volp[s*vol_shape[1]+r_sampp[3*s]];

			resd[3*s+1] =volp[s*vol_shape[1]+r_sampp[3*s+1]];
			resd[3*s+2] =volp[s*vol_shape[1]+r_sampp[3*s+2]];

		}
	}

	return res;
}

PyObject* extract_pkrn(
		PyObject* vol,
		PyObject* r_samp,
		double e
		){

	PyArrayObject* volA = reinterpret_cast<PyArrayObject*>(vol);
	double * volp = reinterpret_cast<double*>(PyArray_DATA(volA));

	PyArrayObject* r_sampA = reinterpret_cast<PyArrayObject*>(r_samp);
	int * r_sampp = reinterpret_cast<int*>(PyArray_DATA(r_sampA));

	npy_intp *vol_shape = PyArray_DIMS(volA);
	npy_intp *samples = PyArray_DIMS(r_sampA);
	npy_intp *sshape = new npy_intp[1];
	sshape[0] = samples[0]*samples[1];

	PyObject* res= PyArray_SimpleNew(1, sshape, NPY_FLOAT64);
	double* resd = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(res)));

//#pragma omp parallel num_threads(12)
#pragma omp parallel num_threads(THREADS_NUM_USED)
	{
		#pragma omp for
		for (long int i= 0; i <vol_shape[0]; i++ ){
			const unsigned long int pos = i* vol_shape[1];
			double min_cost = RAND_MAX;

			for(int k=0; k< vol_shape[1]; k++){
					if(volp[pos+k] < min_cost){
						min_cost = volp[pos+k];
					}

			}


			resd[3*i] = (min_cost == RAND_MAX)? 0.0 :( min_cost +e ) / (volp[pos+r_sampp[3*i]]+e);
			resd[3*i+1] = (min_cost == RAND_MAX)? 0.0 :( min_cost +e ) / (volp[pos+r_sampp[3*i+1]]+e);
			resd[3*i+2] = (min_cost == RAND_MAX)? 0.0 :( min_cost +e ) / (volp[pos+r_sampp[3*i+2]]+e);



		}/*end of each sample*/

	}/*end of omp*/
	return res;
}


PyObject* extract_pkrn_test(
		PyObject* vol,
		double e
		){

	PyArrayObject* volA = reinterpret_cast<PyArrayObject*>(vol);
	double * volp = reinterpret_cast<double*>(PyArray_DATA(volA));

	npy_intp *vol_shape = PyArray_DIMS(volA);

	PyObject* res= PyArray_SimpleNew(2, vol_shape, NPY_FLOAT64);
	double* resd = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(res)));

//#pragma omp parallel num_threads(12)
#pragma omp parallel num_threads(THREADS_NUM_USED)
	{
		#pragma omp for
		for (long int i= 0; i <vol_shape[0]; i++ ){
			const unsigned long int pos = i* vol_shape[1];
			double min_cost = RAND_MAX;

			for(int k=0; k< vol_shape[1]; k++){
					if(volp[pos+k] < min_cost){
						min_cost = volp[pos+k];
					}

			}


			for(int j=0; j< vol_shape[1]; j++){
				resd[pos+j] = (min_cost == RAND_MAX)? 0.0 :( min_cost +e ) / (volp[pos+j]+e);


			}

		}/*end of each sample*/

	}/*end of omp*/
	return res;
}


PyObject* extract_aml(
		PyObject* vol,
		PyObject* r_samp,
		double sigma
		){

	PyArrayObject* volA = reinterpret_cast<PyArrayObject*>(vol);
	double * volp = reinterpret_cast<double*>(PyArray_DATA(volA));

	PyArrayObject* r_sampA = reinterpret_cast<PyArrayObject*>(r_samp);
	int * r_sampp = reinterpret_cast<int*>(PyArray_DATA(r_sampA));

	npy_intp *vol_shape = PyArray_DIMS(volA);
	npy_intp *samples = PyArray_DIMS(r_sampA);
	npy_intp *sshape = new npy_intp[1];
	sshape[0] = samples[0]*samples[1];

	PyObject* res= PyArray_SimpleNew(1, sshape, NPY_FLOAT64);
	double* resd = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(res)));

//#pragma omp parallel num_threads(12)
#pragma omp parallel num_threads(THREADS_NUM_USED)
	{
		#pragma omp for
		for (long int i= 0; i <vol_shape[0]; i++ ){
			const unsigned long int pos = i* vol_shape[1];
			double min_cost = RAND_MAX,denom=0,num = 0;

			for(int k=0; k< vol_shape[1]; k++){
					if(volp[pos+k] < min_cost){
						min_cost = volp[pos+k];
					}

			}

			for(int k=0; k<vol_shape[1]; k++){
					num = volp[pos+k]-min_cost;
					denom += exp( -( num*num )/sigma );
			}

			double tmp_cost = volp[pos+r_sampp[3*i]] - min_cost;
			resd[3*i]=(min_cost == RAND_MAX)? 0.0 : exp( -((tmp_cost * tmp_cost) /sigma )) /denom ;
			tmp_cost = volp[pos+r_sampp[3*i+1]] - min_cost;
			resd[3*i+1] = (min_cost == RAND_MAX)? 0.0 : exp( -((tmp_cost * tmp_cost) /sigma )) /denom ;
			tmp_cost = volp[pos+r_sampp[3*i+2]] - min_cost;
			resd[3*i+2] = (min_cost == RAND_MAX)? 0.0 : exp( -((tmp_cost * tmp_cost) /sigma )) /denom ;



		}/*end of each sample*/

	}/*end of omp*/

	return res;
}


PyObject* extract_aml_testing(
		PyObject* vol,
		double sigma
		){

	PyArrayObject* volA = reinterpret_cast<PyArrayObject*>(vol);
	double * volp = reinterpret_cast<double*>(PyArray_DATA(volA));


	npy_intp *vol_shape = PyArray_DIMS(volA);


	PyObject* res= PyArray_SimpleNew(2, vol_shape, NPY_FLOAT64);
	double* resd = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(res)));

//#pragma omp parallel num_threads(12)
#pragma omp parallel num_threads(THREADS_NUM_USED)
	{
		#pragma omp for
		for (long int i= 0; i <vol_shape[0]; i++ ){
			const unsigned long int pos = i* vol_shape[1];
			double min_cost = RAND_MAX,denom=0,num = 0;

			for(int k=0; k< vol_shape[1]; k++){
					if(volp[pos+k] < min_cost){
						min_cost = volp[pos+k];
					}

			}

			for(int k=0; k<vol_shape[1]; k++){
					num = volp[pos+k]-min_cost;
					denom += exp( -( num*num )/sigma );
			}

			for(int j=0; j< vol_shape[1]; j++){

				double tmp_cost = volp[pos+j] - min_cost;
				resd[pos+j]=(min_cost == RAND_MAX)? 0.0 : exp( -((tmp_cost * tmp_cost) /sigma )) /denom ;
			}



		}/*end of each sample*/

	}/*end of omp*/

	return res;
}

PyObject* get_left_cost_from_right(PyObject* cost ){


    PyArrayObject* costA = reinterpret_cast<PyArrayObject*>(cost);
    double * costp = reinterpret_cast<double*>(PyArray_DATA(costA));
    npy_intp *shape = PyArray_DIMS(costA);

    npy_intp *res_shape = new npy_intp[3];
    res_shape[0] = shape[0]; res_shape[1] = shape[1], res_shape[2] = shape[2] ;


	PyObject* res = PyArray_SimpleNew(3, res_shape, NPY_FLOAT64);

	double* res_data = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(res)));
	int fill_size = shape[0]*shape[1]*shape[2];
	std::fill_n(res_data,fill_size, costp[0]);

//#pragma omp parallel num_threads(12)
#pragma omp parallel num_threads(THREADS_NUM_USED)
	{
		#pragma omp for
		for(int d=0; d<shape[2]; d++){
			for(int i=0; i<shape[0]; i++){

				for(int j=d; j<shape[1]; j++){

					res_data[ i*shape[1]*shape[2] + j*shape[2] +d ] = costp[  i*shape[1]*shape[2] +(j-d)*shape[2] + d ] ;
					//res_data[ d*shape[1]*shape[2] +i*shape[2] + j ] = costp[  d*shape[1]*shape[2] +i*shape[2] + (j+d) ] ;

				}

			}
		}
	}

	return res;
}

PyObject* generate_labels(PyObject* rsamp ){


    PyArrayObject* rsampA = reinterpret_cast<PyArrayObject*>(rsamp);
    npy_intp *shape = PyArray_DIMS(rsampA);

    npy_intp *res_shape = new npy_intp[1];
    res_shape[0] = shape[0]*shape[1];

	PyObject* res = PyArray_SimpleNew(1, res_shape, NPY_INT32);

	int* res_data = static_cast<int*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(res)));

//#pragma omp parallel num_threads(12)
#pragma omp parallel num_threads(THREADS_NUM_USED)
	{
		#pragma omp for
		for(long int d=0; d<shape[0]; d++){
			res_data[3*d]=1;
			res_data[3*d+1]=0;
			res_data[3*d+2]=0;

		}
	}

	return res;
}


BOOST_PYTHON_MODULE(libfeatextract) {

	//omp_set_num_threads(12);
    omp_set_num_threads(THREADS_NUM_USED);

    numeric::array::set_module_and_type("numpy", "ndarray");

    def("get_cost", get_cost);
    def("get_right_cost", get_right_cost);
    def("swap_axes", swap_axes);
    def("swap_axes_back",swap_axes_back);
	def("generate_d_indices",  generate_d_indices);    
    def("get_samples",  get_samples);
    
    def("extract_ratio",  extract_pkrn);
    def("extract_ratio",  extract_pkrn_test);
    def("extract_likelihood",extract_aml);
    def("extract_likelihood",extract_aml_testing);
    def("get_left_cost",get_left_cost_from_right);
    def("generate_labels",generate_labels);
    import_array();
}
