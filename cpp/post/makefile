#Set the compiler

CC = g++
NV = nvcc
#Set compile time flas

CFLAGS= -O3 -g3  -Wall -c -fmessage-length=0 -msse4 -fPIC -MMD -MP
LFLAGS= -shared -Wl,--export-dynamic
# location of the Python header files

PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

#added by CCJ
NUMPY_INCLUDE = /usr/local/lib/python$(PYTHON_VERSION)/dist-packages/numpy/core/include

# location of the Boost Python include files and library

BOOST_INC = /usr/include
BOOST_LIB = /usr/lib
CUDA_LIBS = /usr/local/cuda-8.0/lib64

# compile mesh classes
TARGET = post

$(TARGET).so: $(TARGET).cu
	$(NV) -arch=sm_52 --default-stream per-thread  -O3 -L$(BOOST_LIB) -lboost_python -lpython$(PYTHON_VERSION) -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -I$(BOOST_INC) -Xcompiler -fopenmp -Xcompiler -fPIC --shared $(TARGET).cu -o $(TARGET).so


clean:
	 rm -rf *.so *.o
