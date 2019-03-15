################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../featextract.cpp 

OBJS += \
./featextract.o 

CPP_DEPS += \
./featextract.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -O0 -g3 -Wall -c -fmessage-length=0 -mavx2 -fopenmp -ftree-vectorize -fPIC -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


