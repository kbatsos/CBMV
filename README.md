# CBMV: A Coalesced Bidirectional Matching Volume for Disparity Estimation
CBMV: A Coalesced Bidirectional Matching Volume for Disparity Estimation code repository. If you use this code please cite our paper [CBMV: A Coalesced Bidirectional Matching Volume for Disparity Estimation](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0598.pdf).

```
@inproceedings{batsos2018cbmv,
  title={CBMV: A Coalesced Bidirectional Matching Volume for Disparity Estimation},
  author={Batsos, Konstantinos and Cai, Changjiang and Mordohai, Philipos},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```

---

The repository contains the code, models and procudures for training and testing.
The code includes the cost-optimization and post-processing of 
MC-CNN [Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches](https://arxiv.org/abs/1510.05970) 
as implemented by Jure Zbontar, and modified to interface with python. We aslo incorporate, as an alternative cost-optimization method, 
the code from [Continuous 3D Label Stereo Matching using Local Expansion Moves](https://arxiv.org/pdf/1603.08328.pdf).

# Links

[Training Data](https://drive.google.com/file/d/1RKIhAT5mc9kyWFg9Trg0Ze4qZMzUnbPU/view?usp=sharing)

[CBMV model](https://drive.google.com/file/d/1mjz-Rttdf99BZxne5EFziY5HwB10QPPF/view?usp=sharing): the trained random forest model, with version 0.17.1 sklearn.


# Run CBMV

To run CBMV you need the following python packages:

## Python

- numpy
- scipy
- sklearn
- cPickle 

Much of CBMV code is written as C++ libraries and interface with python via boost python. To compile the C++ libraries you will need the following:

## C++

- OpenCV 3.0
- Boost ( boost python )
- CUDA

## Compilation

Assuming this repository is rooted at "~/cbmv-cvpr18/", the OpenCV library is installed at `/usr/local/opencv-3.2.0`.

- 1) To compile `matchers`, `featextract` and `rectification`:  
We provide both `makefile` and `CMakeLists.txt`(**preferred**) for compilation. If you have installed OpenCV at different directory, please update the line **`g++ -I/usr/local/opencv-3.2.0/include/ ...`**  in the file `*/Debug/subdir.mk` for including header files; and modify the line **`g++ ... -L/usr/local/opencv-3.2.0/lib ...`** in the file `*/Debug/makefile` for library linking. Or to modify the line **`set(OpenCV_DIR "/usr/local/opencv-3.2.0/share/OpenCV/")`** in the file `CMakeList.txt`.

```bash
# 1) Method 1 : using CMake, will generate the libraries in the directory `~/cbmv-cvpr18/cpp/lib`.
cd ~/cbmv-cvpr18/cpp/ && mkdir build && cd build && cmake .. && build 

# 2) Method 2 : using makefile
#To compile `matchers` for four basic matchers:
cd ~/cbmv-cvpr18/cpp/matchers/Debug && make clean && make
# To compile `featextract` for feature extraction:
cd ~/cbmv-cvpr18/cpp/featextract/Debug && make clean && make
# To compile `rectification` for rectifying stereo image pair:
cd ~/cbmv-cvpr18/cpp/rectification/Debug && make clean && make
```

- 2) To compile `post` for postprocessing:  
This part contains the GPU code. We assume the CUDA was installed at `/usr/local/cuda-8.0/`. If not, please modify 
the line **`CUDA_LIBS = /usr/local/cuda-8.0/lib64`** in the file `*/post/makefile`.

```bash
cd ~/cbmv-cvpr18/cpp/post/ && make clean && make
```

- 3) To compile `localexp` for Local Expansion Moves:  
This part requires the Maxflow code by Boykov and Kolmogorov 
[[Code v3.01]](http://vision.csd.uwo.ca/code) [[Code v3.04]](http://pub.ist.ac.at/~vnk/software.html).
For your convenience, we already include and confiure it. But please note they are freely available for **research purposes only**. You could also check 
the [Local Expansion Move repository](https://github.com/t-taniai/LocalExpStereo) for the license.  For `localexp` compilation, we provide the `CMakeLists.txt` file. 
Run the following
```bash
# will generate the libraries in the directory `*/localexp/lib`.
cd ~/cbmv-cvpr18/cpp/localexp && mkdir build && cd build && cmake .. && make
```
You can change the OMP threads number defined by `#define THREADS_NUM_USED 16` at file `~/cbmv-cvpr18/cpp/paramSetting.hpp`, for efficient parallel computation.

---

After compiling the C++ libraries, downloading the required python packages and precomputed model (`Note: pay attention to the version of sklearn`), CBMV can be executed using the following command

```
python main.py  --l ./datasets/ArtL/im0.png --r ./datasets/ArtL/im1.png --w 694 --h 554 --d 128 --model ./modelall_4m.rf --prob_save_path ./
```

For a complete list of arguments please see tha `main.py` file. We provide a script `run_test_cbmv.sh` to run CBMV for testing and see more details in section [Disparity Estimation](#disparity-estimation).


# Training 

 The above link includes the data we used to trained CBMV. To train CBMV you can use the following command:

```
python main.py --train --data_path "./datasets/" --train_add ./mb/additional.txt --train_set ./mb/trainall.txt --model ./models/model.rf
```
The `*.txt` files specify the training sets used during the training process. For a complete list of arguments please see tha `main.py` file. We also provide a script `run_train_cbmv.sh` to train CBMV.


# Disparity Estimation

Giveing the cost volume, there are two methods for cost-optimization in order to generate the disparity map as output.

- Post-processing used by [MC-CNN](https://arxiv.org/abs/1510.05970): see the code `~/cbmv-cvpr18/cpp/post/post.cu` for details.
- [Local Expansion Moves](https://github.com/t-taniai/LocalExpStereo): see the codes `~/cbmv-cvpr18/cpp/localexp/local_exp_stereo.cpp` for more details. You can also modify the hyperparameters defined in function `__postprocessing_localExp(...)` in the file `~/cbmv-cvpr18/test.py`.
