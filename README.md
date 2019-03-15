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

[CBMV model](https://drive.google.com/file/d/1mjz-Rttdf99BZxne5EFziY5HwB10QPPF/view?usp=sharing)


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
If you have installed OpenCV at different directory, please update the line `g++ -I/usr/local/opencv-3.2.0/include/ ...` 
in the file `*/Debug/subdir.mk` for including header files; and modify the line `g++ ... -L/usr/local/opencv-3.2.0/lib ...` 
in the file `*/Debug/makefile` for library linking.

- To compile `matchers` for four basic matchers:
```bash
cd ~/cbmv-cvpr18/cpp/matchers/Debug && make clean && make
```

- To compile `featextract` for feature extraction:
```bash
cd ~/cbmv-cvpr18/cpp/featextract/Debug && make clean && make
```

- To compile `rectification` for rectifying stereo image pair:
```bash
cd ~/cbmv-cvpr18/cpp/rectification/Debug && make clean && make
```

- To compile `post` for postprocessing:  
This part contains the GPU code. We assume the CUDA was installed at `/usr/local/cuda-8.0/`. If not, please modify 
the line `CUDA_LIBS = /usr/local/cuda-8.0/lib64` in the file `*/post/makefile`.

```bash
cd ~/cbmv-cvpr18/cpp/post/ && make clean && make
```

- To compile `localexp` for Local Expansion Moves:  
This part requires the Maxflow code by Boykov and Kolmogorov 
[[Code v3.01]](http://vision.csd.uwo.ca/code) [[Code v3.04]](http://pub.ist.ac.at/~vnk/software.html).
For your convenience, we already include and confiure it. But please note they are freely available for **research purposes only**. You could also check 
the [Local Expansion Move repository](https://github.com/t-taniai/LocalExpStereo) for the license.  For `localexp` compilation, we provide the `CMakeLists.txt` file. Run the following
```bash
cd ~/cbmv-cvpr18/cpp/localexp && mkdir build && cd build && cmake .. && make
```
 will generate the libraries in the directory `*/localexp/lib`.

---

After compiling the C++ libraries, downloading the required python packages and precomputed model, CBMV can be executed using the following command:

```
python main.py  --l ./datasets/ArtL/im0.png --r ./datasets/ArtL/im1.png --w 694 --h 554 --d 128 --model ./modelall_4m.rf --prob_save_path ./
```

For a complete list of arguments please see tha main.py file. 


# Training 

 The above link includes the data we used to trained CBMV. To train CBMV you can use the following command:

```
python main.py --train --data_path "./datasets/" --train_add ./mb/additional.txt --train_set ./mb/trainall.txt --model ./models/model.rf
```
The txt files specify the training sets used during the training process. 


# Testing
- to do ...

## Local Expansion Method
- to do ...
