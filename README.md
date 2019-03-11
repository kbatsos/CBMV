# CBMV: A Coalesced Bidirectional Matching Volume for Disparity Estimation
CBMV: A Coalesced Bidirectional Matching Volume for Disparity Estimation code repository. If you use this code please cite our paper [CBMV: A Coalesced Bidirectional Matching Volume for Disparity Estimation](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0598.pdf)

```
@inproceedings{batsos2018cbmv,
  title={CBMV: A Coalesced Bidirectional Matching Volume for Disparity Estimation},
  author={Batsos, Konstantinos and Cai, Changjiang and Mordohai, Philipos},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```
The code includes the cost-optimization and post-processing of MC-CNN [Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches](https://arxiv.org/abs/1510.05970) as implemented by Jure Zbontar, and modified to interface with python. 

# Links

[Training Data](https://drive.google.com/file/d/1RKIhAT5mc9kyWFg9Trg0Ze4qZMzUnbPU/view?usp=sharing)

[CBMV model](https://drive.google.com/file/d/1mjz-Rttdf99BZxne5EFziY5HwB10QPPF/view?usp=sharing )


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


# Testing : Added by CCJ
- to do ...

## Local Expansion Method
- to do ...

# Dataset Format
- 'disp0GT.pfm';
- left image 'im0.png', right image 'im1.png';
- 'calib.txt';
- to do ...



