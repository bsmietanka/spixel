# Real-Time Coarse-to-fine Topologically Preserving Segmentation

This is an implementation of the algorithms in

    Real-Time Coarse-to-fine Topologically Preserving Segmentation
    by Jian Yao, Marko Boben, Sanja Fidler, Raquel Urtasun [link](http://www.cs.toronto.edu/~urtasun/publications/yao_etal_cvpr15.pdf)

This repository provides python bindings for the orginal code.

## Prerequisites

The dependecies are 
 - OpenCV (http://opencv.org)
 - Numpy
 - Pybind11

Additionally to build this project you need:
 - cmake (http://www.cmake.org/download/)
 - A C++11 compatible C++ compiler (tested on GCC 11.2.0)
 - OpenCV (tested on 4.6.0.66, http://opencv.org/downloads.html)

## Install

Clone the repository, go to project root and run `python3 -m pip install .`.

## Usage

```
import spixel
img = ...
disparity = ...
params = [...]
label_field = spixel.segment(img, *params)
label_field = spixel.segment_disparity(img, disparity, *params)
```

### Contact
If you have questions regarding original code, please contact marko.boben@fri.uni-lj.si
