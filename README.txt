===== DESCRIPTION =====

This is an implementation of the algorithms in
  
    Real-Time Coarse-to-fine Topologically Preserving Segmentation
	by Jian Yao, Marko Boben, Sanja Fidler, Raquel Urtasun
	
published in CVPR 2015.

http://www.cs.toronto.edu/~urtasun/publications/yao_etal_cvpr15.pdf

===== LICENSE =====

This software is copyright by Jian Yao, Marko Boben, Sanja Fidler and Raquel Urtasun. It is released for personal or academic use only. Any commercial use is strictly prohibited except by explicit permission by the authors. For more information on commercial use, contact Raquel Urtasun (urtasun@cs.toronto.edu). The authors of this software and corresponding paper assume no liability for its use and by using this software you agree to these terms.  

Any academic use of this software should cite the following work:

@inproceedings{YaoCVPR15,
    title = {Real-Time Coarse-to-fine Topologically Preserving Segmentation},
    author = {Jian Yao and Marko Boben and Sanja Fidler and Raquel Urtasun},
    booktitle = {CVPR},
    year = {2015}
}

If you use SGM for stereo, please also cite:

@inproceedings{HirschmullerCVPR05,
    title = {Accurate and efficient stereo processing by semi-global matching and mutual information},
    author = {H. Hirschmuller},
    booktitle = {CVPR},
    year = {2005}
}

and the following paper (the SGM part of our code adopts this implementation):

@inproceedings{YamaguchiECCV04,
    title = {Efficient Joint Segmentation, Occlusion Labeling, Stereo and Flow Estimation},
    author = {K. Yamaguchi and D. McAllester and R. Urtasun},
    booktitle = {ECCV},
    year = {2014}
}

===== DEPENDENCIES =====

The only dependecies are 
 - OpenCV (http://opencv.org)
 - tinydir.h (https://github.com/cxong/tinydir) to provide platform independent listing
   of files in a directory and is included in src/contrib

===== BUILD INSTRUCTIONS =====

To build this you need:
 - cmake (http://www.cmake.org/download/)
 - A C++11 compatible C++ compiler (tested on GCC 4.8.2 and Visual Studio 2013)
 - OpenCV (tested on 2.4.10, http://opencv.org/downloads.html)

Build Instructions:
 - Linux: run
    cmake .
    make
	and, optionally, (sudo) make install
 - Windows (Visual Studio): use cmake to create sln file:
    In "Where is the source code" specify where you put the files
    (e.g. C:/Work/C/spixel)
    In "Where to build the binaries" say where to put binaries
    (e.g. C:/Work/C/spixel/bin)
    Press "Configure"
	Select the compiler (e.g. Visual Studio 12 2013 Win64)
	Press "Configure"
	In OpenCV_DIR Specify where opencv is installed 
	Press "Configure" and "Generate"
	Use sln file to build the binary (spixel.exe)	
	

===== USAGE =====

spixel executable takes three or four parameters
For "monocular segmentation" run it as

    spixel config.yml directory_with_images file_pattern

where config.yml is a configuration file in yml format where parameters of the algorithm are specified. For an example see examples/basic directory and run run.sh or run.bat file. Results appear in seg and out directories.

For stereo segmentation run it as 

    spixel config.yml directory_with_images file_pattern disparity_extension_pattern
	
For an example see examples/stereo directory.
Results appear in disp, seg, and out directories.

Descriptions of possible parameters are found in the two yml files.

===== CONTACT =====

If you have questions regarding the code, please contact marko.boben@fri.uni-lj.si.