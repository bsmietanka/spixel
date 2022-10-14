#include "stdafx.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ndarray_converter.h"
#include "segengine.h"

namespace py = pybind11;

cv::Mat segment(cv::Mat& image, int num = 1000, double app_weight = 1.0, double reg_weight = 1.0,
                double len_weight = 1.0, double size_weight = 1.0)
{
    SPSegmentationParameters params;
    params.superpixelNum = num;
    params.appWeight = app_weight;
    params.regWeight = reg_weight;
    params.lenWeight = len_weight;
    params.sizeWeight = size_weight;
    SPSegmentationEngine engine(params, image);
    engine.ProcessImage();
    return engine.GetSegmentation();

}


// If dispPattern is empty, dispDir is the whole file name (in case we call this
// function to process only one file)
cv::Mat segment_disparity(cv::Mat& image, cv::Mat& disparity, int num = 1000,
                          double app_weight = 1.0, double reg_weight = 1.0,
                          double len_weight = 1.0, double size_weight = 1.0)
{
    SPSegmentationParameters params;
    params.superpixelNum = num;
    params.appWeight = app_weight;
    params.regWeight = reg_weight;
    params.lenWeight = len_weight;
    params.sizeWeight = size_weight;
    SPSegmentationEngine engine(params, image, disparity);
    engine.ProcessImageStereo();
    return engine.GetSegmentation();
}


PYBIND11_MODULE(spixel, m)
{

    NDArrayConverter::init_numpy();

    m.doc() = "Compute superpixels on images (optionally with disparity information)";
    m.def("segment", &segment,
          "Compute superpixels on numpy image",
          py::arg("image"), py::arg("num") = 1000,
          py::arg("app_weight") = 1.0, py::arg("reg_weight") = 1.0,
          py::arg("len_weight") = 1.0, py::arg("size_weight") = 1.0);
    m.def("segment_disparity", &segment_disparity,
          "Compute superpixels on images with disparity information",
          py::arg("image"), py::arg("disparity"), py::arg("num") = 1000,
          py::arg("app_weight") = 1.0, py::arg("reg_weight") = 1.0,
          py::arg("len_weight") = 1.0, py::arg("size_weight") = 1.0);
}
