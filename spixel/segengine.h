#pragma once

#include "stdafx.h"
#include "structures.h"

using namespace cv;

struct SPSegmentationParameters {
    int pixelSize = 16;      // Pixel (block) size -- initial size
    int sPixelSize = 5;      // initial size of superpixels in Pixels (blocks)
    
    double appWeight = 1.0;
    double regWeight = 1.0;
    double lenWeight = 1.0;
    double sizeWeight = 1.0;
    double dispWeight = 1.0;
    double smoWeight = 1.0;
    double priorWeight = 1.0;
    double occPriorWeight = 1.0;
    double hiPriorWeight = 1.0;
    double dispBeta = 1.0;
    double inlierThreshold = 3.0;

    int iterations = 1;

    bool stereo = false;

    void read(const FileNode& node) // Read OpenCV serialization for this class
    {
        pixelSize = (int)node["pixelSize"];
        sPixelSize = (int)node["sPixelSize"];
        appWeight = (double)node["appWeight"];
        regWeight = (double)node["regWeight"];
        lenWeight = (double)node["lenWeight"];
        sizeWeight = (double)node["sizeWeight"];
        dispWeight = (double)node["dispWeight"];
        smoWeight = (double)node["smoWeight"];
        priorWeight = (double)node["priorWeight"];
        occPriorWeight = (double)node["occPriorWeight"];
        hiPriorWeight = (double)node["hiPriorWeight"];
        dispBeta = (double)node["dispBeta"];
        stereo = (int)node["stereo"] != 0;
        iterations = (int)node["iterations"];
        inlierThreshold = (int)node["inlierThreshold"];
    }
};

static void read(const FileNode& node, SPSegmentationParameters& x, const SPSegmentationParameters& defaultValue = SPSegmentationParameters());


class SPSegmentationEngine {
private:
    struct PerformanceInfo {
        double init = 0.0;
        double imgproc = 0.0;
        double ransac = 0.0;
        vector<double> levels;
        double total = 0.0;
    };

    PerformanceInfo performanceInfo;

    // Parameters
    SPSegmentationParameters params;

    // Original image to process
    Mat origImg;

    // Depth image (required for stereo)
    Mat depthImg;
    
    // Image to process (in lab color space)
    Mat img;

    // Support structures
    Matrix<Pixel> pixelsImg;    // pixels matrix, dimension varies, depends on level
    Mat1b inliers;              // boolean matrix of "inliers" (for stereo)

    // Info
    vector<Superpixel*> superpixels;
public:
    SPSegmentationEngine(SPSegmentationParameters params, Mat img, Mat depthImg = Mat());
    virtual ~SPSegmentationEngine();

    void ProcessImage();
    void ProcessImageStereo();
    Mat GetSegmentedImage();
    Mat GetSegmentation() const;
    string GetSegmentedImageInfo();
    void PrintDebugInfo();
    void PrintPerformanceInfo();
    int GetNoOfSuperpixels() const;
    double ProcessingTime() { return performanceInfo.total; }
private:
    void Initialize(Superpixel* spGenerator());
    void InitializeStereo();
    void IterateMoves();
    void ReEstimatePlaneParameters();
    void EstimatePlaneParameters();
    void SplitPixels();
    int PixelSize();
    void Reset();
    void InitializeInliers();
};


SPSegmentationParameters ReadParameters(const string& fileName, const SPSegmentationParameters& defaultValue = SPSegmentationParameters());



