#pragma once

#include "stdafx.h"
#include "structures.h"

using namespace cv;


void UpdateFromNode(double& val, const FileNode& node);
void UpdateFromNode(int& val, const FileNode& node);
void UpdateFromNode(bool& val, const FileNode& node);

struct SPSegmentationParameters {
    int pixelSize = 16;      // Pixel (block) size -- initial size
    int sPixelSize = 5;      // initial size of superpixels in Pixels (blocks)
    
    double appWeight = 1.0;
    double regWeight = 1.0;
    double lenWeight = 1.0;
    double sizeWeight = 1.0;        
    double dispWeight = 2000.0;     // \lambda_{disp}
    double smoWeight = 0.2;         // \lambda_{smo}
    double priorWeight = 0.2;       // \lambda_{prior}
    double occPriorWeight = 15.0;   // \lambda_{occ}
    double hiPriorWeight = 5.0;     // \lambda_{hinge}
    double noDisp = 9.0;            // \lambda_{d}
    double inlierThreshold = 3.0;

    int iterations = 1;

    bool stereo = false;

    void read(const FileNode& node)
    {
        UpdateFromNode(pixelSize, node["pixelSize"]);
        UpdateFromNode(sPixelSize, node["sPixelSize"]);
        UpdateFromNode(appWeight, node["appWeight"]);
        UpdateFromNode(regWeight, node["regWeight"]);
        UpdateFromNode(lenWeight, node["lenWeight"]);
        UpdateFromNode(sizeWeight, node["sizeWeight"]);
        UpdateFromNode(dispWeight, node["dispWeight"]);
        UpdateFromNode(smoWeight, node["smoWeight"]);
        UpdateFromNode(priorWeight, node["priorWeight"]);
        UpdateFromNode(occPriorWeight, node["occPriorWeight"]);
        UpdateFromNode(hiPriorWeight, node["hiPriorWeight"]);
        UpdateFromNode(noDisp, node["noDisp"]);
        UpdateFromNode(stereo, node["stereo"]);
        UpdateFromNode(iterations, node["iterations"]);
        UpdateFromNode(inlierThreshold, node["inlierThreshold"]);
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
    Mat1d depthImg;
    
    // Image to process (in lab color space)
    Mat img;

    // Support structures
    Matrix<Pixel> pixelsImg;    // pixels matrix, dimension varies, depends on level
    Mat1b inliers;              // boolean matrix of "inliers" (for stereo)
    Matrix<Pixel*> ppImg;       // matrix of dimension of img, pointers to pixelsImg pixels (for stereo)
    vector<Superpixel*> superpixels;
public:
    SPSegmentationEngine(SPSegmentationParameters params, Mat img, Mat depthImg = Mat());
    virtual ~SPSegmentationEngine();

    void ProcessImage();
    void ProcessImageStereo();
    Mat GetSegmentedImage();
    Mat GetSegmentation() const;
    Mat GetDisparity() const;
    string GetSegmentedImageInfo();
    void PrintDebugInfo();
    void PrintDebugInfoStereo();
    void PrintPerformanceInfo();
    int GetNoOfSuperpixels() const;
    double ProcessingTime() { return performanceInfo.total; }
private:
    void Initialize(Superpixel* spGenerator());
    void InitializeStereo();
    void InitializeStereoEnergies();
    void InitializePPImage();
    void UpdatePPImage();
    void IterateMoves();
    void ReEstimatePlaneParameters();
    void EstimatePlaneParameters();
    void SplitPixels();
    int PixelSize();
    void Reset();
    void UpdateInliers();

    bool TryMovePixel(Pixel* p, Pixel* q, PixelMoveData& psd);
    bool TryMovePixelStereo(Pixel* p, Pixel* q, PixelMoveData& psd);
};


SPSegmentationParameters ReadParameters(const string& fileName, const SPSegmentationParameters& defaultValue = SPSegmentationParameters());



