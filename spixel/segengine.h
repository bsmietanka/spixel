#pragma once

#include "stdafx.h"
#include "structures.h"

using namespace cv;
using namespace std;

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
    int maxUpdates = 400000;
    int reSteps = 5;
    int maxLevels = 10;

    bool instantBoundary = true;
    bool stereo = false;
    bool inpaint = false;           // use opencv's inpaint method to fill gaps in
                                    // disparity image

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
        UpdateFromNode(inpaint, node["inpaint"]);
        UpdateFromNode(instantBoundary, node["instantBoundary"]);
        UpdateFromNode(iterations, node["iterations"]);
        UpdateFromNode(reSteps, node["reSteps"]);
        UpdateFromNode(inlierThreshold, node["inlierThreshold"]);
        UpdateFromNode(maxUpdates, node["maxUpdates"]);
        UpdateFromNode(maxLevels, node["maxLevels"]);
    }
};

static void read(const FileNode& node, SPSegmentationParameters& x, const SPSegmentationParameters& defaultValue = SPSegmentationParameters());

typedef pair<SuperpixelStereo*, SuperpixelStereo*> SPSPair;


inline SPSPair OrderedPair(SuperpixelStereo* sp, SuperpixelStereo* sq)
{
    return sp < sq ? SPSPair(sp, sq) : SPSPair(sq, sp);
}

class BInfoMatrix  {
private:
    BInfo** data;
    int rows;
    int cols;
public:
    BInfoMatrix() : data(nullptr), rows(0), cols(0) { }
    ~BInfoMatrix() { Release(); }

    void Resize(int _rows, int _cols)
    {
        Release();
        rows = _rows;
        cols = _cols;
        data = new BInfo*[rows*cols];
        memset(data, 0, sizeof(BInfo*)*rows*cols);
    }

    BInfo& operator ()(int r, int c) { return *(*data + r*cols + c); }

private:
    void Release()
    {
        if (data != nullptr) {
            BInfo* end = *data + rows*cols, *p = *data;
            for (; p != end; p++) { if (p != nullptr) delete p; }
            delete[] data;
        }
    }
};


class SPSegmentationEngine {
private:
    struct PerformanceInfo {
        double init = 0.0;
        double imgproc = 0.0;
        double ransac = 0.0;
        vector<double> levelTimes;
        vector<int> levelIterations;
        double total = 0.0;
        vector<double> levelMaxEDelta;
    };

    PerformanceInfo performanceInfo;

    // Parameters
    SPSegmentationParameters params;

    // Original image to process
    Mat origImg;

    // Depth image (required for stereo)
    Mat1d depthImg;
    Mat1d depthImgAdj;

    // Image to process (in lab color space)
    Mat img;

    // Support structures
    Matrix<Pixel> pixelsImg;    // pixels matrix, dimension varies, depends on level
    //Mat1b inliers;              // boolean matrix of "inliers" (for stereo)
    Matrix<Pixel*> ppImg;       // matrix of dimension of img, pointers to pixelsImg pixels (for stereo)
    vector<Superpixel*> superpixels;
    //map<SPSPair, BInfo> boundaryData;
public:
    SPSegmentationEngine(SPSegmentationParameters params, Mat img, Mat depthImg = Mat());
    virtual ~SPSegmentationEngine();

    void ProcessImage();
    void ProcessImageStereo();
    Mat GetSegmentedImage();
    Mat GetSegmentedImagePlain();
    Mat GetSegmentedImageStereo();
    Mat GetSegmentation() const;
    Mat GetDisparity() const;
    string GetSegmentedImageInfo();
    void PrintDebugInfo();
    void PrintDebugInfo2();
    void PrintDebugInfoStereo();
    void PrintPerformanceInfo();
    int GetNoOfSuperpixels() const;
    double ProcessingTime() { return performanceInfo.total; }
private:
    void Initialize(Superpixel* spGenerator(int));
    void InitializeStereo();
    void InitializeStereoEnergies();
    void InitializePPImage();
    void UpdatePPImage();
    int IterateMoves(int level);
    void ReEstimatePlaneParameters();
    void EstimatePlaneParameters();
    bool SplitPixels();
    void Reset();
    void UpdateBoundaryData();
    void UpdatePlaneParameters();
    void UpdateStereoSums();
    void UpdateDispSums();

    void DebugNeighborhoods();
    void DebugBoundary();

    bool TryMovePixel(Pixel* p, Pixel* q, PixelMoveData& psd);
    bool TryMovePixelStereo(Pixel* p, Pixel* q, PixelMoveData& psd);
};


SPSegmentationParameters ReadParameters(const string& fileName, const SPSegmentationParameters& defaultValue = SPSegmentationParameters());



