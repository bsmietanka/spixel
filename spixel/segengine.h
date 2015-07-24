#pragma once

#include "stdafx.h"
#include "structures.h"
#include "tsdeque.h"

#define ADD_LEVEL_PARAM_DOUBLE(field, node, name) \
    updateDouble[name] = [](SPSegmentationParameters& params, double val) { params.field = val; };\
    AddLevelParamFromNodeDouble(node, name);


using namespace cv;
using namespace std;

void UpdateFromNode(double& val, const FileNode& node);
void UpdateFromNode(int& val, const FileNode& node);
void UpdateFromNode(bool& val, const FileNode& node);

template <typename T> 
void AddLevelParamFromNode(const FileNode& parentNode, const string& nodeName,
    vector<pair<string, vector<T>>>& paramsList)
{
    FileNode n = parentNode[nodeName];

    if (n.empty()) return;

    vector<T> levParams;

    if (n.type() != FileNode::SEQ) {
        levParams.push_back((T)n);
    } else {
        for (FileNode n1 : n) {
            levParams.push_back((T)n1);
        }
    }
    paramsList.push_back(pair<string, vector<T>>(nodeName, levParams));
}


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
    int peblThreshold = 10;         // planeEstimationBundaryLengthThreshold 
    double updateThreshold = 0.01;

    int iterations = 1;
    int maxUpdates = 400000;
    int reSteps = 5;
    int minLevel = 0;

    bool instantBoundary = true;
    bool stereo = false;
    bool inpaint = false;           // use opencv's inpaint method to fill gaps in
                                    // disparity image
    int nThreads = 4;

    vector<pair<string, vector<double>>> levelParamsDouble;
    vector<pair<string, vector<int>>> levelParamsInt;
    map<string, function<void(SPSegmentationParameters&, double)>> updateDouble;
    map<string, function<void(SPSegmentationParameters&, int)>> updateInt;

    void SetLevelParams(int level);

    void Read(const FileNode& node)
    {
        UpdateFromNode(pixelSize, node["pixelSize"]);
        UpdateFromNode(sPixelSize, node["sPixelSize"]);
        ADD_LEVEL_PARAM_DOUBLE(appWeight, node, "appWeight");
        ADD_LEVEL_PARAM_DOUBLE(regWeight, node, "regWeight");
        ADD_LEVEL_PARAM_DOUBLE(lenWeight, node, "lenWeight");
        ADD_LEVEL_PARAM_DOUBLE(sizeWeight, node, "sizeWeight");
        ADD_LEVEL_PARAM_DOUBLE(dispWeight, node, "dispWeight");
        ADD_LEVEL_PARAM_DOUBLE(smoWeight, node, "smoWeight");
        ADD_LEVEL_PARAM_DOUBLE(priorWeight, node, "priorWeight");
        ADD_LEVEL_PARAM_DOUBLE(occPriorWeight, node, "occPriorWeight");
        ADD_LEVEL_PARAM_DOUBLE(hiPriorWeight, node, "hiPriorWeight");
        UpdateFromNode(noDisp, node["noDisp"]);
        UpdateFromNode(stereo, node["stereo"]);
        UpdateFromNode(inpaint, node["inpaint"]);
        UpdateFromNode(instantBoundary, node["instantBoundary"]);
        UpdateFromNode(iterations, node["iterations"]);
        UpdateFromNode(reSteps, node["reSteps"]);
        UpdateFromNode(inlierThreshold, node["inlierThreshold"]);
        UpdateFromNode(maxUpdates, node["maxUpdates"]);
        UpdateFromNode(minLevel, node["minLevel"]);
        UpdateFromNode(nThreads, node["nThreads"]);
        UpdateFromNode(peblThreshold, node["peblThreshold"]);
        UpdateFromNode(updateThreshold, node["updateThreshold"]);
        SetLevelParams(0);
    }


private:
    void AddLevelParamFromNodeDouble(const FileNode& parentNode, const string& nodeName)
    {
        AddLevelParamFromNode<double>(parentNode, nodeName, levelParamsDouble);
    }

    void AddLevelParamFromNodeInt(const FileNode& parentNode, const string& nodeName)
    {
        AddLevelParamFromNode<int>(parentNode, nodeName, levelParamsInt);
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
    void DebugDispSums();

    bool TryMovePixel(Pixel* p, Pixel* q, PixelMoveData& psd);
    bool TryMovePixelStereo(Pixel* p, Pixel* q, PixelMoveData& psd);

    void IterateInThread(ParallelDeque<Pixel*, Superpixel*>* pList);
};


SPSegmentationParameters ReadParameters(const string& fileName, const SPSegmentationParameters& defaultValue = SPSegmentationParameters());



