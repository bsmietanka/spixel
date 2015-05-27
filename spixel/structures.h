#pragma once

#include "stdafx.h"
#include <vector>
#include <memory>
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include "sallocator.h"

using namespace std;

// Typedefs
/////////////

typedef std::int16_t sint;
typedef std::uint8_t byte;
typedef cv::Point3d Plane_d;

const double eps = 1.0E-6;


// Constants
//////////////

// Border (boundary) types (hinge, coplanarity, etc.)
const int BTCo = 1;
const int BTHi = 2;
const int BTLo = 3;
const int BTRo = 4;


// Border position flags
const byte BLeftFlag = 1;
const byte BRightFlag = 2;
const byte BTopFlag = 4;
const byte BBottomFlag = 8;


// Simple structs and classes
///////////////////////////////

struct BInfo {
    int type = 0;               // Boundary type BTCo, BTHi, ...
    double typePrior = 0;       // Prior weight (dep. of type)
    int length = 0;             // Length of boundary (between two superpixels)
    double coSum = 0.0;         // Smo value (sum) for coplanarity
    double hiSum = 0.0;         // Smo value (sum) for hinge
};


// Functions
//////////////

inline double CalcRegEnergy(double sumRow, double sumCol, double sumRow2, double sumCol2, double size, int numP)
{
    double meanRow = (double)sumRow / size;
    double meanCol = (double)sumCol / size;
    double difSqRow = sumRow2 - 2 * meanRow * sumRow + size * meanRow * meanRow;
    double difSqCol = sumCol2 - 2 * meanCol * sumCol + size * meanCol * meanCol;

    return (difSqCol + difSqRow) / (size / numP);
}

inline double CalcAppEnergy(double sumR, double sumG, double sumB,
    double sumR2, double sumG2, double sumB2, int size, int numP)
{
    double meanR = sumR / size;
    double meanG = sumG / size;
    double meanB = sumB / size;
    double difSqR = sumR2 - 2 * meanR * sumR + size * meanR * meanR;
    double difSqG = sumG2 - 2 * meanG * sumG + size * meanG * meanG;
    double difSqB = sumB2 - 2 * meanB * sumB + size * meanB * meanB;

    return (difSqR + difSqG + difSqB) / (size / numP);
}

inline double DotProduct(const Plane_d& p, double x, double y, double z)
{
    return x*p.x + y*p.y + z*p.z;
}

template<typename T> T Sqr(const T& a)
{
    return a*a;
}

inline double SafeDiv(double a, double b, double aa)
{
    return b == 0 ? a / b : aa;
}

// Matrix -- very simple matrix
/////////////////////////////////

//template <typename T> struct Matrix;

//template <typename T> struct MatrixIter {
//private:
//    Matrix* pMatrix;
//    int pos;
//
//public:
//    MatrixIter(Matrix* pMatrix_, int pos_) : pMatrix(pMatrix_), pos(pos_) { }
//    void operator++() { ++pos; }
//    bool operator!=(const MatrixIter& iter2) const { return pos != pos}
//};

template <typename T> struct Matrix {
    shared_ptr<T> data;
    int rows;
    int cols;

    Matrix() : data(nullptr), rows(0), cols(0) { }

    Matrix(int rows_, int cols_) : data(new T[rows_*cols_], [](T* p) { delete[] p; }), rows(rows_), cols(cols_) { }

    inline T& operator() (int r, int c)
    {
        return *(data.get() + r*cols + c);
    }

    inline const T& operator() (int r, int c) const
    {
        return *(data.get() + r*cols + c);
    }

    inline T& operator() (int i) const
    {
        return *(data.get() + i);
    }

    inline T* PtrTo(int r, int c)
    {
        return data.get() + r*cols + c;
    }

    inline const T* PtrTo(int r, int c) const
    {
        return data.get() + r*cols + c;
    }

    const T* begin() const { return data.get(); }
    const T* end() const { return data.get() + rows*cols; }
    T* begin() { return data.get(); }
    T* end() { return data.get() + rows*cols; }

};


// Pixel/Superpixel
/////////////////////

struct Pixel;
class Superpixel;
class SuperpixelStereo;


//typedef unordered_map<Superpixel*, BInfo> BInfoMapType;

struct PixelData {
    Pixel* p;
    double sumRow, sumCol;
    double sumRow2, sumCol2;
    double sumR, sumG, sumB;
    double sumR2, sumG2, sumB2;
    double sumDispP, sumDispQ;
    int size;
};


typedef unordered_map<SuperpixelStereo*, BInfo> BorderDataMap;

// Note: energy deltas below are "energy_before - energy_after"
struct PixelMoveData {
    bool allowed;       // Operation is allowed, if false, other fields are not valid
    Pixel* p;           // Moved pixel
    Pixel* q;           // Neighbor of p
    double eRegDelta;   // "Regularization" energy delta
    double eAppDelta;   // "Appearance" energy delta
    double eDispDelta;  // Deltas of energy terms in Eq. (7)
    double eSmoDelta;
    double ePriorDelta;
    double bLenDelta;   // boundary length delta
    int pSize;          // superpixel of p size
    int qSize;          // superpixel of q size
    PixelData pixelData;
    BorderDataMap bDataP, bDataQ;
};

struct PixelChangeData {
    double newEReg;  // new "Regularization" energy
    double newEApp;  // new "Appearance" energy
    int newSize;     // new superpixel size
};

struct PixelChangeDataStereo : public PixelChangeData {
    double newEDisp;
};



struct Pixel { // : public custom_alloc {

    Superpixel* superPixel;

    // Geometry info
    sint row, col;                 // pixelsImg position
    sint ulr, ulc, lrr, lrc;       // coordinates of upper left and lower right corners (in img dimensions) 

    // Border info (see flags B*Flag above)
    byte border;

    Pixel() { }

    inline void Initialize(sint row_, sint col_, sint ulr_, sint ulc_, sint lrr_, sint lrc_)
    {
        CV_DbgAssert(ulr_ < lrr_ && ulc_ < lrc_);
        row = row_; col = col_;
        ulr = ulr_; ulc = ulc_; lrr = lrr_; lrc = lrc_;
        border = 0;
        superPixel = nullptr;
    }

    void CalcRowColSum(double& sumRow, double& sumRow2, double& sumCol, double& sumCol2) const
    {
        sumRow = 0.0; sumRow2 = 0.0;
        sumCol = 0.0; sumCol2 = 0.0;
        for (int i = ulr; i < (int)lrr; i++) {
            for (int j = ulc; j < (int)lrc; j++) {
                sumRow += i; sumCol += j;
                sumRow2 += i*i; sumCol2 += j*j;
                if (sumCol2 < 0) {
                    exit(0);
                }
            }
        }
    }

    void CalcRGBSum(const cv::Mat& img, double& sumR, double& sumR2, double& sumG, double& sumG2,
        double& sumB, double& sumB2) const
    {
        sumR = 0; sumR2 = 0;
        sumG = 0; sumG2 = 0;
        sumB = 0; sumB2 = 0;
        for (int i = ulr; i < (int)lrr; i++) {
            for (int j = ulc; j < (int)lrc; j++) {
                const cv::Vec3d val = img.at<cv::Vec3d>(i, j);
                double b = val[0], g = val[1], r = val[2];
                sumR += r; sumR2 += r*r;
                sumG += g; sumG2 += g*g;
                sumB += b; sumB2 += b*b;
            }
        }
        /* sumR /= 255.0;
        sumG /= 255.0;
        sumB /= 255.0;
        sumR2 /= 65025.0; // 65025 = 255*255
        sumG2 /= 65025.0;
        sumB2 /= 65025.0;
        */
    }

    double CalcDispSum(const cv::Mat1d& dispImg, const cv::Mat1b& inliers, const Plane_d& plane, double noDisp) const
    {
        double sumDisp = 0.0;

        for (int i = ulr; i < (int)lrr; i++) {
            for (int j = ulc; j < (int)lrc; j++) {
                if (inliers(i, j) > 0) {
                    const double& disp = dispImg(i, j);
                    double delta = DotProduct(plane, i, j, 1.0) - disp;
                    sumDisp += delta*delta;
                } else {
                    sumDisp += noDisp;
                }
            }
        }
        return sumDisp;
    }

    double CalcDispSum(const cv::Mat1d& dispImg, const Plane_d& plane, double inlierThresh, double noDisp) const
    {
        double sumDisp = 0.0;

        for (int i = ulr; i < (int)lrr; i++) {
            for (int j = ulc; j < (int)lrc; j++) {
                const double& disp = dispImg(i, j);
                if (disp > 0) {
                    double delta = DotProduct(plane, i, j, 1.0) - disp;
                    if (fabs(delta) < inlierThresh) sumDisp += delta*delta;
                    else sumDisp += noDisp;
                } else {
                    sumDisp += noDisp;
                }
            }
        }
        return sumDisp;
    }

    void CalcHiSmoothnessSum(byte sideFlag, const cv::Mat1b& inliers, const Plane_d& planep, const Plane_d& planeq,
        double& sum, int& count) const
    {
        sum = 0.0;
        count = 0;

        if (sideFlag == BLeftFlag) {
            for (int i = ulr; i < (int)lrr; i++) {
                if (inliers(i, ulc) > 0) {
                    double ddelta = DotProduct(planep, i, ulc, 1.0) - DotProduct(planeq, i, ulc, 1.0);
                    sum += ddelta*ddelta;
                    count++;
                }
            }
        } else if (sideFlag == BRightFlag) {
            for (int i = ulr; i < (int)lrr; i++) {
                if (inliers(i, lrc - 1) > 0) {
                    double ddelta = DotProduct(planep, i, lrc - 1, 1.0) - DotProduct(planeq, i, lrc - 1, 1.0);
                    sum += ddelta*ddelta;
                    count++;
                }
            }
        } else if (sideFlag == BTopFlag) {
            for (int j = ulc; j < (int)lrc; j++) {
                if (inliers(ulr, j) > 0) {
                    double ddelta = DotProduct(planep, ulr, j, 1.0) - DotProduct(planeq, ulr, j, 1.0);
                    sum += ddelta*ddelta;
                    count++;
                }
            }
        } else if (sideFlag == BTopFlag) {
            for (int j = ulc; j < (int)lrc; j++) {
                if (inliers(lrr - 1, j) > 0) {
                    double ddelta = DotProduct(planep, lrr - 1, j, 1.0) - DotProduct(planeq, lrr - 1, j, 1.0);
                    sum += ddelta*ddelta;
                    count++;
                }
            }
        }
    }

    double CalcCoSmoothnessSum(const cv::Mat1b& inliers, const Plane_d& planep, const Plane_d& planeq) const
    {
        double sum = 0.0;

        for (int i = ulr; i < (int)lrr; i++) {
            for (int j = ulc; j < (int)lrc; j++) {
                if (inliers(i, j) > 0) {
                    double ddelta = DotProduct(planep, i, j, 1.0) - DotProduct(planeq, i, j, 1.0);
                    sum += ddelta*ddelta;
                }
            }
        } 
        return sum;
    }

    void AddDispPixels(const cv::Mat1d& dispImg, vector<cv::Point3d>& pixels)
    {
        for (int i = ulr; i < (int)lrr; i++) {
            for (int j = ulc; j < (int)lrc; j++) {
                const double& disp = dispImg(i, j);
                if (disp > 0.0)
                    pixels.push_back(cv::Point3d(i, j, disp));
            }
        }
    }
    
    inline void CalcPixelData(const cv::Mat& img, PixelData& pd)
    {
        pd.p = this;
        CalcRowColSum(pd.sumRow, pd.sumRow2, pd.sumCol, pd.sumCol2);
        CalcRGBSum(img, pd.sumR, pd.sumR2, pd.sumG, pd.sumG2, pd.sumB, pd.sumB2);
        pd.size = GetSize();
    }

    inline void CalcPixelDataStereo(const cv::Mat& img, const cv::Mat1d& imgDisp,
        Plane_d& planeP, Plane_d& planeQ, double inlierThresh, double noDisp, PixelData& pd)
    {
        CalcPixelData(img, pd);
        pd.sumDispP = CalcDispSum(imgDisp, planeP, inlierThresh, noDisp);
        pd.sumDispQ = CalcDispSum(imgDisp, planeQ, inlierThresh, noDisp);
    }

    //void AddToInlierSums(const cv::Mat1d& dispImg, const cv::Mat1b& inliers,
    //    int& sumIRow, int& sumIRow2, int& sumICol, int& sumICol2, int& sumIRowCol, double& sumIRowD, double& sumIColD, double& sumID, int& nI)
    //{
    //    for (int i = ulr; i < (int)lrr; i++) {
    //        for (int j = ulc; j < (int)lrc; j++) {
    //            if (inliers(i, j) > 0) {
    //                const double& disp = dispImg(i, j);
    //                sumIRow += i; sumIRow2 += i*i;
    //                sumICol += j; sumICol2 += j*j;
    //                sumIRowCol += i*j;
    //                sumIRowD += i*disp; sumIColD += j*disp;
    //                sumID += disp;
    //                nI++;
    //            }
    //        }
    //    }
    //}

    void UpdatePPImage(Matrix<Pixel*>& ppImg)
    {
        for (int i = ulr; i < (int)lrr; i++) {
            for (int j = ulc; j < (int)lrc; j++) {
                ppImg(i, j) = this;
            }
        }
    }



    // Border operations
    bool BLeft() const { return (border & BLeftFlag) != 0; }
    void SwapBLeft() { border ^= BLeftFlag; }
    void SetBLeft() { border |= BLeftFlag; }
    bool BRight() const { return (border & BRightFlag) != 0; }
    void SwapBRight() { border ^= BRightFlag; }
    void SetBRight() { border |= BRightFlag; }
    bool BTop() const { return (border & BTopFlag) != 0; }
    void SwapBTop() { border ^= BTopFlag; }
    void SetBTop() { border |= BTopFlag; }
    bool BBottom() const { return (border & BBottomFlag) != 0; }
    void SwapBBottom() { border ^= BBottomFlag; }
    void SetBBottom() { border |= BBottomFlag; }
    void CopyBFlag(byte srcBorder, byte flag) { border = (border & (~flag)) | (srcBorder & flag); }

    int GetSize() const { return (lrc - ulc)*(lrr - ulr); }
    int GetCSize() const { return lrc - ulc; }
    int GetRSize() const { return lrr - ulr; }
    int GetBoundaryLength() const { return 2 * (lrc - ulc) + 2 * (lrr - ulr); }
    int CountBorderSides() const { return (int)BLeft() + (int)BRight() + (int)BTop() + (int)BBottom(); }
    void Split(const cv::Mat& img, int row1, int row2, int col1, int col2, Pixel& p11, Pixel& p12, Pixel& p21, Pixel& p22);
    void SplitRow(const cv::Mat& img, int row1, int row2, int col, Pixel& p11, Pixel& p21);
    void SplitColumn(const cv::Mat& img, int row, int col1, int col2, Pixel& p11, Pixel& p12);
    void CopyTo(const cv::Mat& img, int row, int col, Pixel& p11);
    void UpdateInliers(const cv::Mat1d& dispImg, double threshold, cv::Mat1b& inliers) const;
    //void AddToInlierSums(const cv::Mat1d& depthImg, const cv::Mat1b& inliers,
    //    int& sumIRow, int& sumIRow2, int& sumICol, int& sumICol2, int& sumIRowCol, double& sumIRowD, double& sumIColD, double& sumID, int& nI); 
    string GetPixelsAsString();

    //CLASS_ALLOCATION()
};

class Superpixel {
public:
    int borderLength = 0;                   // length of border (in img pixels)
    double sumRow = 0, sumCol = 0;          // sum of row, column indices
    double sumRow2 = 0, sumCol2 = 0;        // sum or row, column squares of indices
    double sumR = 0, sumG = 0, sumB = 0;    // sum of colors
    double sumR2 = 0, sumG2 = 0, sumB2 = 0; // sum of squares of colors
    double eApp = 0.0;
    double eReg = 0.0;
    int size = 0;                           // number of (image) pixels
    int initialSize = 0;
    int numP = 0;

public:
    Superpixel() { }
    virtual ~Superpixel() { }

    // Same as AddPixel below, but energy is not caclulated and must be initialized separately.
    // Should only be used in initialization process.
    virtual void AddPixelInit(PixelData& pd) 
    {
        sumRow += pd.sumRow;
        sumCol += pd.sumCol;
        sumRow2 += pd.sumRow2;
        sumCol2 += pd.sumCol2;
        sumR += pd.sumR;
        sumG += pd.sumG;
        sumB += pd.sumB;
        sumR2 += pd.sumR2;
        sumG2 += pd.sumG2;
        sumB2 += pd.sumB2;
        size += pd.size;
        numP++;
        pd.p->superPixel = this;
    }

    // Add a pixel (block) to superpixel.
    // Sums of colors (for average color) and coordinates (for center of mass) are updated.
    // Note that borderLength is *not* updated and should be set separately!
    virtual void AddPixel(PixelData& pd)
    {
        AddPixelInit(pd);
        eApp = CalcAppEnergy(sumR, sumG, sumB, sumR2, sumG2, sumB2, size, numP);
        eReg = CalcRegEnergy(sumRow, sumCol, sumRow2, sumCol2, size, numP);
        //double eReg2 = regEnergyDebug();
        //double dif = eReg - eReg2;
    }

    /*
    double regEnergyDebug() 
    {
        int sumRow = 0, sumCol = 0;
        int size = 0;

        for (Pixel* pp : pixels) {
            size += pp->GetSize();
            for (int r = pp->ulr; r < pp->lrr; r++) {
                for (int c = pp->ulc; c < pp->lrc; c++) {
                    sumRow += r;
                    sumCol += c;
                }
            }
        }

        double avgRow = (double)sumRow / size;
        double avgCol = (double)sumCol / size;
        double result = 0.0;

        for (Pixel* pp : pixels) {
            for (int r = pp->ulr; r < pp->lrr; r++) {
                for (int c = pp->ulc; c < pp->lrc; c++) {
                    double a = r - avgRow;
                    double b = c - avgCol;
                    result += a*a + b*b;
                }
            }
        }
        return result;
    }
    */

    // Remove a pixel (block) from superpixel.
    // See comment for AddPixel.
    virtual void RemovePixel(const PixelData& pd)
    {
        sumRow -= pd.sumRow;
        sumCol -= pd.sumCol;
        sumRow2 -= pd.sumRow2;
        sumCol2 -= pd.sumCol2;
        sumR -= pd.sumR;
        sumG -= pd.sumG;
        sumB -= pd.sumB;
        sumR2 -= pd.sumR2;
        sumG2 -= pd.sumG2;
        sumB2 -= pd.sumB2;
        size -= pd.size;
        numP--;
        eApp = CalcAppEnergy(sumR, sumG, sumB, sumR2, sumG2, sumB2, size, numP);
        eReg = CalcRegEnergy(sumRow, sumCol, sumRow2, sumCol2, size, numP);
    }

    virtual void FinishInitialization()
    {
        initialSize = size;
        RecalculateEnergies();
    }

    virtual void RecalculateEnergies()
    {
        eApp = CalcAppEnergy(sumR, sumG, sumB, sumR2, sumG2, sumB2, size, numP);
        eReg = CalcRegEnergy(sumRow, sumCol, sumRow2, sumCol2, size, numP);
    }

    void SetBorderLength(int bl) { borderLength = bl; }

    void AddToBorderLength(int bld) { borderLength += bld; }

    int GetBorderLength() { return borderLength; }

    int GetSize() const { return size; }
    int GetInitialSize() const { return initialSize; }

    void GetMean(double& meanR, double& meanC) const
    {
        meanR = (double)sumRow / size;
        meanC = (double)sumCol / size;
    }

    void GetMeanColors(double& r, double& g, double& b) const
    {
        r = sumR / size;
        g = sumG / size;
        b = sumB / size;
    }

    double GetRegEnergy() const
    {
        return eReg;
    }

    double GetAppEnergy() const
    {
        return eApp;
    }

    void GetRemovePixelData(const PixelData& pd, PixelChangeData& pcd) const;
    void GetAddPixelData(const PixelData& pd, PixelChangeData& pcd) const;

};


class SuperpixelStereo : public Superpixel {
public:
    double sumDisp = 0.0;
    int sumIRow = 0, sumICol = 0;         // Sum of terms computed for inlier points
    int sumIRow2 = 0, sumICol2 = 0;
    int sumIRowCol = 0;
    double sumIRowD = 0.0, sumIColD = 0.0;
    double sumID = 0.0;
    int nI = 0;

    //double sumIRow = 0.0;


    unordered_set<Pixel*> pixels;
    BorderDataMap boundaryData;
    Plane_d plane;

    SuperpixelStereo() : Superpixel() { }
    virtual ~SuperpixelStereo() { }

    void AddPixelInit(PixelData& pd) override
    {
        Superpixel::AddPixelInit(pd);
        pixels.insert(pd.p);
    }

    void AddPixel(PixelData& pd) override
    {
        Superpixel::AddPixel(pd);
        pixels.insert(pd.p);
        sumDisp += pd.sumDispQ;
    }

    void RemovePixel(const PixelData& pd) override
    {
        Superpixel::RemovePixel(pd);
        pixels.erase(pd.p);
        sumDisp -= pd.sumDispP;
    }

    double GetDispSum()
    {
        return sumDisp;
    }

    double GetPriorEnergy() const
    {
        double result = 0.0;

        for (auto pair : boundaryData) {
            result += pair.second.typePrior;
        }
        return result;
    }

    double GetSmoEnergy()
    {
        double result = 0.0;

        for (auto& bdItem : boundaryData) {
            const BInfo& bInfo = bdItem.second;
            if (bInfo.length > 0) {
                if (bInfo.type == BTCo) result += bInfo.coSum / (size + bdItem.first->size);
                else if (bInfo.type == BTHi) result += bInfo.coSum / bInfo.length;
            }
        }
        return result;
    }

    void SetPlane(Plane_d& plane);
    void UpdateDispSum(const cv::Mat1d& depthImg, const cv::Mat1b& inliers, double noDisp);
    //void UpdateInlierSums(const cv::Mat1d& depthImg, const cv::Mat1b& inliers);
    //void UpdateInliers(const cv::Mat1d& depthImg, double threshold, cv::Mat1b& inliers);
    void CalcPlaneLeastSquares(const cv::Mat1d& depthImg);
    void CalcPlaneLeastSquares(SuperpixelStereo* sq, const cv::Mat1d& depthImg);

    void ClearPixelSet() { pixels.clear(); }
    void AddToPixelSet(Pixel* p1) { pixels.insert(p1); }
    void AddToPixelSet(Pixel* p1, Pixel* p2) { pixels.insert(p1); pixels.insert(p2); }
    void AddToPixelSet(Pixel* p1, Pixel* p2, Pixel* p3, Pixel* p4) 
    { 
        pixels.insert(p1); pixels.insert(p2); 
        pixels.insert(p3); pixels.insert(p4);
    }

    void GetRemovePixelDataStereo(const PixelData& pd, PixelChangeDataStereo& pcd) const;
    void GetAddPixelDataStereo(const PixelData& pd, PixelChangeDataStereo& pcd) const;

    // For debug purposes!
    double CalcDispEnergy(const cv::Mat1d& dispImg, double inlierThresh, double noDisp);
    void CheckAppEnergy(const cv::Mat& img);
    void CheckRegEnergy();
};

// CircList 
// uses "circular" representation based on fixed size vector
//////////////////////////////////////////////////////////////

template<class T> class CircList {
private:
    std::vector<T> vector;
    size_t start;
    size_t end;
    size_t listSize;

public:
    CircList(size_t maxSize) : start(0), end(0), listSize(0) 
    {
        vector.resize(maxSize);
    }

    size_t Size() const { return listSize; }

    bool Empty() const { return listSize == 0; }
    
    void Clear() { start = end = listSize = 0; }

    void PushBack(const T& value) 
    {
        CV_DbgAssert(listSize + 1 != vector.size());
        vector[end] = value;
        end = (end + 1) % vector.size();
        listSize++;
    }

    T& Front() { return vector[start]; }

    const T& Front() const { return vector[start]; }
    
    void PopFront() { start = (start + 1) % vector.size(); listSize--; }
};

// Simple Union-Find disjont set of integers (i.e., equivalence classes)
// implementation using a vector
// TODO optimize a bit Find if necessary.
////////////////////////////////////////////////////////////////////////////

class UnionFind {
protected:
    std::vector<int> vector;
public:
    UnionFind(int size)
    {
        vector.resize(size, -1);    // -1 is size of each 
    }
 
    void Union(int e1, int e2)
    {
        e1 = Find(e1);
        e2 = Find(e2);

        if (e1 == e2)
            return;

        int size1 = -vector[e1];
        int size2 = -vector[e2];

        if (size1 < size2) { vector[e1] = e2; vector[e2] = -(size1 + size2); }
        else { vector[e2] = e1; vector[e1] = -(size1 + size2); }
    }

    inline int Find(int e)
    {
        int root = e;
        
        while (vector[root] >= 0) root = vector[root];
        if (e != root) {
            while (vector[e] != root) {
                int tmpe = vector[e];
                vector[e] = root;
                e = tmpe;
            }
        }
        return root;
    }

    inline int Size(int e) 
    {
        return -vector[Find(e)];
    }
    
};

// Functions
//////////////

// Return Pixel at (row, col), nullptr if coordinates are invalid
inline const Pixel* PixelAt(const Matrix<Pixel>& pixelsImg, int row, int col)
{
    if (row >= 0 && row < pixelsImg.rows && col >= 0 && col < pixelsImg.cols) return pixelsImg.PtrTo(row, col);
    else return nullptr;
}

inline Pixel* PixelAt(Matrix<Pixel>& pixelsImg, int row, int col)
{
    if (row >= 0 && row < pixelsImg.rows && col >= 0 && col < pixelsImg.cols) return pixelsImg.PtrTo(row, col);
    else return nullptr;
}

//inline double GetPriorEnergy(const unordered_map<Superpixel*, BInfo>& boundaryData)
//{
//    double result = 0.0;
//
//    for (auto pair : boundaryData) {
//        result += pair.second.prior;
//    }
//    return result;
//}

