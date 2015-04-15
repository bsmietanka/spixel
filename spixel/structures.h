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


// Functions
//////////////

inline double CalcRegEnergy(int sumRow, int sumCol, int sumRow2, int sumCol2, int size)
{
    double meanRow = (double)sumRow / size;
    double meanCol = (double)sumCol / size;
    double difSqRow = sumRow2 - 2 * meanRow * sumRow + size * meanRow * meanRow;
    double difSqCol = sumCol2 - 2 * meanCol * sumCol + size * meanCol * meanCol;

    return difSqCol + difSqRow;
}

inline double CalcAppEnergy(double sumR, double sumG, double sumB,
    double sumR2, double sumG2, double sumB2, int size)
{
    double meanR = sumR / size;
    double meanG = sumG / size;
    double meanB = sumB / size;
    double difSqR = sumR2 - 2 * meanR * sumR + size * meanR * meanR;
    double difSqG = sumG2 - 2 * meanG * sumG + size * meanG * meanG;
    double difSqB = sumB2 - 2 * meanB * sumB + size * meanB * meanB;

    return difSqR + difSqG + difSqB;
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


struct BInfo {
    int type = 0;
    int length = 0;
    double prior = 0.0;
    double smo = 0.0;
};

typedef unordered_map<Superpixel*, BInfo> BInfoMapType;

struct PixelData {
    Pixel* p;
    int sumRow, sumCol;
    int sumRow2, sumCol2;
    double sumR, sumG, sumB;
    double sumR2, sumG2, sumB2;
    double sumDispP, sumDispQ;
    int size;
};

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
    int bLenDelta;      // boundary length delta
    int pSize;          // superpixel of p size
    int qSize;          // superpixel of q size
    PixelData pixelData;
    BInfoMapType bDataP, bDataQ;
};

struct PixelChangeData {
    double newEReg;  // new "Regularization" energy
    double newEApp;  // new "Appearance" energy
    int newSize;     // new superpixel size
};

struct PixelChangeDataStereo : public PixelChangeData {
    double newEDisp;
    BInfoMapType newBoundaryData;
};


// Border position flags
const byte BLeftFlag = 1;
const byte BRightFlag = 2;
const byte BTopFlag = 4;
const byte BBottomFlag = 8;

// Border types (hinge, coplanarity, etc.)
const int BTCo = 1;
const int BTHi = 2;
const int BTLo = 3;
const int BTRo = 4;


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

    void CalcRowColSum(int& sumRow, int& sumRow2, int& sumCol, int& sumCol2) const
    {
        sumRow = 0; sumRow2 = 0;
        sumCol = 0; sumCol2 = 0;
        for (int i = ulr; i < (int)lrr; i++) {
            for (int j = ulc; j < (int)lrc; j++) {
                sumRow += i; sumCol += j;
                sumRow2 += i*i; sumCol2 += j*j;
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

    double CalcDispSum(const cv::Mat1w& dispImg, const Plane_d& plane, double beta) const
    {
        double sumDisp = 0.0;

        for (int i = ulr; i < (int)lrr; i++) {
            for (int j = ulc; j < (int)lrc; j++) {
                const ushort& disp = dispImg(i, j);

                if (disp > 0) {
                    double delta = DotProduct(plane, i, j, 1.0) - disp;
                    delta *= delta;
                    sumDisp += beta > delta ? beta : delta;
                }
            }
        }
        return sumDisp;
    }

    void CalcHiSmoothnessSum(byte sideFlag, const Plane_d& planep, const Plane_d& planeq,
        double& sum, int& count) const
    {
        sum = 0.0;
        count = 0;

        if (sideFlag == BLeftFlag) {
            for (int i = ulr; i < (int)lrr; i++) {
                double ddelta = DotProduct(planep, i, ulc, 1.0) - DotProduct(planeq, i, ulc, 1.0);
                sum += ddelta*ddelta;
                count++;    // or calculate
            }
        } else if (sideFlag == BRightFlag) {
            for (int i = ulr; i < (int)lrr; i++) {
                double ddelta = DotProduct(planep, i, lrc - 1, 1.0) - DotProduct(planeq, i, lrc - 1, 1.0);
                sum += ddelta*ddelta;
                count++;    // or calculate
            }
        } else if (sideFlag == BTopFlag) {
            for (int j = ulc; j < (int)lrc; j++) {
                double ddelta = DotProduct(planep, ulr, j, 1.0) - DotProduct(planeq, ulr, j, 1.0);
                sum += ddelta*ddelta;
                count++;    // or calculate
            }
        } else if (sideFlag == BTopFlag) {
            for (int j = ulc; j < (int)lrc; j++) {
                double ddelta = DotProduct(planep, lrr - 1, j, 1.0) - DotProduct(planeq, lrr - 1, j, 1.0);
                sum += ddelta*ddelta;
                count++;    // or calculate
            }
        }
    }

    double CalcCoSmoothnessSum(const Plane_d& planep, const Plane_d& planeq) const
    {
        double sum = 0.0;

        for (int i = ulr; i < (int)lrr; i++) {
            for (int j = ulc; j < (int)lrc; j++) {
                double ddelta = DotProduct(planep, i, j, 1.0) - DotProduct(planeq, i, j, 1.0);
                sum += ddelta*ddelta;
            }
        } 
        return sum;
    }

    // sideFlag in {BLeft, BRight, BTop, BBottom}, type in {BTHi, BTCo, BTLo, BTRo}
    //void CalcBorderTypeSum(const Plane_d& planep, const Plane_d& planeq, 
    //    int sideFlag, int type, double occWeight, double hingeWeight,
    //    double& sumPrior, double& sumSmo, int& count) const
    //{
    //    sumPrior = 0.0;
    //    sumSmo = 0.0;
    //    count = 0;

    //    if (type == BTLo || type == BTRo) {
    //        sumPrior += occWeight;
    //        sumSmo = 0;
    //    } else if (type == BTHi) {
    //        sumPrior += hingeWeight;
    //        AddToHiSmoothnessSum(sideFlag, planep, planeq, sumSmo, count);
    //    } else if (type == BTCo) {
    //        AddToCoSmoothnessSum(sideFlag, planep, planeq, sumSmo, count);
    //    }
    //}

    void AddDispPixels(const cv::Mat1w& dispImg, vector<cv::Point3d>& pixels)
    {
        for (int i = ulr; i < (int)lrr; i++) {
            for (int j = ulc; j < (int)lrc; j++) {
                pixels.push_back(cv::Point3d(i, j, dispImg(i, j)));
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

    inline void CalcPixelDataStereo(const cv::Mat& img, const cv::Mat1w& imgDisp, 
        Plane_d& planeP, Plane_d& planeQ, double beta, PixelData& pd)
    {
        CalcPixelData(img, pd);
        pd.sumDispP = CalcDispSum(imgDisp, planeP, beta);
        pd.sumDispQ = CalcDispSum(imgDisp, planeQ, beta);
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
    string GetPixelsAsString();

    //CLASS_ALLOCATION()
};

class Superpixel {
protected:
    int borderLength = 0;                   // length of border (in img pixels)
    int sumRow = 0, sumCol = 0;             // sum of row, column indices
    int sumRow2 = 0, sumCol2 = 0;           // sum or row, column squares of indices
    double sumR = 0, sumG = 0, sumB = 0;    // sum of colors
    double sumR2 = 0, sumG2 = 0, sumB2 = 0; // sum of squares of colors
    double eApp = 0.0;
    double eReg = 0.0;
    int size = 0;                           // number of (image) pixels
    int initialSize = 0;

public:
    Superpixel() { }

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
        pd.p->superPixel = this;
    }

    // Add a pixel (block) to superpixel.
    // Sums of colors (for average color) and coordinates (for center of mass) are updated.
    // Note that borderLength is *not* updated and should be set separately!
    virtual void AddPixel(PixelData& pd)
    {
        AddPixelInit(pd);
        eApp = CalcAppEnergy(sumR, sumG, sumB, sumR2, sumG2, sumB2, size);
        eReg = CalcRegEnergy(sumRow, sumCol, sumRow2, sumCol2, size);
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
        eApp = CalcAppEnergy(sumR, sumG, sumB, sumR2, sumG2, sumB2, size);
        eReg = CalcRegEnergy(sumRow, sumCol, sumRow2, sumCol2, size);
    }

    virtual void FinishInitialization()
    {
        initialSize = size;
        eApp = CalcAppEnergy(sumR, sumG, sumB, sumR2, sumG2, sumB2, size);
        eReg = CalcRegEnergy(sumRow, sumCol, sumRow2, sumCol2, size);
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
    double sumDisp = 0.0;       // Sum of all disparity energy terms, Eq. (8) equals disp energy

    unordered_set<Pixel*> pixels;
    BInfoMapType boundaryData;
    Plane_d plane;

    SuperpixelStereo() : Superpixel() { }

    void AddPixelInit(PixelData& pd) override
    {
        Superpixel::AddPixelInit(pd);
        pixels.insert(pd.p);
    }

    void AddPixel(PixelData& pd) override
    {
        Superpixel::AddPixel(pd);
        sumDisp += pd.sumDispQ;
    }

    void RemovePixel(const PixelData& pd) override
    {
        Superpixel::RemovePixel(pd);
        pixels.erase(pd.p);
        sumDisp -= pd.sumDispP;
    }

    double GetDispEnergy()
    {
        return sumDisp;
    }

    double GetPriorEnergy() const
    {
        double result = 0.0;

        for (auto pair : boundaryData) {
            result += pair.second.prior;
        }
        return result;
    }

    double GetSmoEnergy()
    {
        double result = 0.0;

        for (auto pair : boundaryData) {
            if (pair.second.length != 0)
                result += pair.second.smo / pair.second.length;
        }
        return result;
    }

    void SetPlane(Plane_d& plane);
    void UpdateDispSum(const cv::Mat1w& depthImg, double beta);

    void ClearPixelSet() { pixels.clear(); }
    void AddToPixelSet(Pixel* p1) { pixels.insert(p1); }
    void AddToPixelSet(Pixel* p1, Pixel* p2) { pixels.insert(p1); pixels.insert(p2); }
    void AddToPixelSet(Pixel* p1, Pixel* p2, Pixel* p3, Pixel* p4) 
    { 
        pixels.insert(p1); pixels.insert(p2); 
        pixels.insert(p3); pixels.insert(p4);
    }

    void GetRemovePixelDataStereo(const PixelData& pd, const cv::Mat1w& dispImg,
        const Matrix<Pixel>& pixelsImg, Pixel* p, Pixel* q,
        PixelChangeDataStereo& pcd) const;
    void GetAddPixelDataStereo(const PixelData& pd, const cv::Mat1w& dispImg,
        const Matrix<Pixel>& pixelsImg, Pixel* p, Pixel* q,
        PixelChangeDataStereo& pcd) const;

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

inline double GetPriorEnergy(const unordered_map<Superpixel*, BInfo>& boundaryData)
{
    double result = 0.0;

    for (auto pair : boundaryData) {
        result += pair.second.prior;
    }
    return result;
}

inline double GetSmoEnergy(const unordered_map<Superpixel*, BInfo>& boundaryData)
{
    double result = 0.0;

    for (auto pair : boundaryData) {
        if (pair.second.length != 0)
            result += pair.second.smo / pair.second.length;
    }
    return result;
}
