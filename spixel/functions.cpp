#include "stdafx.h"
#include "structures.h"
#include "functions.h"
#include <functional>

// Local functions, structures, classes
///////////////////////////////////////////////////////////////////////////////

class DistinctRandomNumberGenerator {
private:
    std::vector<int> v;
public:
    DistinctRandomNumberGenerator(int N) : v(N) 
    {
        for (int i = 0; i < N; i++) v[i] = i;
    }

    void GetNDistinct(int n, int result[])
    {
        int N = v.size();
        for (int k = 0; k < n; k++) {
            int l = rand() % N;
            result[k] = v[l];
            std::swap(v[l], v[--N]);
        }
    }
};

Matrix<int> BitsToPatch3x3(int b)
{
    Matrix<int> patch(3, 3);
    int bit = 1;

    for (int i = 0; i < 9; i++) {
        patch(i / 3, i % 3) = ((bit & b) == 0) ? 0 : 1;
        bit <<= 1;
    }
    return patch;
}

template<typename T> int Patch3x3ToBits(const Matrix<T>& m, int ulr, int ulc, const function<bool(const T*)>& compare)
{
    int patch = 0;
    int bit = 1;

    for (int r = ulr; r < ulr + 3; r++) {
        if (r >= m.rows)
            break;
        if (r < 0) {
            bit <<= 3;
        } else {
            for (int c = ulc; c < ulc + 3; c++) {
                if (c >= 0 && c < m.cols && compare(m.PtrTo(r, c))) patch |= bit;
                bit <<= 1;
            }
        }
    }
    return patch;
}


ConnectivityCache::ConnectivityCache()
{
    Initialize();
}

void ConnectivityCache::Initialize()
{
    cache.resize(512);
    for (int i = 0; i < 512; i++) {
        cache[i] = IsPatch3x3Connected(i);
    }
}

void ConnectivityCache::Print()
{
    for (int i = 0; i < 512; i++) {
        Matrix<int> patch = BitsToPatch3x3(i);
        
        cout << cache[i] << ':';
        for (int r = 0; r < 3; r++) {
            cout << '\t';
            for (int c = 0; c < 3; c++) {
                cout << patch(r, c) << ' ';
            }
            cout << endl;
        }
        cout << endl;
    }
}

bool IsPatch3x3Connected(int bits)
{
    Matrix<int> patch = BitsToPatch3x3(bits);
    int rows = 3;
    int cols = 3;
    UnionFind uf(rows*cols);
    cv::Mat1i comp = cv::Mat1i(rows + 1, cols + 1, -1);
    int cCount = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int q = patch(r, c);

            if (q == 1) {
                int topi = comp(r, c + 1);
                int lefti = comp(r + 1, c);

                if (topi < 0) {
                    if (lefti < 0) comp(r + 1, c + 1) = cCount++;
                    else comp(r + 1, c + 1) = lefti;
                }
                else { // topi >= 0
                    if (lefti < 0) comp(r + 1, c + 1) = topi;
                    else {
                        comp(r + 1, c + 1) = lefti;
                        uf.Union(lefti, topi);
                    }
                }
            }
        }
    }
    return uf.Size(0) == cCount;
}

// Static classes
///////////////////////////////////////////////////////////////////////////////

static ConnectivityCache connectivityCache;


// Timer
///////////////////////////////////////////////////////////////////////////////

ostream& operator<<(ostream& os, const Timer& t)
{
    os << ((double)(t.time) / CLOCKS_PER_SEC);
    return os;
}

// Function definitions
///////////////////////////////////////////////////////////////////////////////

// Return length of superpixel boundary -- for debug purposes only it's inefficient
int CalcSuperpixelBoundaryLength(const Matrix<Pixel>& pixelsImg, Superpixel* sp)
{
    int result = 0;

    for (const Pixel& p : pixelsImg) {
        if (p.superPixel == sp) {
            const Pixel* q = PixelAt(pixelsImg, p.row - 1, p.col);
            if (q == nullptr || q->superPixel != sp) result += p.GetCSize();
            q = PixelAt(pixelsImg, p.row + 1, p.col);
            if (q == nullptr || q->superPixel != sp) result += p.GetCSize();
            q = PixelAt(pixelsImg, p.row, p.col - 1);
            if (q == nullptr || q->superPixel != sp) result += p.GetRSize();
            q = PixelAt(pixelsImg, p.row, p.col + 1);
            if (q == nullptr || q->superPixel != sp) result += p.GetRSize();
        }
    }
    return result;
}

// Return length of boundary between Pixel p and superpixel sp (spbl), between Pixel p and superpixel sq (sqbl) and between
// Pixel p and other superpixel (sobl)
void CalcSuperpixelBoundaryLength(const Matrix<Pixel>& pixelsImg, Pixel* p, Superpixel* sp, Superpixel* sq,
    int& spbl, int& sqbl, int& sobl)
{
    spbl = sqbl = sobl = 0;
    
    const Pixel* q;

    q = PixelAt(pixelsImg, p->row - 1, p->col);
    if (q == nullptr || (q->superPixel != sp && q->superPixel != sq)) sobl += p->GetCSize(); 
    else if (q->superPixel == sp) spbl += p->GetCSize();
    else sqbl += p->GetCSize();
    q = PixelAt(pixelsImg, p->row, p->col - 1);
    if (q == nullptr || (q->superPixel != sp && q->superPixel != sq)) sobl += p->GetRSize();
    else if (q->superPixel == sp) spbl += p->GetRSize();
    else sqbl += p->GetRSize();
    q = PixelAt(pixelsImg, p->row + 1, p->col);
    if (q == nullptr || (q->superPixel != sp && q->superPixel != sq)) sobl += p->GetCSize();
    else if (q->superPixel == sp) spbl += p->GetCSize();
    else sqbl += p->GetCSize();
    q = PixelAt(pixelsImg, p->row, p->col + 1);
    if (q == nullptr || (q->superPixel != sp && q->superPixel != sq)) sobl += p->GetRSize();
    else if (q->superPixel == sp) spbl += p->GetRSize();
    else sqbl += p->GetRSize();
}

// Checks whether Pixel p separates Pixel q at (p->row + dRow, p->col + dCol) from superpixel sp
// At exactly one of dRow and dCol must be 0 and -1 <= dRow, dCol <= 1
//bool SeparatingPixel(const cv::Mat_<Pixel*>& pixelsImg, Pixel* p, int dRow, int dCol)
//{
//    Pixel* q = PixelAt(pixelsImg, p->row + dRow, p->col + dCol);
//
//    if (q == nullptr || p->superPixel != q->superPixel) return false; // on border || not in the same superpixel
//    if (dRow != 0) return q->bLeft && q->bRight;
//    if (dCol != 0) return q->bTop && q->bBottom;
//    return false;
//}
    

// Try to move Pixel p to Superpixel containing Pixel q with coordinates (qRow, qCol)
// Note: pixel q is must be neighbor of p and p->superPixel != q->superPixel
// Fills psd, returns psd.allowed
// Note: energy deltas in psd are "energy_before - energy_after"
bool TryMovePixel(const cv::Mat& img, const Matrix<Pixel>& pixelsImg, Pixel* p, Pixel* q, PixelMoveData& psd)
{
    Superpixel* sp = p->superPixel;
    Superpixel* sq = q->superPixel;

    if (sp == sq || !IsSuperpixelRegionConnectedOptimized(pixelsImg, p, p->row - 1, p->col - 1, p->row + 2, p->col + 2)) {
        psd.allowed = false;
        return false;
    }

    int spSize = sp->GetSize(), sqSize = sq->GetSize();
    double spEApp = sp->GetAppEnergy(), sqEApp = sq->GetAppEnergy();
    double spEReg = sp->GetRegEnergy(), sqEReg = sq->GetRegEnergy();

    PixelChangeData pcd;
    PixelChangeData qcd;
    PixelData pd;
    int spbl, sqbl, sobl;

    p->CalcPixelData(img, pd);
    sp->GetRemovePixelData(pd, pcd);
    sq->GetAddPixelData(pd, qcd);
    CalcSuperpixelBoundaryLength(pixelsImg, p, sp, sq, spbl, sqbl, sobl);

    psd.p = p;
    psd.q = q;
    psd.pSize = pcd.newSize;
    psd.qSize = qcd.newSize;
    psd.eAppDelta = spEApp + sqEApp - pcd.newEApp - qcd.newEApp;
    psd.eRegDelta = spEReg + sqEReg - pcd.newEReg - qcd.newEReg;
    psd.bLenDelta = sqbl - spbl;
    psd.allowed = true;
    psd.pixelData = pd;
    return true;
}

bool TryMovePixelStereo(const cv::Mat& img, const cv::Mat1d& dispImg, const Matrix<Pixel>& pixelsImg, 
    Pixel* p, Pixel* q, double beta, PixelMoveData& psd)
{
    SuperpixelStereo* sp = (SuperpixelStereo*)p->superPixel;
    SuperpixelStereo* sq = (SuperpixelStereo*)q->superPixel;

    if (sp == sq || !IsSuperpixelRegionConnectedOptimized(pixelsImg, p, p->row - 1, p->col - 1, p->row + 2, p->col + 2)) {
        psd.allowed = false;
        return false;
    }

    int spSize = sp->GetSize(), sqSize = sq->GetSize();
    double spEApp = sp->GetAppEnergy(), sqEApp = sq->GetAppEnergy();
    double spEReg = sp->GetRegEnergy(), sqEReg = sq->GetRegEnergy();
    double spEDisp = sp->GetDispEnergy(), sqEDisp = sq->GetDispEnergy();
    double spESmo = sp->GetSmoEnergy(), sqESmo = sq->GetSmoEnergy();
    double spEPrior = sp->GetPriorEnergy(), sqEPrior = sq->GetPriorEnergy();
    
    PixelChangeDataStereo pcd;
    PixelChangeDataStereo qcd;
    PixelData pd;
    int spbl, sqbl, sobl;

    p->CalcPixelDataStereo(img, dispImg, sp->plane, sq->plane, beta, pd);
    sp->GetRemovePixelDataStereo(pd, dispImg, pixelsImg, p, q, pcd);
    sq->GetAddPixelDataStereo(pd, dispImg, pixelsImg, p, q, qcd);
    CalcSuperpixelBoundaryLength(pixelsImg, p, sp, sq, spbl, sqbl, sobl);

    psd.p = p;
    psd.q = q;
    psd.pSize = pcd.newSize;
    psd.qSize = qcd.newSize;
    psd.eAppDelta = spEApp + sqEApp - pcd.newEApp - qcd.newEApp;
    psd.eRegDelta = spEReg + sqEReg - pcd.newEReg - qcd.newEReg;
    psd.bLenDelta = sqbl - spbl;
    psd.eDispDelta = spEDisp + sqEDisp - pcd.newEDisp - qcd.newEDisp;
    psd.ePriorDelta = 0;
    psd.eSmoDelta = spESmo + sqESmo - GetSmoEnergy(pcd.newBoundaryData) - GetSmoEnergy(qcd.newBoundaryData);
    psd.allowed = true;
    psd.pixelData = pd;
    psd.bDataP = std::move(pcd.newBoundaryData);
    psd.bDataQ = std::move(qcd.newBoundaryData);
    return true;
}

// Move Pixel p from superpixel of p to superpixel of q
void MovePixel(Matrix<Pixel>& pixelsImg, PixelMoveData& pmd)
{
    Pixel* p = pmd.p;
    Pixel* q = pmd.q;
    Superpixel* sp = p->superPixel;
    Superpixel* sq = q->superPixel;
    int spbl, sqbl, sobl;

    CalcSuperpixelBoundaryLength(pixelsImg, p, sp, sq, spbl, sqbl, sobl);

    // Change precalculated sums
    sp->RemovePixel(pmd.pixelData);
    sp->AddToBorderLength(spbl - sqbl - sobl);
    sq->AddPixel(pmd.pixelData);
    sq->AddToBorderLength(spbl - sqbl + sobl);

    // Fix border information
    Pixel* r;
    
    if ((r = PixelAt(pixelsImg, p->row, p->col - 1)) != nullptr && (r->superPixel == sq || r->superPixel == sp)) {
        r->SwapBRight();
        p->SwapBLeft();
    }
    if ((r = PixelAt(pixelsImg, p->row, p->col + 1)) != nullptr && (r->superPixel == sq || r->superPixel == sp)) {
        r->SwapBLeft();
        p->SwapBRight();
    }
    if ((r = PixelAt(pixelsImg, p->row - 1, p->col)) != nullptr && (r->superPixel == sq || r->superPixel == sp)) {
        r->SwapBBottom();
        p->SwapBTop();
    }
    if ((r = PixelAt(pixelsImg, p->row + 1, p->col)) != nullptr && (r->superPixel == sq || r->superPixel == sp)) {
        r->SwapBTop();
        p->SwapBBottom();
    }

}

void MovePixelStereo(Matrix<Pixel>& pixelsImg, PixelMoveData& pmd)
{
    MovePixel(pixelsImg, pmd);

    SuperpixelStereo* sp = (SuperpixelStereo*)pmd.p->superPixel;
    SuperpixelStereo* sq = (SuperpixelStereo*)pmd.q->superPixel;

    sp->boundaryData = std::move(pmd.bDataP);
    sq->boundaryData = std::move(pmd.bDataQ);
}

// Return true if superpixel sp is connected in region defined by upper left/lower right corners of pixelsImg
// Corners are adjusted to be valid for image but we assume that lrr >= 0 and lrc >= 0 and ulr < pixelsImg.rows and ulc < pixelsImg.cols
bool IsSuperpixelRegionConnected(const Matrix<Pixel>& pixelsImg, Pixel* p, int ulr, int ulc, int lrr, int lrc)
{
    if (ulr < 0) ulr = 0;
    if (ulc < 0) ulc = 0;
    if (lrr >= pixelsImg.rows) lrr = pixelsImg.rows - 1;
    if (lrc >= pixelsImg.cols) lrc = pixelsImg.cols - 1;

    int rows = lrr - ulr;
    int cols = lrc - ulc;
    UnionFind uf(rows * cols);
    cv::Mat1i comp = cv::Mat1i(rows + 1, cols + 1, -1);
    int cCount = 0;

    for (int r = ulr; r < lrr; r++) {
        int ir = r - ulr + 1;

        for (int c = ulc; c < lrc; c++) {
            const Pixel* q = &pixelsImg(r, c);

            if (p != q && q->superPixel == p->superPixel) {
                int ic = c - ulc + 1;
                int topi = comp(ir - 1, ic);
                int lefti = comp(ir, ic - 1);

                if (topi < 0) {
                    if (lefti < 0) comp(ir, ic) = cCount++;
                    else comp(ir, ic) = lefti;
                } else { // topi >= 0
                    if (lefti < 0) comp(ir, ic) = topi;
                    else {
                        comp(ir, ic) = lefti;
                        uf.Union(lefti, topi);
                    }
                }
            }
        }
    }
    return uf.Size(0) == cCount;
}

// See IsSuperpixelRegionConnected; optimized version which uses pre-calculated connectivity data for 3x3 regions
bool IsSuperpixelRegionConnectedOptimized(const Matrix<Pixel>& pixelsImg, Pixel* p, int ulr, int ulc, int lrr, int lrc)
{
    CV_DbgAssert(ulr + 3 == lrr && ulc + 3 == lrc);

    int bits = Patch3x3ToBits<Pixel>(pixelsImg, ulr, ulc, [&p](const Pixel* q) { return p != q && q->superPixel == p->superPixel; });

    return connectivityCache.IsConnected(bits);
}

// RANSAC & supporting functions
///////////////////////////////////////////////////////////////////////////////

// Try to fit plane to points
bool Plane3P(const cv::Point3d& p1, const cv::Point3d& p2, const cv::Point3d& p3, Plane_d& plane)
{
    cv::Point3d normal = (p1 - p2).cross(p1 - p3);
    if (normal.z < eps && -normal.z < eps) return false;
    else {
        plane.x = -normal.x/normal.z;
        plane.y = -normal.y/normal.z;
        plane.z = p1.z - plane.x*p1.x - plane.y*p1.y;
    }
    return true;
}


// See http://en.wikipedia.org/wiki/RANSAC
// In our case, it will be n = 3, p = 0.99
int EstimateRANSACSteps(int n, int nInliers, int nPoints, double p)
{
    double w = (double)nInliers / nPoints;

    if (w == 1.0) return 1;
    return (int)round(log(1 - p) / log(1 - pow(w, n))); 
}

// Returns false if pixels.size() < 3 or no 3 points were found to 
// form a plane; pixels are (xi, yi, di)
// Plane parameters (a, b, c) satisfy equations (xi, yi, 1).(a, b, c) == di
// for selected three points
bool RANSACPlane(const vector<cv::Point3d>& pixels, Plane_d& plane)
{
    const double inlierThreshold = 1.0;
    const double confidence = 0.99;

    if (pixels.size() < 3)
        return false;

    int bestInlierCount = 0;
    Plane_d stepPlane;
    int N = 2*pixels.size();
    int n = 0;

    while (n < N) {
        const cv::Point3d& p1 = pixels[rand() % pixels.size()];
        const cv::Point3d& p2 = pixels[rand() % pixels.size()];
        const cv::Point3d& p3 = pixels[rand() % pixels.size()];

        if (Plane3P(p1, p2, p3, stepPlane)) {
            int inlierCount = 0;
            for (const cv::Point3d& p : pixels) {
                if (fabs(p.x*stepPlane.x + p.y*stepPlane.y + stepPlane.z - p.z) < inlierThreshold) {
                    inlierCount++;
                }
            }
            if (inlierCount > bestInlierCount) {
                bestInlierCount = inlierCount;
                plane = stepPlane;

                int NN = EstimateRANSACSteps(3, bestInlierCount, pixels.size(), confidence);
                
                if (NN < N) N = NN;
            }
        }
        n++;
    }
    return bestInlierCount > 0;
}


bool UpdateSuperpixelPlaneRANSAC(SuperpixelStereo* sp, const cv::Mat1d& depthImg, double beta)
{
    vector<cv::Point3d> pixels;
    Plane_d plane;

    pixels.reserve(sp->GetSize());
    for (Pixel* p : sp->pixels) {
        p->AddDispPixels(depthImg, pixels);

    }
    if (!RANSACPlane(pixels, plane)) return false;
    sp->SetPlane(plane);
    sp->UpdateDispSum(depthImg, beta);
    return true;
}

double CalcDispEnergy(SuperpixelStereo* sp, cv::Mat1d& dispImg, double beta)
{
    double result = 0.0;

    for (Pixel* p : sp->pixels) {
        const double& disp = dispImg(p->row, p->col);

        if (disp == 0) continue;
        double delta = (DotProduct(sp->plane, p->row, p->col, 1.0) - disp);
        delta *= delta;
        result += max(beta, delta);
    }
    return result;
}

void CalcOccSmoothnessEnergy(SuperpixelStereo* sp, SuperpixelStereo* sq, double occWeight, double hingeWeight,
    double& ePrior, double& eSmo)
{
    ePrior = occWeight;
    eSmo = 0;
}

double CalcCoSmoothnessSum(SuperpixelStereo* sp, SuperpixelStereo* sq)
{
    if (sp->pixels.empty() && sq->pixels.empty())
        return 0.0;

    double eSmo = 0;

    for (Pixel* p : sp->pixels) {
        eSmo += p->CalcCoSmoothnessSum(sp->plane, sq->plane);
    }
    for (Pixel* q : sq->pixels) {
        eSmo += q->CalcCoSmoothnessSum(sp->plane, sq->plane);
    }
    return eSmo;
}

/*
void CalcBTEnergy(SuperpixelStereo* sp, SuperpixelStereo* sq, double occWeight, double hingeWeight,
    double& ePrior, double& eSmo)
{
    int type = sp->boundaryType[sq];

    if (type == BTLo || type == BTRo) {
        ePrior = occWeight;
        eSmo = 0;
    } else if (type == BTHi) {
        ePrior = hingeWeight;
        eSmo = CalcHiSmoothnessEnergy(sp, sq);
    } else if (type == BTCo) {
        ePrior = 0;
        eSmo = CalcCoSmoothnessEnergy(sp, sq);
    }
}
*/
