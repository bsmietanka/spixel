#include "stdafx.h"
#include "structures.h"
#include "functions.h"

// Pixel defininions
///////////////////////////

// For debug purposes
string Pixel::GetPixelsAsString()
{
    stringstream ss;

    ss << '{';
    for (int r = ulr; r < lrr; r++) {
        for (int c = ulc; c < lrc; c++) {
            if (r != ulr || c != ulc) ss << ',';
            ss << '{' << r << ',' << c << '}';
        }
    }
    ss << '}';
    return ss.str();
}

void Pixel::Split(const cv::Mat& img, int row1, int row2, int col1, int col2, Pixel& p11, Pixel& p12, Pixel& p21, Pixel& p22)
{
    int cols = GetCSize();
    int cols1 = cols / 2;
    int rows = GetRSize();
    int rows1 = rows / 2;
    
    p11.Initialize(row1, col1, ulr, ulc, ulr + rows1, ulc + cols1);
    p12.Initialize(row1, col2, ulr, ulc + cols1, ulr + rows1, ulc + cols);
    p21.Initialize(row2, col1, ulr + rows1, ulc, ulr + rows, ulc + cols1);
    p22.Initialize(row2, col2, ulr + rows1, ulc + cols1, ulr + rows, ulc + cols);
    p11.superPixel = superPixel;
    p12.superPixel = superPixel;
    p21.superPixel = superPixel;
    p22.superPixel = superPixel;
    p11.border = border & (BTopFlag | BLeftFlag);
    p12.border = border & (BTopFlag | BRightFlag);
    p21.border = border & (BLeftFlag | BBottomFlag);
    p22.border = border & (BRightFlag | BBottomFlag);
}

void Pixel::SplitRow(const cv::Mat& img, int row1, int row2, int col, Pixel& p11, Pixel& p21)
{
    int cols = GetCSize();
    int rows = GetRSize();
    int rows1 = rows / 2;

    p11.Initialize(row1, col, ulr, ulc, ulr + rows1, ulc + cols);
    p21.Initialize(row2, col, ulr + rows1, ulc, ulr + rows, ulc + cols);
    p11.superPixel = superPixel;
    p21.superPixel = superPixel;
    p11.border = border & (BTopFlag | BLeftFlag);
    p21.border = border & (BLeftFlag | BBottomFlag);
}

void Pixel::CopyTo(const cv::Mat& img, int row, int col, Pixel& p11)
{
    int cols = GetCSize();
    int rows = GetRSize();

    p11.Initialize(row, col, ulr, ulc, ulr + rows, ulc + cols);
    p11.superPixel = superPixel;
    p11.border = border & (BTopFlag | BLeftFlag);
}

void Pixel::SplitColumn(const cv::Mat& img, int row, int col1, int col2, Pixel& p11, Pixel& p12)
{
    int cols = GetCSize();
    int cols1 = cols / 2;
    int rows = GetRSize();

    p11.Initialize(row, col1, ulr, ulc, ulr + rows, ulc + cols1);
    p12.Initialize(row, col2, ulr, ulc + cols1, ulr + rows, ulc + cols);
    p11.superPixel = superPixel;
    p12.superPixel = superPixel;
    p11.border = border & (BTopFlag | BLeftFlag);
    p12.border = border & (BTopFlag | BRightFlag);
}

void Pixel::UpdateInliers(const cv::Mat1d& dispImg, double threshold, cv::Mat1b& inliers) const
{
    const Plane_d& plane = ((SuperpixelStereo*)superPixel)->plane;

    for (int i = ulr; i < (int)lrr; i++) {
        for (int j = ulc; j < (int)lrc; j++) {
            const double& disp = dispImg(i, j);
            inliers(i, j) = fabs(DotProduct(plane, i, j, 1.0) - disp) < threshold;
        }
    }
}

//DEFINE_CLASS_ALLOCATOR_2(Pixel, 10000000)

// Superpixel defininions
///////////////////////////

// We remove Pixel p from this superpixel, recalculates size and energies
void Superpixel::GetRemovePixelData(const PixelData& pd, PixelChangeData& pcd) const
{
    pcd.newEApp = CalcAppEnergy(sumR - pd.sumR, sumG - pd.sumG, sumB - pd.sumB, 
        sumR2 - pd.sumR2, sumG2 - pd.sumG2, sumB2 - pd.sumB2, size - pd.size);
    pcd.newEReg = CalcRegEnergy(sumRow - pd.sumRow, sumCol - pd.sumCol,
        sumRow2 - pd.sumRow2, sumCol2 - pd.sumCol2, size - pd.size);
    pcd.newSize = size - pd.size;
}


// We add Pixel p to this superpixel, recalculates size and energies
void Superpixel::GetAddPixelData(const PixelData& pd, PixelChangeData& pcd) const
{
    pcd.newEApp = CalcAppEnergy(sumR + pd.sumR, sumG + pd.sumG, sumB + pd.sumB,
        sumR2 + pd.sumR2, sumG2 + pd.sumG2, sumB2 + pd.sumB2, size + pd.size);
    pcd.newEReg = CalcRegEnergy(sumRow + pd.sumRow, sumCol + pd.sumCol,
        sumRow2 + pd.sumRow2, sumCol2 + pd.sumCol2, size + pd.size);
    pcd.newSize = size + pd.size;
}

// SuperpixelStereo definitions
/////////////////////////////////

void SuperpixelStereo::SetPlane(Plane_d& plane_)
{
    plane = plane_;
}

void SuperpixelStereo::UpdateDispSum(const cv::Mat1d& depthImg, const cv::Mat1b& inliers, double beta)
{
    sumDisp = 0;
    for (Pixel* p : pixels) {
        sumDisp += p->CalcDispSum(depthImg, inliers, plane, beta);
    }
}

//void SuperpixelStereo::UpdateInlierSums(const cv::Mat1d& depthImg, const cv::Mat1b& inliers)
//{
//    sumIRow = sumICol = 0;
//    sumIRow2 = sumICol2 = 0;
//    sumIRowCol = 0;
//    sumIRowD = sumIColD = 0.0;
//    sumID = 0.0;
//    nI = 0;
//    for (Pixel* p : pixels) {
//        p->AddToInlierSums(depthImg, inliers,
//            sumIRow, sumIRow2, sumICol, sumICol2, sumIRowCol, sumIRowD, sumIColD, sumID, nI);
//    }
//}

void SuperpixelStereo::CalcPlaneLeastSquares(const cv::Mat1d& depthImg, const cv::Mat1b& inliers)
{
    LeastSquaresPlane(sumIRow, sumIRow2, sumICol, sumICol2, sumIRowCol, sumIRowD, sumIColD, sumID, nI, plane);
}

void SuperpixelStereo::GetRemovePixelDataStereo(const PixelData& pd, 
    const Matrix<Pixel>& pixelsImg,
    const cv::Mat1d& dispImg,
    const cv::Mat1b& inliers,
    Pixel* p, Pixel* q,
    PixelChangeDataStereo& pcd) const 
{
    GetRemovePixelData(pd, pcd);

    SuperpixelStereo* sp = (SuperpixelStereo*)p->superPixel;
    SuperpixelStereo* sq = (SuperpixelStereo*)q->superPixel;
    const Pixel* r;
    
    pcd.newEDisp = sumDisp - pd.sumDispP;
    pcd.newBoundaryData = boundaryData;
    int sqBType = pcd.newBoundaryData[sq].type;

    if (sqBType == BTCo) {
        pcd.newBoundaryData[sq].smo -= p->CalcCoSmoothnessSum(inliers, sp->plane, sq->plane);
    } else if (sqBType == BTHi) {
        double sum = 0;
        int count = 0;

        if ((r = PixelAt(pixelsImg, p->row, p->col + 1)) != nullptr) {
            if (r->superPixel == sq) {
                r->CalcHiSmoothnessSum(BLeftFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo -= sum;
                pcd.newBoundaryData[sq].length -= count;
            } else if (r->superPixel == sp) {
                r->CalcHiSmoothnessSum(BLeftFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo += sum;
                pcd.newBoundaryData[sq].length += count;
            }
        }
        if ((r = PixelAt(pixelsImg, p->row, p->col - 1)) != nullptr) {
            if (r->superPixel == sq) {
                p->CalcHiSmoothnessSum(BLeftFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo -= sum;
                pcd.newBoundaryData[sq].length -= count;
            } else if (r->superPixel == sp) {
                p->CalcHiSmoothnessSum(BLeftFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo += sum;
                pcd.newBoundaryData[sq].length += count;
            }
        }
        if ((r = PixelAt(pixelsImg, p->row + 1, p->col)) != nullptr) {
            if (r->superPixel == sq) {
                r->CalcHiSmoothnessSum(BTopFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo -= sum;
                pcd.newBoundaryData[sq].length -= count;
            } else if (r->superPixel == sp) {
                r->CalcHiSmoothnessSum(BTopFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo += sum;
                pcd.newBoundaryData[sq].length += count;
            }
        }
        if ((r = PixelAt(pixelsImg, p->row - 1, p->col)) != nullptr) {
            if (r->superPixel == sq) {
                p->CalcHiSmoothnessSum(BTopFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo -= sum;
                pcd.newBoundaryData[sq].length -= count;
            } else if (r->superPixel == sp) {
                p->CalcHiSmoothnessSum(BTopFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo += sum;
                pcd.newBoundaryData[sq].length += count;
            }
        }
    }
}

void SuperpixelStereo::GetAddPixelDataStereo(const PixelData& pd,
    const Matrix<Pixel>& pixelsImg,
    const cv::Mat1d& dispImg,
    const cv::Mat1b& inliers,
    Pixel* p, Pixel* q,
    PixelChangeDataStereo& pcd) const
{
    GetAddPixelData(pd, pcd);

    SuperpixelStereo* sp = (SuperpixelStereo*)p->superPixel;
    SuperpixelStereo* sq = (SuperpixelStereo*)q->superPixel;
    const Pixel* r;

    pcd.newEDisp = sumDisp + pd.sumDispQ;
    pcd.newBoundaryData = boundaryData;
    int sqBType = pcd.newBoundaryData[sq].type;

    if (sqBType == BTCo) {
        pcd.newBoundaryData[sq].smo += p->CalcCoSmoothnessSum(inliers, sp->plane, sq->plane);
    } else if (sqBType == BTHi) {
        double sum = 0;
        int count = 0;

        if ((r = PixelAt(pixelsImg, p->row, p->col + 1)) != nullptr) {
            if (r->superPixel == sq) {
                r->CalcHiSmoothnessSum(BLeftFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo += sum;
                pcd.newBoundaryData[sq].length += count;
            } else if (r->superPixel == sp) {
                r->CalcHiSmoothnessSum(BLeftFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo -= sum;
                pcd.newBoundaryData[sq].length -= count;
            }
        }
        if ((r = PixelAt(pixelsImg, p->row, p->col - 1)) != nullptr) {
            if (r->superPixel == sq) {
                p->CalcHiSmoothnessSum(BLeftFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo += sum;
                pcd.newBoundaryData[sq].length += count;
            } else if (r->superPixel == sp) {
                p->CalcHiSmoothnessSum(BLeftFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo -= sum;
                pcd.newBoundaryData[sq].length -= count;
            }
        }
        if ((r = PixelAt(pixelsImg, p->row + 1, p->col)) != nullptr) {
            if (r->superPixel == sq) {
                r->CalcHiSmoothnessSum(BTopFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo += sum;
                pcd.newBoundaryData[sq].length += count;
            } else if (r->superPixel == sp) {
                r->CalcHiSmoothnessSum(BTopFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo -= sum;
                pcd.newBoundaryData[sq].length -= count;
            }
        }
        if ((r = PixelAt(pixelsImg, p->row - 1, p->col)) != nullptr) {
            if (r->superPixel == sq) {
                p->CalcHiSmoothnessSum(BTopFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo += sum;
                pcd.newBoundaryData[sq].length += count;
            } else if (r->superPixel == sp) {
                p->CalcHiSmoothnessSum(BTopFlag, inliers, sp->plane, sq->plane, sum, count);
                pcd.newBoundaryData[sq].smo -= sum;
                pcd.newBoundaryData[sq].length -= count;
            }
        }
    }
}



