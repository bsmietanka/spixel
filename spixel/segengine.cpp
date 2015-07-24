#include "stdafx.h"
#include "segengine.h"
#include "functions.h"
#include "utils.h"
#include "tsdeque.h"
#include <unordered_map>
#include <fstream>   
#include <thread>


// Constants
///////////////////////////////////////////////////////////////////////////////

static const int nDeltas[][2] = { { -1, 0 }, { 1, 0 }, { 0, 1 }, { 0, -1 } };
static const int nDeltas0[][2] = { { 0, 0 }, { -1, 0 }, { 1, 0 }, { 0, 1 }, { 0, -1 } };


// Local functions
///////////////////////////////////////////////////////////////////////////////

double TotalEnergyDelta(const SPSegmentationParameters& params, PixelMoveData* pmd) 
{
    double eSum = 0.0;
    int initPSizeQuarter = pmd->p->superPixel->GetInitialSize()/4;
    
    eSum += params.regWeight * pmd->eRegDelta;
    eSum += params.appWeight * pmd->eAppDelta;
    eSum += params.lenWeight * pmd->bLenDelta;
    if (params.stereo) {
        eSum += params.dispWeight * pmd->eDispDelta;
        eSum += params.priorWeight * pmd->ePriorDelta;
        eSum += params.smoWeight * pmd->eSmoDelta;
    }

    if (pmd->pSize < initPSizeQuarter)
        eSum -= params.sizeWeight * (initPSizeQuarter - pmd->pSize);
    return eSum;
}

// On-the-fly re-estimation
void UpdateHingeBoundaryData(const Pixel* r, byte sideFlag, const Pixel* sideP, SuperpixelStereo* sp, SuperpixelStereo* sq,
    const cv::Mat1d& depthImg, double inlierThresh,
    BorderDataMap& bcdp, BorderDataMap& bcdq,
    unordered_set<SuperpixelStereo*>& nbsps)
{
    if (r == nullptr || sideP == nullptr) return;

    SuperpixelStereo* sr = (SuperpixelStereo*)r->superPixel;
    double sum = 0;
    int count = 0;
    int length = 0;

    if (sr == sp) {
        sideP->CalcHiSmoothnessSumEI(sideFlag, depthImg, inlierThresh, sp->plane, sq->plane, sum, count, length);
        BInfo& bip = bcdp[sq];
        bip.hiSum += sum;
        bip.hiCount += count;
        bip.length += length;
        BInfo& biq = bcdq[sp];
        biq.hiSum += sum;
        biq.hiCount += count;
        biq.length += length;
    } else if (sr == sq) {
        sideP->CalcHiSmoothnessSumEI(sideFlag, depthImg, inlierThresh, sp->plane, sq->plane, sum, count, length);
        BInfo& bip = bcdp[sq];
        bip.hiSum -= sum;
        bip.hiCount -= count;
        bip.length -= length;
        BInfo& biq = bcdq[sp];
        biq.hiSum -= sum;
        biq.hiCount -= count;
        biq.length -= length;
    } else {
        sideP->CalcHiSmoothnessSumEI(sideFlag, depthImg, inlierThresh, sp->plane, sr->plane, sum, count, length);
        BInfo& bip = bcdp[sr];
        bip.hiSum -= sum;
        bip.hiCount -= count;
        bip.length -= length;
        sideP->CalcHiSmoothnessSumEI(sideFlag, depthImg, inlierThresh, sq->plane, sr->plane, sum, count, length);
        BInfo& biq = bcdq[sr];
        biq.hiSum += sum;
        biq.hiCount += count;
        biq.length += length;
        nbsps.insert(sr);
    }
}

void EstimateBorderType(const SPSegmentationParameters& params, SuperpixelStereo* sp, int newSpSize, 
    SuperpixelStereo* sq, int newSqSize, BInfo& bInfo)
{
    double eHi = bInfo.hiCount == 0 ? HUGE_VAL : (bInfo.hiSum / bInfo.hiCount + params.hiPriorWeight);
    double eCo = bInfo.coCount == 0 ? HUGE_VAL : (bInfo.coSum / bInfo.coCount); // (newSpSize + newSqSize); // + 0
    double eOcc = params.occPriorWeight;

    if (eCo <= eHi && eCo < eOcc) {
        bInfo.type = BTCo;
        bInfo.typePrior = 0;
    } else if (eHi <= eOcc && eHi <= eCo) {
        bInfo.type = BTHi;
        bInfo.typePrior = params.hiPriorWeight;
    } else {
        bInfo.type = BTLo;
        bInfo.typePrior = params.occPriorWeight;
    }
}

// On-the-fly re-estimation
void CalcBorderChangeDataStereo(const Matrix<Pixel>& pixelsImg, const cv::Mat1d& depthImg, 
    const SPSegmentationParameters& params, Pixel* p, Pixel* q, BorderDataMap& bcdp, BorderDataMap& bcdq, unordered_set<SuperpixelStereo*>& prem)
{
    // Move p from sp -> sq
    SuperpixelStereo* sp = (SuperpixelStereo*)p->superPixel;
    SuperpixelStereo* sq = (SuperpixelStereo*)q->superPixel;
    const Pixel* r;
    unordered_set<SuperpixelStereo*> nbsps;    // Neighbors of p (other than sp and sq), make something more lightwight

    // Make copy
    bcdp = sp->boundaryData;
    bcdq = sq->boundaryData;

    // Update hinge sums for neighbors of p
    r = PixelAt(pixelsImg, p->row, p->col + 1);
    UpdateHingeBoundaryData(r, BLeftFlag, r, sp, sq, depthImg, params.inlierThreshold, bcdp, bcdq, nbsps);
    r = PixelAt(pixelsImg, p->row, p->col - 1);
    UpdateHingeBoundaryData(r, BLeftFlag, p, sp, sq, depthImg, params.inlierThreshold, bcdp, bcdq, nbsps);
    r = PixelAt(pixelsImg, p->row + 1, p->col);
    UpdateHingeBoundaryData(r, BTopFlag, r, sp, sq, depthImg, params.inlierThreshold, bcdp, bcdq, nbsps);
    r = PixelAt(pixelsImg, p->row - 1, p->col);
    UpdateHingeBoundaryData(r, BTopFlag, p, sp, sq, depthImg, params.inlierThreshold, bcdp, bcdq, nbsps);

    // Update coplanarity sums for neighbors of sp and sq
    for (auto& bdIter : sp->boundaryData) {
        SuperpixelStereo* sr = bdIter.first;
        if (sr != sq) {
            BInfo& bi = bcdp[sr];
            p->SubtractFromCoSmoothnessSum(depthImg, params.inlierThreshold, sp->plane, sr->plane, bi.coSum, bi.coCount);
        }
    }
    for (auto& bdIter : sq->boundaryData) {
        SuperpixelStereo* sr = bdIter.first;
        if (sr != sp) {
            BInfo& bi = bcdq[sr];
            p->AddToCoSmoothnessSum(depthImg, params.inlierThreshold, sq->plane, sr->plane, bi.coSum, bi.coCount);
        }
        nbsps.erase(sr);
    }

    // Eventual new neighbor(s) of sq
    for (SuperpixelStereo* snew : nbsps) {
        BInfo& bi = bcdq[snew];
        CalcCoSmoothnessSum(depthImg, params.inlierThreshold, sq, snew, bi.coSum, bi.coCount);
        p->AddToCoSmoothnessSum(depthImg, params.inlierThreshold, sq->plane, snew->plane, bi.coSum, bi.coCount);
    }

    prem.clear();

    // Remove "neighbors" of sp with length 0
    for (auto& bdIter : bcdp) {
        if (bdIter.second.length <= 0) prem.insert(bdIter.first);
    }
    for (SuperpixelStereo* sr : prem) {
        bcdp.erase(sr);
    }

    // Estimate type for each neighbor
    for (auto& bdIter : bcdp) {
        EstimateBorderType(params, sp, sp->GetSize() - p->GetSize(), bdIter.first, bdIter.first->GetSize() + ((bdIter.first == sq) ? p->GetSize() : 0), bdIter.second);
    }
    for (auto& bdIter : bcdq) {
        EstimateBorderType(params, sq, sq->GetSize() + p->GetSize(), bdIter.first, bdIter.first->GetSize() - ((bdIter.first == sp) ? p->GetSize() : 0), bdIter.second);
    }

}

PixelMoveData* FindBestMoveData(const SPSegmentationParameters& params, PixelMoveData d[4])
{
    PixelMoveData* dArray[4];
    int dArraySize = 0;

    // Only allowed moves and energy delta should be positive (or over threshold)
    for (int m = 0; m < 4; m++) {
        PixelMoveData* md = &d[m];
        if (md->allowed && TotalEnergyDelta(params, md) > params.updateThreshold) dArray[dArraySize++] = md;
    }

    if (dArraySize == 0) return nullptr;

    return *std::max_element(dArray, dArray + dArraySize, 
        [&params](PixelMoveData* a, PixelMoveData* b) { return TotalEnergyDelta(params, a) < TotalEnergyDelta(params, b); });
}

PixelMoveData* FindBestMoveData(const SPSegmentationParameters& params, PixelMoveData* d1, PixelMoveData* d2)
{
    if (!d1->allowed || TotalEnergyDelta(params, d1) <= 0) return (!d2->allowed || TotalEnergyDelta(params, d2) <= 0) ? nullptr : d2;
    else if (!d2->allowed || TotalEnergyDelta(params, d2) <= 0) return d1;
    else return TotalEnergyDelta(params, d1) < TotalEnergyDelta(params, d2) ? d2 : d1;
}


double GetSmoEnergy(const BorderDataMap& bd, SuperpixelStereo* sp, int pSize, SuperpixelStereo* sq, int qSize)
{
    double result = 0.0;

    for (auto& bdItem : bd) {
        const BInfo& bInfo = bdItem.second;
        if (bInfo.length > 0) {
            if (bInfo.type == BTCo) result += bInfo.coSum / bInfo.coCount; // (pSize + (bdItem.first == sq ? qSize : bdItem.first->GetSize()));
            else if (bInfo.type == BTHi) result += bInfo.hiSum / bInfo.hiCount;
        }
    }
    return result;
}

// SPSegmentationParameters
///////////////////////////////////////////////////////////////////////////////

static void read(const FileNode& node, SPSegmentationParameters& x, const SPSegmentationParameters& defaultValue)
{
    if (node.empty()) x = defaultValue;
    else x.Read(node);
}

void SPSegmentationParameters::SetLevelParams(int level)
{
    for (pair<string, vector<double>>& pData : levelParamsDouble) {
        if (!pData.second.empty()) {
            updateDouble[pData.first](*this,
                (level < pData.second.size()) ? pData.second[level] : pData.second.back());
        }
    }
    for (pair<string, vector<int>>& pData : levelParamsInt) {
        if (!pData.second.empty()) {
            updateInt[pData.first](*this,
                (level < pData.second.size()) ? pData.second[level] : pData.second.back());
        }
    }
}


void UpdateFromNode(double& val, const FileNode& node)
{
    if (!node.empty()) val = (double)node;
}

void UpdateFromNode(int& val, const FileNode& node)
{
    if (!node.empty()) val = (int)node;
}

void UpdateFromNode(bool& val, const FileNode& node)
{
    if (!node.empty()) val = (int)node != 0;
}

// SPSegmentationEngine
///////////////////////////////////////////////////////////////////////////////


SPSegmentationEngine::SPSegmentationEngine(SPSegmentationParameters params, Mat im, Mat depthIm) :
    params(params), origImg(im)
{
    img = ConvertRGBToLab(im);
    depthImg = AdjustDisparityImage(depthIm);
    if (params.stereo) {
        if (params.inpaint) depthImgAdj = InpaintDisparityImage(depthImg);
        else depthImgAdj = FillGapsInDisparityImage(depthImg);
    }
    //depthImg = FillGapsInDisparityImage(depthImg);
    //inliers = Mat1b(depthImg.rows, depthImg.cols);
}

SPSegmentationEngine::~SPSegmentationEngine()
{
    Reset();
}

void SPSegmentationEngine::Initialize(Superpixel* spGenerator(int))
{
    int pixelSize = params.pixelSize;
    int imgPixelsRows = img.rows / pixelSize + (img.rows % pixelSize == 0 ? 0 : 1);
    int imgPixelsCols = img.cols / pixelSize + (img.cols % pixelSize == 0 ? 0 : 1);

    // Initialize 'pixels', 'pixelsImg'
    pixelsImg = Matrix<Pixel>(imgPixelsRows, imgPixelsCols);

    for (int pi = 0; pi < imgPixelsRows; pi++) {
        for (int pj = 0; pj < imgPixelsCols; pj++) {
            int i0 = pi*pixelSize;
            int j0 = pj*pixelSize;
            int i1 = min(i0 + pixelSize, img.rows);
            int j1 = min(j0 + pixelSize, img.cols);
            pixelsImg(pi, pj).Initialize(pi, pj, i0, j0, i1, j1);
        }
    }

    // Create superpixels (from 'pixelsImg' matrix) and borders matrices
    int sPixelSize = params.sPixelSize;
    int imgSPixelsRows = imgPixelsRows / sPixelSize + (imgPixelsRows % sPixelSize == 0 ? 0 : 1);
    int imgSPixelsCols = imgPixelsCols / sPixelSize + (imgPixelsCols % sPixelSize == 0 ? 0 : 1);
    PixelData pd;
    int superPixelIdCount = 0;

    superpixels.clear();
    superpixels.reserve(imgSPixelsCols*imgSPixelsRows);
    for (int pi = 0; pi < imgSPixelsRows; pi++) {
        for (int pj = 0; pj < imgSPixelsCols; pj++) {
            int i0 = pi*sPixelSize;
            int j0 = pj*sPixelSize;
            int i1 = min(i0 + sPixelSize, pixelsImg.rows);
            int j1 = min(j0 + sPixelSize, pixelsImg.cols);
            Superpixel* sp = spGenerator(superPixelIdCount++); // Superpixel();

            // Update superpixels pointers in each pixel
            for (int i = i0; i < i1; i++) {
                for (int j = j0; j < j1; j++) {
                    pixelsImg(i, j).CalcPixelData(img, pd);
                    sp->AddPixelInit(pd);
                }
            }
            sp->FinishInitialization();

            // Init pixelsBorder matrix and border length and border info in each Pixel
            int spRSize = 0, spCSize = 0;

            for (int i = i0; i < i1; i++) {
                pixelsImg(i, j0).SetBLeft();
                pixelsImg(i, j1 - 1).SetBRight();
                spRSize += pixelsImg(i, j0).GetRSize();
            }
            for (int j = j0; j < j1; j++) {
                pixelsImg(i0, j).SetBTop();
                pixelsImg(i1 - 1, j).SetBBottom();
                spCSize += pixelsImg(i0, j).GetCSize();
            }
            sp->SetBorderLength(2 * spRSize + 2 * spCSize);
            superpixels.push_back(sp);
        }
    }

}

void SPSegmentationEngine::InitializeStereo()
{
    Initialize([](int id) -> Superpixel* { return new SuperpixelStereo(id); });
    InitializePPImage();
    EstimatePlaneParameters();
    InitializeStereoEnergies();
}

void SPSegmentationEngine::InitializePPImage()
{
    ppImg = Matrix<Pixel*>(img.rows, img.cols);
    UpdatePPImage();
}

void SPSegmentationEngine::UpdatePPImage()
{
    for (Pixel& p : pixelsImg) {
        p.UpdatePPImage(ppImg);
    }
}


void SPSegmentationEngine::Reset()
{
    for (Superpixel* sp : superpixels) {
        delete sp;
    }
}

// Returns time spent (in seconds)
void SPSegmentationEngine::ProcessImage()
{
    Timer t0;

    Initialize([](int id) { return new Superpixel(id); });

    t0.Stop();
    performanceInfo.init = t0.GetTimeInSec();
    t0.Resume();

    Timer t1;
    bool splitted;
    int level = ceil(log2(params.pixelSize));

    do {
        Timer t2;

        performanceInfo.levelIterations.push_back(0);
        for (int iteration = 0; iteration < params.iterations; iteration++) {
            int iters = IterateMoves(level);
            if (iters > performanceInfo.levelIterations.back())
                performanceInfo.levelIterations.back() = iters;
        }
        splitted = SplitPixels();
        t2.Stop();
        performanceInfo.levelTimes.push_back(t2.GetTimeInSec());
    } while (splitted && level >= params.minLevel);

    t0.Stop();
    t1.Stop();
    performanceInfo.total = t0.GetTimeInSec();
    performanceInfo.imgproc = t1.GetTimeInSec();
}

void SPSegmentationEngine::ProcessImageStereo()
{
    Timer t0;

    InitializeStereo();

    t0.Stop();
    performanceInfo.init = t0.GetTimeInSec();
    t0.Resume();

    Timer t1;
    bool splitted;
    int level = ceil(log2(params.pixelSize));

    //imwrite("c:\\tmp\\a--.png", this->GetSegmentedImage());

    do {
        Timer t2;
        performanceInfo.levelIterations.push_back(0);
        for (int iteration = 0; iteration < params.iterations; iteration++) {
            int iters = IterateMoves(level);

            //imwrite(Format("c:\\tmp\\dbgdisp-%d-I.png", level), GetSegmentedImage());

            //stringstream ss;
            //ss << levelCount;
            //imwrite("c:\\tmp\\a-" + ss.str() + ".png", this->GetSegmentedImage());
            if (iters > performanceInfo.levelIterations.back())
                performanceInfo.levelIterations.back() = iters;
            ReEstimatePlaneParameters();
            //imwrite(Format("c:\\tmp\\dbgdisp-%d-R.png", level), GetSegmentedImage());
        }
        splitted = SplitPixels();
        level--;

        t2.Stop();
        performanceInfo.levelTimes.push_back(t2.GetTimeInSec());
    } while (splitted && level >= params.minLevel);

    t0.Stop();
    t1.Stop();
    performanceInfo.total = t0.GetTimeInSec();
    performanceInfo.imgproc = t1.GetTimeInSec();
}

void SPSegmentationEngine::ReEstimatePlaneParameters()
{
    for (int s = 0; s < params.reSteps; s++) {
        UpdatePlaneParameters();
        UpdateBoundaryData();
    }
    UpdateDispSums();
}

void SPSegmentationEngine::UpdatePlaneParameters()
{
    UpdateStereoSums();
    for (int i = 0; i < superpixels.size(); i++) {
        SuperpixelStereo* sp = (SuperpixelStereo*)superpixels[i];
        bool updated = false;

        //for (auto& bd : sp->boundaryData) {
        //    SuperpixelStereo* sq = (SuperpixelStereo*)bd.first;
        //    if (bd.second.type == BTCo /* && sp < bd.first */) {
        //        sp->CalcPlaneLeastSquares(sq, depthImg);
        //        updated = true;
        //    }
        //}
        //if (!updated) {
        //    sp->CalcPlaneLeastSquares(depthImg);
        //}
        sp->CalcPlaneLeastSquares(sp->boundaryData.begin(), sp->boundaryData.end(), 
            [this](BorderDataMap::const_iterator iter) { 
                return iter->second.type == BTCo && iter->second.length > params.peblThreshold ? iter->first : nullptr; 
            },
            depthImg);
        //sp->CalcPlaneLeastSquares(sp->boundaryData.begin(), sp->boundaryData.end(),
        //    [](BorderDataMap::const_iterator iter) { return nullptr; },
        //    depthImg);
    }
}

void SPSegmentationEngine::PrintDebugInfo2()
{
    ofstream ofs("C:\\tmp\\debugse2.txt");
    for (Superpixel* sp : superpixels) {
        SuperpixelStereo* sps = (SuperpixelStereo*)sp;
        ofs << sps->plane.x << " " << sps->plane.y << " " << sps->plane.z << endl;
    }
    ofs.close();
}

void SPSegmentationEngine::EstimatePlaneParameters()
{
    Timer t;

    #pragma omp parallel for
    for (int i = 0; i < superpixels.size(); i++) {
        SuperpixelStereo* sp = (SuperpixelStereo*)superpixels[i];
        //InitSuperpixelPlane(sp, depthImg);
        InitSuperpixelPlane(sp, depthImgAdj);
    }
    t.Stop();
    performanceInfo.ransac += t.GetTimeInSec();
    //PrintDebugInfo2();

    //UpdateInliers();

}

void SPSegmentationEngine::InitializeStereoEnergies()
{
    //UpdateInliers();
    //UpdateStereoSums();
    UpdateBoundaryData();
    UpdateDispSums();
    //DebugBoundary();
}
 
void SPSegmentationEngine::UpdateStereoSums()
{
    for (Superpixel* sp : superpixels) {
        SuperpixelStereo* sps = (SuperpixelStereo*)sp;

        sps->sumIRow = 0; sps->sumICol = 0;         // Sum of terms computed for inlier points
        sps->sumIRow2 = 0; sps->sumICol2 = 0;
        sps->sumIRowCol = 0;
        sps->sumIRowD = 0.0, sps->sumIColD = 0.0;
        sps->sumID = 0.0;
        sps->nI = 0;
    }
    for (int i = 0; i < ppImg.rows; i++) {
        for (int j = 0; j < ppImg.cols; j++) {
            Pixel* p = ppImg(i, j);
            SuperpixelStereo* sps = (SuperpixelStereo*)p->superPixel;
            const double& disp = depthImg(i, j);

            if (disp > 0) {
                double delta = DotProduct(sps->plane, i, j, 1.0) - disp;

                if (fabs(delta) < params.inlierThreshold) {
                    sps->sumIRow += i; sps->sumIRow2 += i*i;
                    sps->sumICol += j; sps->sumICol2 += j*j;
                    sps->sumIRowCol += i*j;
                    sps->sumIRowD += i*disp; sps->sumIColD += j*disp;
                    sps->sumID += disp;
                    sps->nI++;
                }
            }
        }
    }
}

void SPSegmentationEngine::UpdateDispSums()
{
    for (Superpixel* sp : superpixels) {
        SuperpixelStereo* sps = (SuperpixelStereo*)sp;

        sps->sumDisp = 0.0;
    }
    for (int i = 0; i < ppImg.rows; i++) {
        for (int j = 0; j < ppImg.cols; j++) {
            Pixel* p = ppImg(i, j);
            SuperpixelStereo* sps = (SuperpixelStereo*)p->superPixel;
            const double& disp = depthImg(i, j);

            if (disp > 0) {
                double delta = DotProduct(sps->plane, i, j, 1.0) - disp;

                if (fabs(delta) < params.inlierThreshold) sps->sumDisp += delta*delta;
                else sps->sumDisp += params.noDisp;
            } else {
                sps->sumDisp += params.noDisp;
            }
        }
    }
}


// Called in initialization and in re-estimation between layers.
void SPSegmentationEngine::UpdateBoundaryData()
{
    const int directions[2][3] = { { 0, 1, BLeftFlag }, { 1, 0, BTopFlag } };

    // clear neighbors
    for (Superpixel* sp : superpixels) {
        SuperpixelStereo* sps = (SuperpixelStereo*)sp;
        sps->boundaryData.clear();
    }

    // update length & hiSum (written to smoSum)
    for (Pixel& p : pixelsImg) {
        SuperpixelStereo* sp = (SuperpixelStereo*)p.superPixel;

        for (int dir = 0; dir < 2; dir++) {
            Pixel* q = PixelAt(pixelsImg, p.row + directions[dir][0], p.col + directions[dir][1]);

            if (q != nullptr) {
                SuperpixelStereo* sq = (SuperpixelStereo*)q->superPixel;

                if (q->superPixel != sp) {
                    BInfo& bdpq = sp->boundaryData[sq];
                    BInfo& bdqp = sq->boundaryData[sp];
                    double sum;
                    int size;
                    int length;

                    q->CalcHiSmoothnessSumEI(directions[dir][2], depthImg, params.inlierThreshold, sp->plane, sq->plane, sum, size, length);
                    bdpq.hiCount += size;
                    bdpq.hiSum += sum;
                    bdpq.length += length;
                    bdqp.hiCount += size;
                    bdqp.hiSum += sum;
                    bdqp.length += length;
                }
            }
        }
    }

    //#pragma omp parallel
    for (Superpixel* s : superpixels) {
        SuperpixelStereo* sp = (SuperpixelStereo*)s;
        
        for (auto& bdIter : sp->boundaryData) {
            BInfo& bInfo = bdIter.second;
            SuperpixelStereo* sq = bdIter.first;
            double eSmoCoSum;
            int eSmoCoCount;
            double eSmoHiSum = bInfo.hiSum;
            double eSmoOcc = 1; // Phi!?

            CalcCoSmoothnessSum(depthImg, params.inlierThreshold, sp, sq, bInfo.coSum, bInfo.coCount);
            eSmoCoSum = bInfo.coSum;
            eSmoCoCount = bInfo.coCount;

            //double eHi = params.smoWeight*eSmoHiSum / item.second.bSize + params.priorWeight*params.hiPriorWeight;
            //double eCo = params.smoWeight*eSmoCoSum / (sp->GetSize() + sq->GetSize());
            //double eOcc = params.smoWeight*eSmoOcc + params.priorWeight*params.occPriorWeight;
            double eHi = bInfo.hiCount == 0 ? HUGE_VAL : (eSmoHiSum / bInfo.hiCount + params.hiPriorWeight);
            double eCo = eSmoCoCount == 0 ? HUGE_VAL : (eSmoCoSum / eSmoCoCount); //(sp->GetSize() + sq->GetSize()); // + 0
            double eOcc = params.occPriorWeight;

            if (eCo <= eHi && eCo < eOcc) {
                bInfo.type = BTCo;
                bInfo.typePrior = 0;
            } else if (eHi <= eOcc && eHi <= eCo) {
                bInfo.type = BTHi;
                bInfo.typePrior = params.hiPriorWeight;
            } else {
                bInfo.type = BTLo;
                bInfo.typePrior = params.occPriorWeight;
            }
        }
    }

}

// Return true if pixels were actually split.
bool SPSegmentationEngine::SplitPixels()
{
    int imgPixelsRows = 0;
    int imgPixelsCols = 0;
    int maxPixelSize = 1;

    for (int i = 0; i < pixelsImg.rows; i++) {
        int rSize = pixelsImg(i, 0).GetRSize();
     
        imgPixelsRows += (rSize == 1) ? 1 : 2;
        if (rSize > maxPixelSize) maxPixelSize = rSize;
    }
    for (int j = 0; j < pixelsImg.cols; j++) {
        int cSize = pixelsImg(0, j).GetCSize();

        imgPixelsCols += (cSize == 1) ? 1 : 2;
        if (cSize > maxPixelSize) maxPixelSize = cSize;
    }

    if (maxPixelSize == 1) 
        return false;

    Matrix<Pixel> newPixelsImg(imgPixelsRows, imgPixelsCols);

    if (params.stereo) {
        for (Superpixel*& sp : superpixels) {
            ((SuperpixelStereo*)sp)->ClearPixelSet();
        }
    }

    int newRow = 0;

    for (int i = 0; i < pixelsImg.rows; i++) {
        int newCol = 0;
        int pRowSize = pixelsImg(i, 0).GetRSize();

        for (int j = 0; j < pixelsImg.cols; j++) {
            Pixel& p = pixelsImg(i, j);
            int pColSize = p.GetCSize();

            if (pRowSize == 1 && pColSize == 1) {
                Pixel& p11 = newPixelsImg(newRow, newCol);

                p.CopyTo(img, newRow, newCol, p11);
            } else if (pColSize == 1) { // split only row
                Pixel& p11 = newPixelsImg(newRow, newCol);
                Pixel& p21 = newPixelsImg(newRow + 1, newCol);

                p.SplitRow(img, newRow, newRow + 1, newCol, p11, p21);
            } else if (pRowSize == 1) { // split only column
                Pixel& p11 = newPixelsImg(newRow, newCol);
                Pixel& p12 = newPixelsImg(newRow, newCol + 1);

                p.SplitColumn(img, newRow, newCol, newCol + 1, p11, p12);
            } else { // split row and column
                Pixel& p11 = newPixelsImg(newRow, newCol);
                Pixel& p12 = newPixelsImg(newRow, newCol + 1);
                Pixel& p21 = newPixelsImg(newRow + 1, newCol);
                Pixel& p22 = newPixelsImg(newRow + 1, newCol + 1);

                p.Split(img, newRow, newRow + 1, newCol, newCol + 1, p11, p12, p21, p22);
            }
            newCol += (pColSize > 1) ? 2 : 1;
        }
        newRow += (pRowSize > 1) ? 2 : 1;
    }
    pixelsImg = newPixelsImg;

    for (Superpixel* sp : superpixels) {
        sp->RecalculateEnergies();
    }
    if (params.stereo) {
        for (Pixel& p : pixelsImg) {
            ((SuperpixelStereo*)p.superPixel)->AddToPixelSet(&p);
        }
        UpdatePPImage();
    }
    return true;
}

// Lock function for non-stereo iterations
// These are neighbors of p
void LockNonStereo(Pixel*& p, unordered_set<Superpixel*>& toLock, const Matrix<Pixel>& pixelsImg)
{
    for (int m = 0; m < 5; m++) {
        const Pixel* q = PixelAt(pixelsImg, p->row + nDeltas0[m][0], p->col + nDeltas0[m][1]);
        if (q != nullptr) 
            toLock.insert(q->superPixel);
    }
}

// Lock function for stereo iterations
void LockStereo(Pixel*& p, unordered_set<Superpixel*>& toLock, const Matrix<Pixel>& pixelsImg)
{
    for (int m = 0; m < 5; m++) {
        const Pixel* q = PixelAt(pixelsImg, p->row + nDeltas0[m][0], p->col + nDeltas0[m][1]);
        if (q != nullptr) {
            SuperpixelStereo* sps = (SuperpixelStereo*)q->superPixel;
            for (auto& bdIter : sps->boundaryData) {
                toLock.insert(bdIter.first);
            }
        }
    }
}

static int dbgImageNum = 0;

void SPSegmentationEngine::IterateInThread(ParallelDeque<Pixel*, Superpixel*>* pList)
{
    PixelMoveData tryMoveData[4];
    Superpixel* nbsp[5];
    int nbspSize;

    while (!pList->Empty() && pList->GetPopCount() < params.maxUpdates) {
        Pixel* p = pList->PopAndLock();

        if (p == nullptr) 
            continue;
        
        nbsp[0] = p->superPixel;
        nbspSize = 1;
        for (int m = 0; m < 4; m++) {
            Pixel* q = PixelAt(pixelsImg, p->row + nDeltas[m][0], p->col + nDeltas[m][1]);

            if (q == nullptr) tryMoveData[m].allowed = false;
            else {
                bool newNeighbor = true;

                for (int i = 0; i < nbspSize; i++) {
                    if (q->superPixel == nbsp[i]) {
                        newNeighbor = false;
                        break;
                    }
                }
                if (!newNeighbor) tryMoveData[m].allowed = false;
                else {
                    if (params.stereo) TryMovePixelStereo(p, q, tryMoveData[m]);
                    else TryMovePixel(p, q, tryMoveData[m]);
                    nbsp[nbspSize++] = q->superPixel;
                }
            }
        }

        PixelMoveData* bestMoveData = FindBestMoveData(params, tryMoveData);

        if (bestMoveData != nullptr) {
            if (params.stereo) {
                //SuperpixelStereo* sps = (SuperpixelStereo*)(bestMoveData->p->superPixel);
                //double calc = sps->CalcDispEnergy(depthImg, params.inlierThreshold, params.noDisp);
                //if (fabs(calc - sps->GetDispSum()) > 0.01) {
                //    cout << "Disp sum mismatch";
                //}
                //sps->CheckRegEnergy();
                //sps->CheckAppEnergy(img);

                double delta = TotalEnergyDelta(params, bestMoveData);

                //if (performanceInfo.levelMaxEDelta[level] < delta)
                //    performanceInfo.levelMaxEDelta[level] = delta;

                MovePixelStereo(pixelsImg, *bestMoveData);

                //char fname[1000];
                //sprintf(fname, "c:\\tmp\\dbgbound-%03d.png", dbgImageNum++);
                //imwrite(fname, GetSegmentedImageStereo());
                //DebugBoundary();
                //DebugDispSums();
                //DebugNeighborhoods();

                //if (++dbgImageNum < 500) {
                //    char fname[1000];
                //    sprintf(fname, "c:\\tmp\\dbgbound-%03d.png", dbgImageNum);
                //    //if (dbgImageNum >= 160) 
                //        imwrite(fname, GetSegmentedImageStereo());
                //}

            } else {
                MovePixel(pixelsImg, *bestMoveData);
            }

            pList->PushBack(p);
            for (int m = 0; m < 4; m++) {
                Pixel* qq = PixelAt(pixelsImg, p->row + nDeltas[m][0], p->col + nDeltas[m][1]);
                if (qq != nullptr && p->superPixel != qq->superPixel)
                    pList->PushBack(qq);
            }
        }

        pList->Release(p);
    }
}

// Returns number of iterations
//int SPSegmentationEngine::IterateMoves(int level)
//{
//    ParallelDeque<Pixel*, Superpixel*> list(pixelsImg.rows * pixelsImg.cols);
//    int itemCount = 1;
//
//    list.SetEmptyVal(nullptr);
//    if (params.stereo) {
//        list.SetLockFunction([&](const Pixel*& p, unordered_set<const Superpixel*>& forb) { LockNonStereo(p, forb, pixelsImg); });
//    } else {
//        list.SetLockFunction([&](const Pixel*& p, unordered_set<const Superpixel*>& forb) { LockStereo(p, forb, pixelsImg); });
//    }
//
//    // Initialize pixel (block) border list 
//    for (Pixel& p : pixelsImg) {
//        Pixel* q;
//
//        for (int m = 0; m < 4; m++) {
//            q = PixelAt(pixelsImg, p.row + nDeltas[m][0], p.col + nDeltas[m][1]);
//            if (q != nullptr && p.superPixel != q->superPixel) {
//                list.PushBack(&p);
//                break;
//            }
//        }
//    }
//
//    int count = 0;
//    PixelMoveData tryMoveData[4];
//    Superpixel* nbsp[5];
//    int nbspSize;
//
//    if (performanceInfo.levelMaxEDelta.size() <= level) {
//        performanceInfo.levelMaxEDelta.resize(level + 1);
//        performanceInfo.levelMaxEDelta[level] = 0;
//    }
//
//    while (!list.Empty() && count < params.maxUpdates) {
//        Pixel* p = list.PopAndLock();
//
//        nbsp[0] = p->superPixel;
//        nbspSize = 1;
//        for (int m = 0; m < 4; m++) {
//            Pixel* q = PixelAt(pixelsImg, p->row + nDeltas[m][0], p->col + nDeltas[m][1]);
//
//            if (q == nullptr) tryMoveData[m].allowed = false;
//            else {
//                bool newNeighbor = true;
//
//                for (int i = 0; i < nbspSize; i++) {
//                    if (q->superPixel == nbsp[i]) {
//                        newNeighbor = false;
//                        break;
//                    }
//                }
//                if (!newNeighbor) tryMoveData[m].allowed = false;
//                else {
//                    if (params.stereo) TryMovePixelStereo(p, q, tryMoveData[m]); 
//                    else TryMovePixel(p, q, tryMoveData[m]);
//                    nbsp[nbspSize++] = q->superPixel;
//                }
//            }
//        }
//
//        PixelMoveData* bestMoveData = FindBestMoveData(params, tryMoveData);
//
//        if (bestMoveData != nullptr) {
//            if (params.stereo) {
//                //SuperpixelStereo* sps = (SuperpixelStereo*)(bestMoveData->p->superPixel);
//                //double calc = sps->CalcDispEnergy(depthImg, params.inlierThreshold, params.noDisp);
//                //if (fabs(calc - sps->GetDispSum()) > 0.01) {
//                //    cout << "Disp sum mismatch";
//                //}
//                //sps->CheckRegEnergy();
//                //sps->CheckAppEnergy(img);
//
//                double delta = TotalEnergyDelta(params, bestMoveData);
//
//                if (performanceInfo.levelMaxEDelta[level] < delta)
//                    performanceInfo.levelMaxEDelta[level] = delta;
//
//                MovePixelStereo(pixelsImg, *bestMoveData);
//                //DebugBoundary();
//                //DebugDispSums();
//                //DebugNeighborhoods();
//
//            } else {
//                MovePixel(pixelsImg, *bestMoveData);
//            }
//
//            list.PushBack(p);
//            for (int m = 0; m < 4; m++) {
//                Pixel* qq = PixelAt(pixelsImg, p->row + nDeltas[m][0], p->col + nDeltas[m][1]);
//                if (qq != nullptr && p->superPixel != qq->superPixel)
//                    list.PushBack(qq);
//            }
//        }
//
//        list.PopFront();
//        count++;
//    }
//    return count;
//}

int SPSegmentationEngine::IterateMoves(int level)
{
    params.SetLevelParams(level);
    cout << "appWeight: " << params.appWeight << endl;

    ParallelDeque<Pixel*, Superpixel*> list(pixelsImg.rows * pixelsImg.cols);
    int itemCount = 1;

    list.SetEmptyVal(nullptr);
    if (params.stereo) {
        list.SetLockFunction([&](Pixel*& p, unordered_set<Superpixel*>& forb) { LockStereo(p, forb, pixelsImg); });
    } else {
        list.SetLockFunction([&](Pixel*& p, unordered_set<Superpixel*>& forb) { LockNonStereo(p, forb, pixelsImg); });
    }

    // Initialize pixel (block) border list 
    for (Pixel& p : pixelsImg) {
        Pixel* q;

        for (int m = 0; m < 4; m++) {
            q = PixelAt(pixelsImg, p.row + nDeltas[m][0], p.col + nDeltas[m][1]);
            if (q != nullptr && p.superPixel != q->superPixel) {
                list.PushBack(&p);
                break;
            }
        }
    }

    thread threads[1];

    for (int ti = 0; ti < 1; ti++) {
        threads[ti] = thread([&](ParallelDeque<Pixel*, Superpixel*>* pList) { IterateInThread(pList); }, &list);
    }
    for (auto& t : threads) { t.join(); }

    return list.GetPopCount();
}

Mat SPSegmentationEngine::GetSegmentedImage()
{
    if (params.stereo) return GetSegmentedImageStereo();
    else return GetSegmentedImagePlain();
}

Mat SPSegmentationEngine::GetSegmentedImagePlain()
{
    Mat result = origImg.clone();
    Vec3b blackPixel(0, 0, 0);

    for (Pixel& p : pixelsImg) {
        if (p.BLeft()) {
            for (int r = p.ulr; r < p.lrr; r++) {
                result.at<Vec3b>(r, p.ulc) = blackPixel;
            }
        }
        if (p.BRight()) {
            for (int r = p.ulr; r < p.lrr; r++) {
                result.at<Vec3b>(r, p.lrc - 1) = blackPixel;
            }
        }
        if (p.BTop()) {
            for (int c = p.ulc; c < p.lrc; c++) {
                result.at<Vec3b>(p.ulr, c) = blackPixel;
            }
        }
        if (p.BBottom()) {
            for (int c = p.ulc; c < p.lrc; c++) {
                result.at<Vec3b>(p.lrr - 1, c) = blackPixel;
            }
        }
    }
    return result;
}

const Vec3b& BoundaryColor(Pixel* p, Pixel* q)
{
    static const Vec3b pixelColors[] = { Vec3b(0, 0, 0), Vec3b(0, 255, 0), Vec3b(255, 0, 0), Vec3b(0, 0, 196), Vec3b(0, 0, 196) };

    if (p == nullptr || q == nullptr) 
        return pixelColors[0];

    SuperpixelStereo* sp = (SuperpixelStereo*)p->superPixel;
    SuperpixelStereo* sq = (SuperpixelStereo*)q->superPixel;

    if (sp == nullptr || sq == nullptr) 
        return pixelColors[0];

    auto bdIter = sp->boundaryData.find(sq);
    if (bdIter == sp->boundaryData.end()) 
        return pixelColors[0];
    else {
        if (bdIter->second.type > 0 || bdIter->second.type < 5) return pixelColors[bdIter->second.type];
        else return pixelColors[0];
    }
}

Mat SPSegmentationEngine::GetSegmentedImageStereo()
{ 
    if (!params.stereo) return GetSegmentedImagePlain();

    Mat result = origImg.clone();

    for (Pixel& p : pixelsImg) {
        if (p.BLeft()) {
            Pixel* q = PixelAt(pixelsImg, p.row, p.col - 1);
            const Vec3b& color = BoundaryColor(&p, q);

            for (int r = p.ulr; r < p.lrr; r++) {
                result.at<Vec3b>(r, p.ulc) = color;
            }
        }
        if (p.BRight()) {
            Pixel* q = PixelAt(pixelsImg, p.row, p.col + 1);
            const Vec3b& color = BoundaryColor(&p, q);

            for (int r = p.ulr; r < p.lrr; r++) {
                result.at<Vec3b>(r, p.lrc - 1) = color;
            }
        }
        if (p.BTop()) {
            Pixel* q = PixelAt(pixelsImg, p.row - 1, p.col);

            const Vec3b& color = BoundaryColor(&p, q);
            for (int c = p.ulc; c < p.lrc; c++) {
                result.at<Vec3b>(p.ulr, c) = color;
            }
        }
        if (p.BBottom()) {
            Pixel* q = PixelAt(pixelsImg, p.row + 1, p.col);
            const Vec3b& color = BoundaryColor(&p, q);

            for (int c = p.ulc; c < p.lrc; c++) {
                result.at<Vec3b>(p.lrr - 1, c) = color;
            }
        }
    }
    return result;

}

Mat SPSegmentationEngine::GetDisparity() const
{
    Mat_<unsigned short> result = Mat_<unsigned short>(ppImg.rows, ppImg.cols);

    for (int i = 0; i < ppImg.rows; i++) {
        for (int j = 0; j < ppImg.cols; j++) {
            SuperpixelStereo* sps = (SuperpixelStereo*)ppImg(i, j)->superPixel;
            double val = DotProduct(sps->plane, i, j, 1.0);
            result(i, j) = val < 256.0 ? (val < 0 ? 0 : val * 256.0) : 65535;
        }
    }
    return result;
}


Mat SPSegmentationEngine::GetSegmentation() const
{
    Mat result = Mat_<unsigned short>(pixelsImg.rows, pixelsImg.cols);
    unordered_map<Superpixel*, int> indexMap;
    int maxIndex = 0;

    for (const Pixel& p : pixelsImg) {
        if (indexMap.find(p.superPixel) == indexMap.end()) {
            indexMap[p.superPixel] = maxIndex++;
        }
    }
    for (const Pixel& p : pixelsImg) {
        result.at<unsigned short>(p.row, p.col) = indexMap[p.superPixel];
    }
    return result;
}

string SPSegmentationEngine::GetSegmentedImageInfo()
{
    map<Superpixel*, vector<Pixel*>> spMap;
    stringstream ss;

    for (Pixel& p : pixelsImg) {
        Superpixel* sp = p.superPixel;
        spMap[sp].push_back(&p);
    }
    ss << '{';
    bool firstSp = true;
    for (auto mPair : spMap) {
        if (firstSp) firstSp = false; else ss << ',';
        ss << '{';
        ss << mPair.first->GetAppEnergy();
        ss << ',';
        ss << mPair.first->GetRegEnergy();
        ss << ',';
        ss << mPair.first->GetSize();
        ss << ',';
        ss << mPair.first->GetBorderLength();
        ss << ',';

        double mr, mc;
        mPair.first->GetMean(mr, mc);
        ss << '{' << mr << ',' << mc << '}';

        ss << ",";
        ss << fixed << mPair.first->GetRegEnergy();

        ss << ',' << '{';

        bool firstP = true;
        for (Pixel* p : mPair.second) {
            if (firstP) firstP = false; else ss << ',';
            ss << p->GetPixelsAsString();
        }
        ss << '}' << '}';
    }
    ss << '}';
    return ss.str();
}

void SPSegmentationEngine::PrintDebugInfo()
{
    double appESum = 0.0;
    double regESum = 0.0;
    double dispESum = 0.0;

    for (Superpixel* sp : superpixels) {
        appESum += sp->GetAppEnergy();
        regESum += sp->GetRegEnergy();
    }
    cout << "Reg energy mean: " << regESum / superpixels.size() << endl;
    cout << "Disp energy mean: " << dispESum / superpixels.size() << endl;
}


void SPSegmentationEngine::PrintDebugInfoStereo()
{
    StatData stat;

    MeanAndVariance(superpixels.begin(), superpixels.end(),
        [](Superpixel* sp) { return ((SuperpixelStereo*)sp)->GetAppEnergy(); },
        stat);
    cout << "App energy mean: " << stat.mean << ", variance: " << stat.var << ", min: " << stat.min << ", max: " << stat.max << endl;

    MeanAndVariance(superpixels.begin(), superpixels.end(),
        [](Superpixel* sp) { return ((SuperpixelStereo*)sp)->GetRegEnergy(); },
        stat);
    cout << "Reg energy mean: " << stat.mean << ", variance: " << stat.var << ", min: " << stat.min << ", max: " << stat.max << endl;

    MeanAndVariance(superpixels.begin(), superpixels.end(),
        [](Superpixel* sp) { return ((SuperpixelStereo*)sp)->GetBorderLength(); },
        stat);
    cout << "Border length mean: " << stat.mean << ", variance: " << stat.var << ", min: " << stat.min << ", max: " << stat.max << endl;

    MeanAndVariance(superpixels.begin(), superpixels.end(),
        [](Superpixel* sp) { return ((SuperpixelStereo*)sp)->GetDispSum(); },
        stat);
    cout << "Disp energy mean: " << stat.mean << ", variance: " << stat.var << ", min: " << stat.min << ", max: " << stat.max << endl;

    MeanAndVariance(superpixels.begin(), superpixels.end(),
        [](Superpixel* sp) { return ((SuperpixelStereo*)sp)->GetSmoEnergy(); },
        stat);
    cout << "Smo energy mean: " << stat.mean << ", variance: " << stat.var << ", min: " << stat.min << ", max: " << stat.max << endl;
}

int SPSegmentationEngine::GetNoOfSuperpixels() const
{
    return (int)superpixels.size();
}

void SPSegmentationEngine::PrintPerformanceInfo()
{
    cout << "Initialization time: " << performanceInfo.init << " sec." << endl;
    cout << "Ransac time: " << performanceInfo.ransac << " sec." << endl;
    cout << "Time of image processing: " << performanceInfo.imgproc << " sec." << endl;
    cout << "Total time: " << performanceInfo.total << " sec." << endl;
    cout << "Times for each level (in sec.): ";
    for (double& t : performanceInfo.levelTimes)
        cout << t << ' ';
    cout << endl;
    cout << "Max energy deltaa for each level: ";
    for (double& t : performanceInfo.levelMaxEDelta)
        cout << t << ' ';
    cout << endl;
    for (int& c : performanceInfo.levelIterations)
        cout << c << ' ';
    cout << endl;

    int minBDSize = INT_MAX;
    int maxBDSize = 0;

    if (params.stereo) {
        for (Superpixel* sp : superpixels) {
            SuperpixelStereo* sps = (SuperpixelStereo*)sp;
            if (minBDSize > sps->boundaryData.size())
                minBDSize = sps->boundaryData.size();
            if (maxBDSize < sps->boundaryData.size())
                maxBDSize = sps->boundaryData.size();
        }
        cout << "Max boundary size: " << maxBDSize << endl;
        cout << "Min boundary size: " << minBDSize << endl;
    }
}

/*
void SPSegmentationEngine::UpdateInliers()
{
    for (Superpixel* sp : superpixels) {
        SuperpixelStereo* sps = (SuperpixelStereo*)sp;

        sps->sumIRow = 0; sps->sumICol = 0;         // Sum of terms computed for inlier points
        sps->sumIRow2 = 0; sps->sumICol2 = 0;
        sps->sumIRowCol = 0;
        sps->sumIRowD = 0.0, sps->sumIColD = 0.0;
        sps->sumID = 0.0;
        sps->nI = 0;
    }
    for (int i = 0; i < ppImg.rows; i++) {
        for (int j = 0; j < ppImg.cols; j++) {
            Pixel* p = ppImg(i, j);
            SuperpixelStereo* sps = (SuperpixelStereo*)p->superPixel;
            const double& disp = depthImg(i, j);

            if (disp > 0) {
                bool inlier = fabs(DotProduct(sps->plane, i, j, 1.0) - disp) < params.inlierThreshold;

                if (inlier) {
                    sps->sumIRow += i; sps->sumIRow2 += i*i;
                    sps->sumICol += j; sps->sumICol2 += j*j;
                    sps->sumIRowCol += i*j;
                    sps->sumIRowD += i*disp; sps->sumIColD += j*disp;
                    sps->sumID += disp;
                    sps->nI++;
                }
            }
        }
    }
}
*/

// Try to move Pixel p to Superpixel containing Pixel q with coordinates (qRow, qCol)
// Note: pixel q is must be neighbor of p and p->superPixel != q->superPixel
// Fills psd, returns psd.allowed
// Note: energy deltas in psd are "energy_before - energy_after"
bool SPSegmentationEngine::TryMovePixel(Pixel* p, Pixel* q, PixelMoveData& psd)
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

bool SPSegmentationEngine::TryMovePixelStereo(Pixel* p, Pixel* q, PixelMoveData& psd)
{
    SuperpixelStereo* sp = (SuperpixelStereo*)p->superPixel;
    SuperpixelStereo* sq = (SuperpixelStereo*)q->superPixel;

    if (sp == sq || !IsSuperpixelRegionConnectedOptimized(pixelsImg, p, p->row - 1, p->col - 1, p->row + 2, p->col + 2)) {
        psd.allowed = false;
        return false;
    }

    double pSize = p->GetSize(), qSize = q->GetSize();
    int spSize = sp->GetSize(), sqSize = sq->GetSize();
    double spEApp = sp->GetAppEnergy(), sqEApp = sq->GetAppEnergy();
    double spEReg = sp->GetRegEnergy(), sqEReg = sq->GetRegEnergy();
    double spEDisp = sp->GetDispSum(), sqEDisp = sq->GetDispSum();
    double spESmo = sp->GetSmoEnergy(), sqESmo = sq->GetSmoEnergy();
    double spEPrior = sp->GetPriorEnergy(), sqEPrior = sq->GetPriorEnergy();

    PixelChangeDataStereo pcd;
    PixelChangeDataStereo qcd;
    PixelData pd;
    int spbl, sqbl, sobl;

    p->CalcPixelDataStereo(img, depthImg, sp->plane, sq->plane, params.inlierThreshold, params.noDisp, pd);
    sp->GetRemovePixelDataStereo(pd, pcd);
    sq->GetAddPixelDataStereo(pd, qcd);
    CalcBorderChangeDataStereo(pixelsImg, depthImg, params, p, q, psd.bDataP, psd.bDataQ, psd.prem);
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
    psd.eSmoDelta = spESmo + sqESmo - GetSmoEnergy(psd.bDataP, sp, psd.pSize, sq, psd.qSize) -
        GetSmoEnergy(psd.bDataQ, sq, psd.qSize, sp, psd.pSize);
    psd.allowed = true;
    psd.pixelData = pd;
    //psd.bDataP =
    //psd.bDataQ =
    return true;
}

void SPSegmentationEngine::DebugNeighborhoods()
{
    const int directions[2][3] = { { 0, 1, BLeftFlag }, { 1, 0, BTopFlag } };

    // update length & hiSum (written to smoSum)
    for (Pixel& p : pixelsImg) {
        SuperpixelStereo* sp = (SuperpixelStereo*)p.superPixel;

        for (int dir = 0; dir < 2; dir++) {
            Pixel* q = PixelAt(pixelsImg, p.row + directions[dir][0], p.col + directions[dir][1]);

            if (q != nullptr) {
                SuperpixelStereo* sq = (SuperpixelStereo*)q->superPixel;

                if (sp->id != sq->id) {
                    if (sp->boundaryData.find(sq) == sp->boundaryData.end())
                        throw Exception();
                    if (sq->boundaryData.find(sp) == sq->boundaryData.end())
                        throw Exception();
                }
            }
        }
    }

}

void SPSegmentationEngine::DebugBoundary()
{
    const int directions[2][3] = { { 0, 1, BLeftFlag }, { 1, 0, BTopFlag } };

    // unordered map is not used; pair does not have hash function, but performance is not important....
    map<pair<SuperpixelStereo*, SuperpixelStereo*>, BInfo> newBoundaryData;

    // update length & hiSum (written to smoSum)
    for (Pixel p : pixelsImg) {
        SuperpixelStereo* sp = (SuperpixelStereo*)p.superPixel;

        for (int dir = 0; dir < 2; dir++) {
            Pixel* q = PixelAt(pixelsImg, p.row + directions[dir][0], p.col + directions[dir][1]);

            if (q != nullptr) {
                SuperpixelStereo* sq = (SuperpixelStereo*)q->superPixel;

                if (q->superPixel != sp) {
                    BInfo& bdpq = newBoundaryData[pair<SuperpixelStereo*, SuperpixelStereo*>(sp, sq)];
                    BInfo& bdqp = newBoundaryData[pair<SuperpixelStereo*, SuperpixelStereo*>(sq, sp)];
                    double sum;
                    int size;
                    int length;

                    q->CalcHiSmoothnessSumEI(directions[dir][2], depthImg, params.inlierThreshold, sp->plane, sq->plane, sum, size, length);
                    bdpq.hiCount += size;
                    bdpq.hiSum += sum;
                    bdpq.length += length;
                    bdqp.hiCount += size;
                    bdqp.hiSum += sum;
                    bdqp.length += length;
                }
            }
        }
    }


    for (auto& bdInfoIter : newBoundaryData) {
        SuperpixelStereo* sp = bdInfoIter.first.first;
        SuperpixelStereo* sq = bdInfoIter.first.second;
        BInfo& bInfo = bdInfoIter.second;
        double eSmoCoSum;
        int eSmoCoCount;
        double eSmoHiSum = bInfo.hiSum;
        double eSmoOcc = 1; // Phi!?

        CalcCoSmoothnessSum(depthImg, params.inlierThreshold, sp, sq, bInfo.coSum, bInfo.coCount);
        eSmoCoSum = bInfo.coSum;
        eSmoCoCount = bInfo.coCount;

        //double eHi = params.smoWeight*eSmoHiSum / item.second.bSize + params.priorWeight*params.hiPriorWeight;
        //double eCo = params.smoWeight*eSmoCoSum / (sp->GetSize() + sq->GetSize());
        //double eOcc = params.smoWeight*eSmoOcc + params.priorWeight*params.occPriorWeight;
        double eHi = bInfo.hiCount == 0 ? HUGE_VAL : (eSmoHiSum / bInfo.hiCount + params.hiPriorWeight);
        double eCo = eSmoCoCount == 0 ? HUGE_VAL : (eSmoCoSum / eSmoCoCount); //(sp->GetSize() + sq->GetSize()); // + 0
        double eOcc = params.occPriorWeight;

        if (eCo <= eHi && eCo < eOcc) {
            bInfo.type = BTCo;
            bInfo.typePrior = 0;
        } else if (eHi <= eOcc && eHi <= eCo) {
            bInfo.type = BTHi;
            bInfo.typePrior = params.hiPriorWeight;
        } else {
            bInfo.type = BTLo;
            bInfo.typePrior = params.occPriorWeight;
        }
    }

    for (Superpixel* s : superpixels) {
        SuperpixelStereo* sp = (SuperpixelStereo*)s;

        for (auto& bdIter : sp->boundaryData) {
            BInfo& bInfo = bdIter.second;
            SuperpixelStereo* sq = bdIter.first;

            auto nnbIter = newBoundaryData.find(pair<SuperpixelStereo*, SuperpixelStereo*>(sp, sq));
            if (nnbIter == newBoundaryData.end()) throw exception("Can not find boundary");
            else { // nnbiter-> re-calculated // bInfo -> currently in 
                if (nnbIter->second.type != bInfo.type) {
                    throw exception("type mismatch");
                }
                if (bInfo.type == BTCo) {
                    if (nnbIter->second.coCount != bInfo.coCount) {
                        throw exception("energy mismatch -- coCount");
                    }
                    if (fabs(nnbIter->second.coSum - bInfo.coSum) > 0.01) {
                        throw exception("energy mismatch -- coSum");
                    }
                }
                if (bInfo.type == BTHi) {
                    if (fabs(nnbIter->second.hiSum - bInfo.hiSum) > 0.01) {
                        throw exception("energy mismatch -- hiSum");
                    }
                    if (nnbIter->second.hiCount != bInfo.hiCount) {
                        throw exception("energy mismatch -- hiCount");
                    }
                }
            }
        }
    }

}

void SPSegmentationEngine::DebugDispSums()
{
    for (Superpixel* sp : superpixels) {
        SuperpixelStereo* sps = (SuperpixelStereo*)sp;
        double disp = sps->CalcDispEnergy(depthImg, params.inlierThreshold, params.noDisp);
        if (fabs(sps->sumDisp - disp) > 0.01) {
            throw exception("disp sum mismatch");
        }
    }
}


// Functions
///////////////////////////////////////////////////////////////////////////////


SPSegmentationParameters ReadParameters(const string& fileName, const SPSegmentationParameters& defaultValue)
{
    try {
        FileStorage fs(fileName, FileStorage::READ);
        SPSegmentationParameters sp;

        fs.root() >> sp;
        return sp;
    } catch (exception& e) {
        cerr << e.what() << endl;
        return defaultValue;
    }
}


