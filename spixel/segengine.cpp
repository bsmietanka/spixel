#include "stdafx.h"
#include "segengine.h"
#include "functions.h"
#include "utils.h"
#include <unordered_map>

// Local functions
///////////////////////////////////////////////////////////////////////////////

double TotalEnergyDelta(const SPSegmentationParameters& params, PixelMoveData* pmd) 
{ 
    double eSum = 0.0;
    int initPSizeHalf = pmd->p->superPixel->GetInitialSize()/2;
    
    eSum += params.regWeight * pmd->eRegDelta;
    eSum += params.appWeight * pmd->eAppDelta;
    eSum += params.lenWeight * pmd->bLenDelta;
    if (params.stereo) {
        eSum += params.dispWeight * pmd->eDispDelta;
        eSum += params.priorWeight * pmd->ePriorDelta;
        eSum += params.smoWeight * pmd->eSmoDelta;
    }

    if (pmd->pSize < initPSizeHalf)  
        eSum -= params.sizeWeight * (initPSizeHalf - pmd->pSize);
    return eSum;
}


PixelMoveData* FindBestMoveData(const SPSegmentationParameters& params, PixelMoveData d[4])
{
    PixelMoveData* dArray[4];
    int dArraySize = 0;

    // Only allowed moves and energy delta should be positive
    for (int m = 0; m < 4; m++) {
        PixelMoveData* md = &d[m];
        if (md->allowed && TotalEnergyDelta(params, md) > 0) dArray[dArraySize++] = md;
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

// SPSegmentationParameters
///////////////////////////////////////////////////////////////////////////////

static void read(const FileNode& node, SPSegmentationParameters& x, const SPSegmentationParameters& defaultValue)
{
    if (node.empty()) x = defaultValue;
    else x.read(node);
}


// SPSegmentationEngine
///////////////////////////////////////////////////////////////////////////////

SPSegmentationEngine::SPSegmentationEngine(SPSegmentationParameters params, Mat im, Mat depthIm) :
    params(params), origImg(im)
{
    img = ConvertRGBToLab(im);
    depthImg = AdjustDisparityImage(depthIm);
    inliers = Mat1b(depthImg.rows, depthImg.cols);
}

SPSegmentationEngine::~SPSegmentationEngine()
{
    Reset();
}

void SPSegmentationEngine::Initialize(Superpixel* spGenerator())
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

    superpixels.clear();
    superpixels.reserve(imgSPixelsCols*imgSPixelsRows);
    for (int pi = 0; pi < imgSPixelsRows; pi++) {
        for (int pj = 0; pj < imgSPixelsCols; pj++) {
            int i0 = pi*sPixelSize;
            int j0 = pj*sPixelSize;
            int i1 = min(i0 + sPixelSize, pixelsImg.rows);
            int j1 = min(j0 + sPixelSize, pixelsImg.cols);
            Superpixel* sp = spGenerator(); // Superpixel();

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

struct BoundaryData {
    int bSize;      // Boundary size
    double hiSum;
};

typedef pair<SuperpixelStereo*, SuperpixelStereo*> SPSPair;

SPSPair OrderedPair(SuperpixelStereo* sp, SuperpixelStereo* sq)
{
    return sp < sq ? SPSPair(sp, sq) : SPSPair(sq, sp);
}

void SPSegmentationEngine::InitializeStereo()
{
    Initialize([]() -> Superpixel* { return new SuperpixelStereo(); });
    InitializePPImage();
    EstimatePlaneParameters();
}

void SPSegmentationEngine::InitializePPImage()
{
    ppImg = Matrix<Pixel*>(img.rows, img.cols);
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

    Initialize([]() { return new Superpixel(); });

    t0.Stop();
    performanceInfo.init = t0.GetTimeInSec();
    t0.Resume();

    Timer t1;

    do {
        Timer t2;
        for (int iteration = 0; iteration < params.iterations; iteration++) {
            IterateMoves();
        }
        SplitPixels();
        t2.Stop();
        performanceInfo.levels.push_back(t2.GetTimeInSec());
    } while (PixelSize() > 1);

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

    do {
        Timer t2;
        for (int iteration = 0; iteration < params.iterations; iteration++) {
            IterateMoves();
            ReEstimatePlaneParameters();
        }
        SplitPixels();
        t2.Stop();
        performanceInfo.levels.push_back(t2.GetTimeInSec());
    } while (PixelSize() > 1);

    t0.Stop();
    t1.Stop();
    performanceInfo.total = t0.GetTimeInSec();
    performanceInfo.imgproc = t1.GetTimeInSec();
}


// Get size of the first pixel (this one has probably standard dimension)
int SPSegmentationEngine::PixelSize()
{
    return pixelsImg(0).GetCSize();
}

void SPSegmentationEngine::ReEstimatePlaneParameters()
{
    //#pragma omp parallel for
    for (int i = 0; i < superpixels.size(); i++) {
        SuperpixelStereo* sp = (SuperpixelStereo*)superpixels[i];
        sp->CalcPlaneLeastSquares(depthImg, inliers);
    }
    SPSegmentationEngine::UpdateInliers();
}

void SPSegmentationEngine::EstimatePlaneParameters()
{
    const int directions[2][3] = { { 0, -1, BLeftFlag }, { -1, 0, BTopFlag } };

    vector<cv::Point3d> pixels3d;

    Timer t;

    #pragma omp parallel for
    for (int i = 0; i < superpixels.size(); i++) {
        SuperpixelStereo* sp = (SuperpixelStereo*)superpixels[i];
        UpdateSuperpixelPlaneRANSAC(sp, depthImg, params.dispBeta);
    }
    SPSegmentationEngine::UpdateInliers();
    ReEstimatePlaneParameters();

    t.Stop();
    performanceInfo.ransac += t.GetTimeInSec();

    // Update boundary data
    map<SPSPair, BoundaryData> boundaryMap;

    // update map
    for (Pixel& p : pixelsImg) {
        SuperpixelStereo* sp = (SuperpixelStereo*)p.superPixel;
        Pixel* q;
        SuperpixelStereo* sq;

        for (int dir = 0; dir < 2; dir++) {
            q = PixelAt(pixelsImg, p.row + directions[dir][0], p.col + directions[dir][1]);

            if (q != nullptr) {
                sq = (SuperpixelStereo*)q->superPixel;

                if (q->superPixel != sp) {
                    BoundaryData& bd = boundaryMap[OrderedPair(sp, sq)];
                    double sum;
                    int size;

                    p.CalcHiSmoothnessSum(directions[dir][2], inliers, sp->plane, sq->plane, sum, size);
                    bd.bSize += size;
                    bd.hiSum += sum;
                }
            }
        }
    }

    for (auto& item : boundaryMap) {
        SuperpixelStereo* sp = item.first.first;
        SuperpixelStereo* sq = item.first.second;

        double eSmoCoSum = CalcCoSmoothnessSum(inliers, sp, sq);
        double eSmoHiSum = item.second.hiSum;
        double eSmoOcc = 1; // Phi!?

        double eHi = params.smoWeight*eSmoHiSum / item.second.bSize + params.priorWeight*params.hiPriorWeight;
        double eCo = params.smoWeight*eSmoCoSum / (sp->GetSize() + sq->GetSize());
        double eOcc = params.smoWeight*eSmoOcc + params.priorWeight*params.occPriorWeight;

        auto& bd = sp->boundaryData[sq];

        if (eCo <= eHi && eCo < eOcc) {
            bd.length = sp->GetSize() + sq->GetSize();
            bd.type = BTCo;
            bd.smo = eSmoCoSum;
            bd.prior = 0;
        } else if (eHi <= eOcc && eHi <= eCo) {
            bd.length = item.second.bSize;
            bd.type = BTHi;
            bd.smo = eSmoHiSum;
            bd.prior = params.hiPriorWeight;
        } else {
            bd.length = sp->GetSize() + sq->GetSize();
            bd.type = BTLo;
            bd.smo = eSmoOcc;
            bd.prior = params.occPriorWeight;
        }
        sq->boundaryData[sp] = bd;
    }

}

void SPSegmentationEngine::SplitPixels()
{
    int currentPixelSize = PixelSize();

    if (currentPixelSize <= 1) return;

    int imgPixelsRows = pixelsImg.rows * 2;
    int imgPixelsCols = pixelsImg.cols * 2;

    if (pixelsImg(0, pixelsImg.cols - 1).GetCSize() == 1) imgPixelsCols--;
    if (pixelsImg(pixelsImg.rows - 1, 0).GetRSize() == 1) imgPixelsRows--;

    Matrix<Pixel> newPixelsImg(imgPixelsRows, imgPixelsCols);

    if (params.stereo) {
        for (Superpixel*& sp : superpixels) {
            ((SuperpixelStereo*)sp)->ClearPixelSet();
        }
    }

    for (Pixel& p : pixelsImg) {
        int pRowSize = p.GetRSize();
        int pColSize = p.GetCSize();
        int newRow = 2 * p.row;
        int newCol = 2 * p.col;

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
    }
    pixelsImg = newPixelsImg;

    if (params.stereo) {
        for (Pixel& p : pixelsImg) {
            ((SuperpixelStereo*)p.superPixel)->AddToPixelSet(&p);
        }
        InitializePPImage();
    }
}

void SPSegmentationEngine::IterateMoves()
{
    // tuple: (p, q), p, q are neighboring Pixels
    // We move pixel p to superpixel q->superPixel
    // typedef tuple<Pixel*, Pixel*> ListItem;
    static const int nDeltas[][2] = { { -1, 0 }, { 1, 0 }, { 0, 1 }, { 0, -1 } };

    CircList<Pixel*> list(10 * pixelsImg.rows * pixelsImg.cols); 
    int itemCount = 1;

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

    int count = 0;
    PixelMoveData tryMoveData[4];
    Superpixel* nbsp[5];
    int nbspSize;

    while (!list.Empty()) {
        Pixel*& p = list.Front();

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
                    if (params.stereo) TryMovePixelStereo(img, depthImg, inliers, pixelsImg, p, q, params.dispBeta, tryMoveData[m]); 
                    else TryMovePixel(img, pixelsImg, p, q, tryMoveData[m]);
                    nbsp[nbspSize++] = q->superPixel;
                }
            }
        }

        PixelMoveData* bestMoveData = FindBestMoveData(params, tryMoveData);

        if (bestMoveData != nullptr) {
            if (params.stereo) MovePixelStereo(pixelsImg, *bestMoveData);
            else MovePixel(pixelsImg, *bestMoveData);

            list.PushBack(p);
            for (int m = 0; m < 5; m++) {
                Pixel* qq = PixelAt(pixelsImg, p->row + nDeltas[m][0], p->col + nDeltas[m][1]);
                if (qq != nullptr && p->superPixel != qq->superPixel)
                    list.PushBack(qq);
            }
        }

        list.PopFront();
        count++;
    }

}

Mat SPSegmentationEngine::GetSegmentedImage()
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
    for (Superpixel* sp : superpixels) {
        cout << "Border length: " << sp->GetBorderLength() << "/";
        cout << CalcSuperpixelBoundaryLength(pixelsImg, sp) << endl;
    }
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
    for (double& t : performanceInfo.levels)
        cout << t << ' ';
    cout << endl;
}

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
            const double& disp = depthImg(i, j);
            const Plane_d& plane = ((SuperpixelStereo*)p->superPixel)->plane;
            SuperpixelStereo* sps = (SuperpixelStereo*)p->superPixel;
            
            sps->sumIRow += i; sps->sumIRow2 += i*i;
            sps->sumICol += j; sps->sumICol2 += j*j;
            sps->sumIRowCol += i*j;
            sps->sumIRowD += i*disp; sps->sumIColD += j*disp;
            sps->sumID += disp;
            sps->nI++;
            inliers(i, j) = fabs(DotProduct(sps->plane, i, j, 1.0) - disp) < params.inlierThreshold;
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


