// spixel.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "segengine.h"
#include "functions.h"
#include "utils.h"
#include <fstream>

using namespace cv;
using namespace std;

//
// Test performances/algorithms/... delete for final version
//

void test0() 
{
    Matrix<Superpixel> spvec(1000, 1000);

    for (Superpixel& sp : spvec) {
        sp.FinishInitialization();
    }
}

void test1()
{
    Matrix<Superpixel*> spvec(1000, 1000);
    for (Superpixel*& sp : spvec) {
        sp = new Superpixel();
    }
    for (Superpixel* sp : spvec) {
        sp->FinishInitialization();
    }
    for (Superpixel*& sp : spvec) {
        delete sp;
    }

}

void test2()
{

    map<Superpixel*, int> map0;

    for (int i = 0; i < 1000000; i++) {
        map<Superpixel*, int> map1;
        map1[new Superpixel()] = i;
        map1[new Superpixel()] = i + 2;

        map0 = map1;
    }

}

void test3()
{
    vector<pair<Superpixel*, int>> map0;

    for (int i = 0; i < 1000000; i++) {
        vector<pair<Superpixel*, int>> map1;

        Superpixel* sp1 = new Superpixel();
        bool found = false;
        for (auto a : map1) {
            if (a.first == sp1) {
                found = true;
                break;
            }
        }
        if (!found) map1.push_back(pair<Superpixel*, int>(sp1, i));

        sp1 = new Superpixel();
        found = false;
        for (auto a : map1) {
            if (a.first == sp1) {
                found = true;
                break;
            }
        }
        if (!found) map1.push_back(pair<Superpixel*, int>(sp1, i));

        map0 = map1;
    }

}

struct Pix {
    int r, c;
    int i0, j0, i1, j1;
};

struct Pix0 {
    int r, c;
};


void test4()
{
    Mat1d mat(1000, 500);

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            mat(i, j) = i + j;
        }
    }

}

void test5(int a, int b)
{
    Mat1d mat(1000, 500);

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            for (int i0 = i; i0 < i + a; i0++) {
                for (int j0 = j; j0 < j + b; j0++)
                    mat(i0, j0) = i0 + j0;
            }
        }
    }

}

void test()
{
    Timer t;
    
    test0();
    t.Stop();

    cout << "Test 0 time: " << t.GetTimeInSec() << endl;
    
    t.Reset();
    test1();
    t.Stop();
    cout << "Test 1 time: " << t.GetTimeInSec() << endl;

    t.Reset();
    test2();
    t.Stop();
    cout << "Test 2 time: " << t.GetTimeInSec() << endl;

    t.Reset();
    test3();
    t.Stop();
    cout << "Test 3 time: " << t.GetTimeInSec() << endl;

    t.Reset();
    for (int a = 0; a < 100; a++) test4();
    t.Stop();
    cout << "Test 4 time: " << t.GetTimeInSec() << endl;

    t.Reset();
    for (int a = 0; a < 100; a++) test5(1, 1);
    t.Stop();
    cout << "Test 5 time: " << t.GetTimeInSec() << endl;

    //Mat img = imread("C:\\Work\\C\\speedup\\ours\\speedup\\x64\\Release\\000002_10L_left_disparity.png", CV_LOAD_IMAGE_ANYDEPTH);
    //
    //Mat imgt = AdjustDisparityImage(img);
    //imwrite("c:\\tmp\\intp.png", imgt);
    //cout << "test time: " << t.GetTimeInSec() << endl;
}

void ProcessFiles(const string& paramFile, const string& dirName, const string& pattern,
        const string& dispPattern)
{
    SPSegmentationParameters params = ReadParameters(paramFile);

    if (params.stereo && dispPattern.empty()) {
        cout << "Can not process stereo without disparity image." << endl;
        return;
    }

    vector<string> files;
    string fileDir = dirName;

    FindFiles(fileDir, pattern, files, false);
    EndDir(fileDir);

    int nProcessed = 0;
    double totalTime = 0.0;

    for (const string& f : files) {
        string fileName = fileDir + f;
        Mat image = imread(fileName, CV_LOAD_IMAGE_COLOR);
        Mat dispImage = Mat1w();

        if (image.rows == 0 || image.cols == 0) {
            cout << "Failed reading image '" << fileName << "'" << endl;
            continue;
        }
        if (params.stereo) {
            string dispFileName = ChangeExtension(fileName, dispPattern);

            dispImage = imread(dispFileName, CV_LOAD_IMAGE_ANYDEPTH);
            if (dispImage.rows == 0 || dispImage.cols == 0) {
                cout << "Failed reading dispaimage '" << dispFileName << "'" << endl;
                continue;
            }
        }
        cout << "Processing: " << fileName << endl;

        SPSegmentationEngine engine(params, image, dispImage);


        if (params.stereo) {
            engine.ProcessImageStereo();
            engine.PrintDebugInfoStereo();
        } else {
            engine.ProcessImage();
        }
        engine.PrintPerformanceInfo();
        totalTime += engine.ProcessingTime();

        string outImage = ChangeExtension(fileDir + "out/" + f, "_sp.png");
        string outImageSeg = ChangeExtension(fileDir + "seg/" + f, ".png");

        imwrite(outImage, engine.GetSegmentedImage());
        imwrite(outImageSeg, engine.GetSegmentation());

        if (params.stereo) {
            string outImageDisp = ChangeExtension(fileDir + "disp/" + f, ".png");
            imwrite(outImageDisp, engine.GetDisparity());
        }

        cout << "  no of superpixels: " << engine.GetNoOfSuperpixels() << endl;
        nProcessed++;
    }

    cout << "Processed " << nProcessed << " files in " << totalTime << " sec. ";
    if (nProcessed > 0) {
        cout << "Average per image " << (totalTime / nProcessed) << " sec.";
    }
    cout << endl;
}




int _tmain(int argc, char* argv[])
{
    // test();
    //_CrtSetBreakAlloc(298);
    //srand(1);
    if (argc == 4) {
        ProcessFiles(argv[1], argv[2], argv[3], "");
    } else if (argc == 5) {
        ProcessFiles(argv[1], argv[2], argv[3], argv[4]);
    } else {
        cout << "Usage: spixel config_file.yml file_dir file_pattern" << endl;
        cout << "   or: spixel config_file.yml file_dir file_pattern disparity_extension" << endl;
    }
    //_CrtDumpMemoryLeaks();
    return 0;
}

