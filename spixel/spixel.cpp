// spixel.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "segengine.h"
#include "functions.h"
#include "utils.h"
#include <fstream>

using namespace cv;
using namespace std;


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
            if (params.debugOutput) engine.PrintDebugInfoStereo();
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

        cout << "No. of superpixels: " << engine.GetNoOfSuperpixels() << endl;
        nProcessed++;
    }

    cout << "Processed " << nProcessed << " files in " << totalTime << " sec. ";
    if (nProcessed > 0) {
        cout << "Average per image " << (totalTime / nProcessed) << " sec.";
    }
    cout << endl;
}




int main(int argc, char* argv[])
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

