#include <string>
#include <vector>

using namespace std;

// Returns list of files in dir following pattern. See implementation comment for restrictions.
void FindFiles(const string& dir, const string& pattern, vector<string>& files, bool fullPath = false);

// Returns fileName with extension replaced by newExt (extension is substring of fileName from (including) the last dot)
string ChangeExtension(const string& fileName, const string& newExt);

// Append '/' to the end of dirName if not present and non-empty
void EndDir(string& dirName);

// From Jian's original code
cv::Mat ConvertRGBToLab(const cv::Mat& img);

// Input is single channel ushort image, output is single channel double image
// Input is divided by 256.0 and gaps (regions with value 0) are "filled", 
cv::Mat AdjustDisparityImage(const cv::Mat& img);