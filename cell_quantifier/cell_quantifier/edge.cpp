#include "edge.hpp"

// Automatically find a high threshold for canny by using Otsu's method,
// then set tLow to 0.5 * tHigh and detect contours.
cv::Mat segmentEdge(cv::Mat input, int cKernelsize)
{
	cv::Mat gray, otsu, canny;
	cv::cvtColor(input, gray, CV_BGR2GRAY);


	double otsu_thresh = cv::threshold(gray, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU); // get Otsu's threshold value
	cv::Canny(gray, canny, 0.5 * otsu_thresh, otsu_thresh, cKernelsize); // use as higher threshold and derive lower from it

	return canny;
}