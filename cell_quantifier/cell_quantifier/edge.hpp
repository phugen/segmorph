/*
	Header for segmentation approach that DIRECTLY tries to find edges.
*/

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


cv::Mat segmentEdge(cv::Mat input, int cKernelsize);
