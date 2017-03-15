/*
	Header for a Gaussian Mixture Model (GMM) approach to cell segmentation.
*/


#include <opencv2/ml.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <fstream>

cv::Mat segmentGMM(cv::Mat image, int nGaussians, int iterations, float eps);