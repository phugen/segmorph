/*
Test suite for various segmentation techniques.
*/

#include "SegmentationTest.hpp"

void testThreshold(cv::Mat image)
{
	cv::Mat imageGray = cv::Mat::zeros(cv::Size(image.rows, image.cols), image.type());
	cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);

	// Adaptive threshold filter
	cv::Mat threshold = cv::Mat::zeros(cv::Size(imageGray.rows, imageGray.cols), imageGray.type());
	cv::adaptiveThreshold(imageGray, threshold, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 3, 2);

	// show result
	cv::namedWindow("Adaptive Threshold 11x11, C=2", cv::WINDOW_NORMAL);
	cv::imshow("Adaptive Threshold 11x11, C=2", threshold);
	cv::waitKey(0);
}