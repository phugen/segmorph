#include "graphCut.hpp"

/*
	Find a binary segmentation of the image by using a graph cut
	approach - construct a 2D graph from an initial Gaussian mixture
	model and find its minimum cut to minimize the energy of the image.
*/
cv::Mat segmentGraphcut(cv::Mat input, int iterations)
{
	cv::Mat threechan, background_pixels, segmented;
	input.convertTo(threechan, CV_8UC3); // convert to 3-channel image
	background_pixels = cv::Mat::zeros(threechan.size(), threechan.type());
	background_pixels.setTo(cv::Scalar(255, 0, 0)); // fill with solid background

	segmented = cv::Mat::zeros(threechan.size(), threechan.type());
	segmented.setTo(cv::Scalar(255, 255, 255));

	// define bounding rectangle for certain background
	// TODO: Rough segmentation, then draw larger bounding boxes and 
	// work on each detected part separately, then stitch the image back together.
	cv::Rect rectangle(53, 242, 150, 80);
	cv::Mat bgdModel, fgdModel; // for internal algorithm use

	// segment image
	cv::grabCut(threechan, background_pixels, rectangle, bgdModel, fgdModel, iterations, cv::GC_INIT_WITH_RECT);

	// use output as mask to generate final output image
	//cv::Mat foreground(threechan.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	//threechan.copyTo(foreground, background_pixels); // only copy foreground pixels

	cv::compare(background_pixels, cv::GC_PR_FGD, background_pixels, cv::CMP_EQ);
	cv::Mat foreground(threechan.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	threechan.copyTo(foreground, background_pixels);

	// draw rectangle on original image
	cv::rectangle(threechan, rectangle, cv::Scalar(255, 255, 255), 1);
	cv::namedWindow("Rect");
	cv::imshow("Rect", threechan);

	cv::namedWindow("graph cut mask", cv::WINDOW_NORMAL);
	cv::imshow("graph cut mask", foreground);

	return foreground;
}