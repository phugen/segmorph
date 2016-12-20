/*
	Segments an image based on the active contour (snake) algorithm.
*/

#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>


cv::Mat segmentSnake(cv::Mat input);