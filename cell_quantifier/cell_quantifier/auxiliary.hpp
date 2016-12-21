#pragma once

/*
	Header that defines constants and helper functions of general character.
*/

// misc includes
#include <windows.h>
#include <tchar.h>
#include <sys/types.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include "interpolation.hpp"

// STDlib includes
#include <cmath>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Constants
const double pi = 3.14159265358979323846;

// Misc functions
int mod(int a, int b);
double clamp(double val, double low, double high);
std::vector<std::string> listFilesInDirectory(std::string path);


// OpenCV-related functions
void overlayFoundBorders(std::string GTPath, cv::Mat segmented, std::string windowName);

void augmentImageAndLabel(std::string imagePath, std::string labelPath, double magnitude, int iterations);
void elasticDeformation(cv::Mat* image, cv::Mat* label, double sigma, double alpha);
bool isInsideConvexQuadrilateral(cv::Point2d p, cv::Point2d q1, cv::Point2d q2, cv::Point2d q3, cv::Point2d q4);
bool winding_isInPolygon(cv::Point2d P, std::vector<cv::Point2d> V);
double triangleArea(cv::Point2d a, cv::Point2d b, cv::Point2d c);

