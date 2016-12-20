#pragma once

/*
	The header file for the IPAN planar curve corner detection algorithm, described in
	"A Simple and Efficient Algorithm for Detection	of High Curvature Points in Planar Curves"
*/

// OpenCV headers
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Auxiliary container headers
#include <vector>

// Math headers / defines
#include <cmath>

// custom IPAN headers
#include "ipanPoint.hpp"



class Ipan
{
private:
	static void calcStrengths(std::vector<IpanPoint*> curvePoints, double dmin, double dmax, double alphamax);

public:
	Ipan();
	~Ipan();

	static std::vector<IpanPoint*> getCorners(std::vector<IpanPoint*> curvePoints, double dmin, double dmax, double alphamax);

};