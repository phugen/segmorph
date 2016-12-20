/*
	An implementation of an IPAN curve point in the 2D plane, based on the 
	Point2d OpenCV class with added corner strength value.
*/

#include "ipanPoint.hpp"



IpanPoint::IpanPoint(double x, double y) : cv::Point2d(x, y)
{
	// simply call Point2d constructor with given coordinates
}


IpanPoint::~IpanPoint(){}


// Returns the corner strength of this point.
double IpanPoint::getStrength()
{
	return IpanPoint::strength;
}

// Sets the corner strength for this point.
void IpanPoint::setStrength(double val)
{
	IpanPoint::strength = val;
}

// Overload << to put x and y coordinates of point
std::ostream& operator<<(std::ostream &strm, const IpanPoint &p)
{
	return strm << p.x << "|" << p.y << "\n";
}