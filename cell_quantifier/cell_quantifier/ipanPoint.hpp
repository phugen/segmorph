#pragma once

/*
The header file for an IPAN point; IPAN points are normal 2-dimensional points that
can be assigned a real-numbered corner strength value.
*/

// OpenCV headers
#include <opencv2/core.hpp>


class IpanPoint : public cv::Point2d 
{
private:
	double strength; // corner strength value
	friend std::ostream& operator<<(std::ostream&, const IpanPoint&);

public:
	IpanPoint() = delete; // standard constructor forbidden
	IpanPoint(double a, double b);
	//IpanPoint(cv::Point2d);
	~IpanPoint();

	double getStrength();
	void setStrength(double val);
};