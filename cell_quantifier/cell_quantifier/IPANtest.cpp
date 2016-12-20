/*
	IPAN test suite.
*/

#include "ipanTest.hpp"


bool testIpan(double dmin, double dmax, double alphamax)
{
	std::vector<IpanPoint*> curvePoints, corners;
	std::vector<cv::Point*> curvePoints_cv;

	// define a square curve
	curvePoints.emplace_back(new IpanPoint(100,300));
	curvePoints.emplace_back(new IpanPoint(200,300));
	curvePoints.emplace_back(new IpanPoint(300,300));
	curvePoints.emplace_back(new IpanPoint(300,200));
	curvePoints.emplace_back(new IpanPoint(300,100));
	curvePoints.emplace_back(new IpanPoint(200,100));
	curvePoints.emplace_back(new IpanPoint(100,100));
	curvePoints.emplace_back(new IpanPoint(100,200));

	// create preview image
	cv::Mat matCurve = cv::Mat::zeros(cv::Size(500, 500), CV_8UC3);
	matCurve = cv::Scalar(255, 255, 255); // white BG

	// convert IPAN points to cv::Points
	for (auto point = curvePoints.begin(); point != curvePoints.end(); point++)
	{
		curvePoints_cv.emplace_back(new cv::Point((*point)->x, (*point)->y));
	}

	// draw curve lines
	for (int i = 0; i < curvePoints_cv.size() - 1; i++)
	{
		line(matCurve, *curvePoints_cv.at(i), *curvePoints_cv.at(i+1), cv::Scalar(0, 0, 0), 5, CV_AA, 0);
	}
	// TODO: closed toggle oder check ob first = last coordinates?
	line(matCurve, *curvePoints_cv.at(0), *curvePoints_cv.at(curvePoints_cv.size() - 1), cv::Scalar(0, 0, 0), 5, CV_AA, 0); // draw final, "closing" line

	// draw curve points
	for (auto point = curvePoints_cv.begin(); point != curvePoints_cv.end(); point++)
	{
		cv::circle(matCurve, **point, 5, cv::Scalar(255, 0, 0), 10, CV_AA, 0);
	}
	

	//const cv::Point *previewPoints = (const cv::Point*) cv::Mat(curvePoints_cv).data; // convert points to mat for easy pointer conversion
	//int curveIndices = cv::Mat(curvePoints_cv).rows; // same trick for indices
	//cv::polylines(matCurve, &previewPoints, &curveIndices, 1, true, cv::Scalar(0, 0, 0), 50, CV_AA, 0);
	

	cv::namedWindow("Initial curve", 0);
	cv::imshow("Initial curve", matCurve);
	//cv::waitKey(0);

	// find corners of curve
	corners = Ipan::getCorners(curvePoints, dmin, dmax, alphamax);

	if (corners.size() == 0)
	{
		std::cout << "No corners found... perhaps something went wrong?\n";
	}

	else
	{
		// draw red circles around detected corners
		for (auto corner = corners.begin(); corner != corners.end(); corner++)
		{
			cv::circle(matCurve, cv::Point((*corner)->x, (*corner)->y), 50, cv::Scalar(0, 0, 255), 2, CV_AA, 0);
		}

		// show corners
		cv::namedWindow("Detected corners", 0);
		cv::imshow("Detected corners", matCurve);
		cv::waitKey(0);
	}



	return true;
}