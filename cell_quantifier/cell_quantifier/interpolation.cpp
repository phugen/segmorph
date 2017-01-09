#include "interpolation.hpp"
#include "auxiliary.hpp"
#include "ElasticDeformation.hpp"
#include <iostream>


// Calculate the cross product of two points.
double cross(cv::Point2d a, cv::Point2d b)
{
	return a.x*b.y - a.y*b.x;
}


// Find color of point p by checking which
// of the four quad points is closest to it
// and taking the color of that point.
cv::Vec3d nearestNeighborInterpolation(cv::Mat image, cv::Point2d* p, cv::Point2d* a, cv::Point2d* b, cv::Point2d* c, cv::Point2d* d)
{
	// sort points a to d by their distance from p
	std::vector<cv::Point2d*> sortedByDist = {a, b, c, d};
	std::sort(sortedByDist.begin(), sortedByDist.end(), [p](const cv::Point2d* a, const cv::Point2d* b)
		{ 
			return (cv::norm(*p - *a) < cv::norm(*p - *b)); // euclidean distance
	});

	// return color of closest point
	cv::Point2d closest = *sortedByDist.front();
	cv::Vec3d color = readSafe<cv::Vec3d>(&image, closest.y, closest.x);
		

	return color;
}


// Interpolates the color of a point p in a triangle
// that is defined by the three points a, b and c. The area
// of the sub-triangles ABP, BCP and CAP weights the color of p.
cv::Vec3d barycentricInterpolation(cv::Mat image, cv::Point2d* p, cv::Point2d* a, cv::Point2d* b, cv::Point2d* c)
{
	// get weight coefficients from triangle areas
	double tri_area = triangleArea(a, b, c);

	double alpha_a = triangleArea(b, c, p) / tri_area;
	double alpha_b = triangleArea(a, p, c) / tri_area;
	double alpha_c = triangleArea(a, b, p) / tri_area;

	// alpha_a + alpha_b + alpha_c = 1, get interpolated color
	cv::Vec3d pixelVal = readSafe<cv::Vec3d>(&image, a->y, a->x) * alpha_a
					   + readSafe<cv::Vec3d>(&image, b->y, b->x) * alpha_b
					   + readSafe<cv::Vec3d>(&image, c->y, c->x) * alpha_c;

	return pixelVal;
}


// Find u,v interpolation values for a point p
// that is lies between the four vertices of the quadroid 
// defined by a, b, c and d.
// see: http://www.iquilezles.org/www/articles/ibilinear/ibilinear.htm
// and: http://stackoverflow.com/questions/808441/inverse-bilinear-interpolation
cv::Vec2d invBilinearInterpolation(cv::Mat image, cv::Point2d p, cv::Point2d a, cv::Point2d b, cv::Point2d c, cv::Point2d d)
{
	cv::Point2d e = b - a;
	cv::Point2d f = d - a;
	cv::Point2d g = a - b + c - d;
	cv::Point2d h = p - a;

	double k2 = cross(g, f);
	double k1 = cross(e, f) + cross(h, g);
	double k0 = cross(h, e);

	// perfect rectangle; switch to linear (non-quadratic) case.
	if (k2 == 0)
	{
		double v = -k0 / k1;
		double u = (h.x - f.x * v) / (e.x + g.x * v);

		return cv::Vec2d(u, v);
	}

	double w = k1*k1 - 4.0*k0*k2;

	// no solution because sqrt(-w) would be complex (TODO: what does this mean?)
	if (w < 0.0)
		return cv::Vec2d(-1.0, -1.0);

	w = sqrt(w);

	// first solution (neg. square root)
	double v1 = (-k1 - w) / (2.0*k2);
	double u1 = (h.x - f.x*v1) / (e.x + g.x*v1);

	// second solution (pos. square root)
	double v2 = (-k1 + w) / (2.0*k2);
	double u2 = (h.x - f.x*v2) / (e.x + g.x*v2);

	double u = u1;
	double v = v1;

	// if first solution params not in [0, 1]
	// choose second solution
	if (v < 0.0 || v > 1.0 || u < 0.0 || u > 1.0)
	{
		u = u2;
		v = v2;
	}

	// if those aren't either, there is none (I guess)
	if (v < 0.0 || v > 1.0 || u < 0.0 || u > 1.0)
	{
		u = clamp(u, 0, 1); // -1.0;
		v = clamp(v, 0, 1); // -1.0;
	}

	return cv::Vec2d(u, v);
}


float bilinearInterpolation()
{
	// bilinear interpolation for irregular (non-quadratic) four-point shapes
	// see: http://math.stackexchange.com/questions/828392/spatial-interpolation-for-irregular-grid

	// Inititalize matrices and vectors:
	// X = coordinate matrix
	/*	cv::Vec<int32_t, 6> x_1 = cv::Vec<int32_t, 6>(q1.x*q1.x, q1.x*q1.y, q1.y*q1.y, q1.x, q1.y, 1);
	cv::Vec<int32_t, 6> x_2 = cv::Vec<int32_t, 6>(q2.x*q2.x, q2.x*q2.y, q2.y*q2.y, q2.x, q2.y, 1);
	cv::Vec<int32_t, 6> x_3 = cv::Vec<int32_t, 6>(q3.x*q3.x, q3.x*q3.y, q3.y*q3.y, q3.x, q3.y, 1);
	cv::Vec<int32_t, 6> x_4 = cv::Vec<int32_t, 6>(q4.x*q.x, q4.x*q4.y, q4.y*q4.y, q4.x, q4.y, 1);

	cv::Mat X = cv::Mat(4, 6, CV_32SC1);
	X.row(0) = x_1;
	X.row(1) = x_2;
	X.row(2) = x_3;
	X.row(3) = x_4;

	// a = unknown bilinear parameter vector
	cv::Vec<int32_t, 6> a = cv::Vec<int32_t, 6>();

	// z = output vector (brightness of the pixels)
	cv::Mat z = cv::Mat(4, 1, CV_32SC1);
	cv::Vec<int32_t, 4> z = cv::Vec<int32_t, 4>(image->at<uchar>(q1), image->at<uchar>(q2), image->at<uchar>(q3), image->at<uchar>(q4));

	// E = diagonal matrix with entries [1,1,1,0,0,0]
	cv::Vec<int32_t, 6> e_1 = cv::Vec<int32_t, 6>(1, 0, 0, 0, 0, 0);
	cv::Vec<int32_t, 6> e_2 = cv::Vec<int32_t, 6>(0, 1, 0, 0, 0, 0);
	cv::Vec<int32_t, 6> e_3 = cv::Vec<int32_t, 6>(0, 0, 1, 0, 0, 0);
	cv::Vec<int32_t, 6> e_4 = cv::Vec<int32_t, 6>(0, 0, 0, 0, 0, 0);
	cv::Vec<int32_t, 6> e_5 = cv::Vec<int32_t, 6>(0, 0, 0, 0, 0, 0);
	cv::Vec<int32_t, 6> e_6 = cv::Vec<int32_t, 6>(0, 0, 0, 0, 0, 0);

	cv::Mat E = cv::Mat(6, 6, CV_32SC1);
	E.row(0) = e_1;
	E.row(1) = e_2;
	E.row(2) = e_3;
	E.row(3) = e_4;
	E.row(4) = e_5;
	E.row(5) = e_6;
	*/

	return 0.0;
}