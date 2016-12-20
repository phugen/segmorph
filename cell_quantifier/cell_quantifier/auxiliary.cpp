#include "auxiliary.hpp"
#include "ElasticDeformation.hpp"


/*
	"Real" modulo operator. Unlike %, this also handles
	 negative values for a and b.
*/
int mod(int a, int b)
{
	if (b < 0)
		return mod(a, -b);

	int ret = a % b;

	if (ret < 0)
		ret += b;

	return ret;
}


/*
	Clamps the input value to the specified range.
*/
double clamp(double val, double low, double high)
{
	if (val < low)
		return low;

	else if (val > high)
		return high;

	else
		return val;
}


// Copyright 2000 softSurfer, 2012 Dan Sunday
// This code may be freely used and modified for any purpose
// providing that this copyright notice is included with it.
// SoftSurfer makes no warranty for this code, and cannot be held
// liable for any real or imagined damage resulting from its use.
// Users of this code must verify correctness for their application.

// isLeft(): tests if a point is Left|On|Right of an infinite line.
//    Input:  three points P0, P1, and P2
//    Return: >0 for P2 left of the line through P0 and P1
//            =0 for P2  on the line
//            <0 for P2  right of the line
//    See: Algorithm 1 "Area of Triangles and Polygons"
double isLeft(cv::Point2d P0, cv::Point2d P1, cv::Point2d P2)
{
	double eps = 1e-10;
	double val = (P1.x - P0.x) * (P2.y - P0.y) - (P2.x - P0.x) * (P1.y - P0.y);

	// small error area around 0 counts as
	// zero instead of left or right
	if (fabs(val) < eps)
		return 0;

	else
		return val;
}

// wn_PnPoly(): winding number test for a point in a polygon
//      Input:   P = a point,
//               V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
//      Return:  wn = the winding number (=0 only when P is outside)
bool winding_isInPolygon(cv::Point2d P, std::vector<cv::Point2d> V)
{
	size_t n = V.size() - 1;
	int    wn = 0;    // the  winding number counter

	// loop through all edges of the polygon
	for (size_t i = 0; i<n; i++) {   // edge from V[i] to  V[i+1]
		if (V[i].y <= P.y) {          // start y <= P.y
			if (V[i + 1].y  > P.y)      // an upward crossing
			if (isLeft(V[i], V[i + 1], P) > 0)  // P left of  edge
				++wn;            // have  a valid up intersect
		}
		else {                        // start y > P.y (no test needed)
			if (V[i + 1].y <= P.y)     // a downward crossing
			if (isLeft(V[i], V[i + 1], P) < 0)  // P right of  edge
				--wn;            // have  a valid down intersect
		}
	}

	// if winding number >= 0, point is inside or on border
	return (wn >= 0);
}



/*
	Returns true if the point is inside the quadrilateral
	defined by the clockwise points q1 (top left) to q4,
	otherwise returns false.
	Points on the boundary count as inside.
*/
bool isInsideConvexQuadrilateral(cv::Point2d p, cv::Point2d q1, cv::Point2d q2, cv::Point2d q3, cv::Point2d q4)
{
	double eps = 1.1; // use error instead of == because we're comparing floating-point numbers!

	// if p is one of the corner points, it surely
	// is inside the quadrilateral
	if (fabs(cv::norm(p - q1)) < eps ||
		fabs(cv::norm(p - q2)) < eps ||
		fabs(cv::norm(p - q3)) < eps ||
		fabs(cv::norm(p - q4)) < eps)
	{
		return true;
	}
		
	// Split quadrilateral into four triangles
	// using adjacent points and p, then calculate
	// their area.
	double area1 = fabs((q1.x * (q2.y - p.y) + q2.x * (p.y - q1.y) + p.x * (q1.y - q2.y)) / 2.);
	double area2 = fabs((q2.x * (q3.y - p.y) + q3.x * (p.y - q2.y) + p.x * (q2.y - q3.y)) / 2.);
	double area3 = fabs((q3.x * (q4.y - p.y) + q4.x * (p.y - q3.y) + p.x * (q3.y - q4.y)) / 2.);
	double area4 = fabs((q4.x * (q1.y - p.y) + q1.x * (p.y - q4.y) + p.x * (q4.y - q1.y)) / 2.);

	// p lies on the boundary, so two triangles have zero area.
	// Count those points as inside the triangle.
	if (area1 == 0 || area2 == 0 || area3 == 0 || area4 == 0)
		return true;

	double tris_area = area1 + area2 + area3 + area4;

	// calculate area of quadrilateral. If the total area of
	// all triangles defined by (q_i, q_i+1, p) is larger than
	// the quadrilateral area, then p is outside the quadrilateral.

	// find lengths of sides:
	double l_a = sqrt(pow(q4.x - q3.x, 2) + pow(q4.y - q3.y, 2));
	double l_b = sqrt(pow(q3.x - q2.x, 2) + pow(q3.y - q2.y, 2));
	double l_c = sqrt(pow(q2.x - q1.x, 2) + pow(q2.y - q1.y, 2));
	double l_d = sqrt(pow(q1.x - q4.x, 2) + pow(q1.y - q4.y, 2));

	// semiperimeter of a quadrilateral:
	double semi = (l_a + l_b + l_c + l_d) / 2;

	// angles between a/d and b/c:
	cv::Vec2d a = cv::Vec2d(q3.x - q4.x, q3.y - q4.y);
	cv::Vec2d d = cv::Vec2d(q1.x - q4.x, q1.y - q4.y);
	cv::Vec2d b = cv::Vec2d(q3.x - q2.x, q3.y - q2.y);
	cv::Vec2d c = cv::Vec2d(q1.x - q2.x, q1.y - q2.y);

	// get opposite angles between vectors AD and BC in radians
	double alpha = acos(a.dot(d) / (cv::norm(a) * cv::norm(d)));
	double gamma = acos(b.dot(c) / (cv::norm(b) * cv::norm(c)));
	
	// Bretschneider's formula for the area of quadrilaterals:
	double area_quad = sqrt((semi - l_a) * (semi - l_b) * (semi - l_c) * (semi - l_d) - 0.5*l_a*l_b*l_c*l_d * (1 + cos(alpha + gamma)));

	return (fabs(area_quad - tris_area) < eps);
}


// Returns the paths of all files in the directory that
// the supplied path points to. (WINDOWS only)
std::vector<std::string> listFilesInDirectory(std::string path)
{
	std::vector<std::string> fileNames;
	std::string wildcard_path = path + "*";
	
	WIN32_FIND_DATA ffd;
	HANDLE h = FindFirstFileA(wildcard_path.c_str(), &ffd);
	bool success = false;

	if (h != INVALID_HANDLE_VALUE)
	{
		success = true;

		while (success)
		{
			// do something with ffd
			success = FindNextFileA(h, &ffd);

			if (strcmp(ffd.cFileName, ".") && strcmp(ffd.cFileName, "..")) // ignore hidden . and .. directories
				fileNames.emplace_back(path + std::string(ffd.cFileName)); // concatenate path and file name and place it in list
		}
	}

	return fileNames;
}


// Overlays the region borders found by a previously executed segmentation
// algorithm with the ground truth of the original image. 
void overlayFoundBorders(std::string GTPath, cv::Mat segmented, std::string windowName)
{
	// find contours of cells in image with Canny operator
	cv::Mat matCanny, redLines;
	redLines = cv::imread(GTPath);
	//cv::cvtColor(erodedGrayscale, redLines, CV_GRAY2BGR);
	

	int edgeThresh = 1;
	int lowThreshold = 100;
	int ratio = 3;
	int kernel_size = 3;

	cv::Canny(segmented, matCanny, lowThreshold, lowThreshold * ratio, kernel_size);

	cv::namedWindow("Canny", cv::WINDOW_NORMAL);
	cv::imshow("Canny", matCanny);

	// superimpose contours found by Canny operator on
	// original image for comparison
	for (int y = 0; y < redLines.rows; y++)
	{
		for (int x = 0; x < redLines.cols; x++)
		{
			// use Canny image as mask; if edge, color original
			// image pixel red to show contour
			//uchar grayval = image.at<uchar>(y, x);

			if (matCanny.at<uchar>(y, x) == 255)
				redLines.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
		}
	}

	cv::namedWindow(windowName, cv::WINDOW_NORMAL);
	cv::imshow(windowName, redLines);
}



/*
	Creates more training data from an image (and its ground truth label image) by applying the
	following transformations randomly:
		
	- Horizontal / Vertical mirroring
	- Rotation
	- Scaling
	- Elastic grid deformation by random vectors of max. strength <magnitude>

	The "iterations" variable controls how many distorted versions of the input
	image are created.
*/
void augmentImageAndLabel(std::string imagePath, std::string labelPath, double magnitude, int iterations)
{
	cv::Mat imageNew, labelNew;

	// load original image
	cv::Mat image = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);
	if (!image.data)
	{
		std::cout << "augmentImageAndLabel: Couldn't load image at path " << imagePath << std::endl;
		cv::waitKey(0);
		exit(-1);
	}

	// load its label image
	cv::Mat label = cv::imread(labelPath, CV_LOAD_IMAGE_COLOR);
	if (!label.data)
	{
		std::cout << "augmentImageAndLabel: Couldn't load label at path " << labelPath << std::endl;
		cv::waitKey(0);
		exit(-1);
	}

	// convert to floating point representation for consistency
	// in later steps which work with floating-point matrices
	image.convertTo(image, CV_64FC3);
	label.convertTo(label, CV_64FC3);

	
	// initialize the random number generator with time-dependent seed
	std::mt19937_64 rng;
	uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::seed_seq ss{ uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32) };
	rng.seed(ss);

	// initialize a uniform distribution between 0 and 1
	std::uniform_int_distribution<int> unif(0, 1);

	// randomly flip image
	int flipVert = unif(rng);
	int flipHor = unif(rng);

	cv::Mat flipped;
	cv::Mat flippedLabel;

	/*if (flipVert)
	{
		flip(image, flipped, 1);
		flip(label, flippedLabel, 1);

		image = flipped;
		label = flippedLabel;
	}

	if (flipHor)
	{
		flip(image, flipped, 0);
		flip(label, flippedLabel, 0);
	}*/

	flipped = image;
	flippedLabel = label;
	

	// perform elastic deformation ; TODO: find good values so that pixels don't swap places
	cv::Mat elasticImage = flipped;
	cv::Mat elasticLabel = flippedLabel;

	int gridSize = image.cols/8;
	int sigma = 5;
	int alpha = 1500;
	elasticDeformation(&elasticImage, &elasticLabel, gridSize, sigma, alpha);

	// save augmented image
	std::string imageOutputPath = imagePath.replace(imagePath.cend() - 4, imagePath.cend(), "") + "_AUGMENTED.bmp";
	std::string labelOutputPath = labelPath.replace(labelPath.cend() - 4, labelPath.cend(), "") + "_AUGMENTED.bmp";

	cv::imwrite(imageOutputPath, elasticImage);
	cv::imwrite(labelOutputPath, elasticLabel);

	//cv::waitKey(0);
}