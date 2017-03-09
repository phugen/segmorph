#include "auxiliary.hpp"
#include "ElasticDeformation.hpp"
#include "extractSamples.hpp"


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


// Analogous to mod above, but for floating-point numbers.
double fmod_custom(double x, double m)
{
	double res = INFINITY;

	if (m == 0)
		return res;

	double posMod, negMod, posM, posX;

	posM = m < 0 ? -m : m;
	posX = x < 0 ? -x : x;

	posMod = fmod(posX, posM);
	negMod = fmod(-posX, posM) + posM;

	// pick up the correct res
	if (x >= 0)
	{
		if (m > 0)
		{
			res = posMod;
		}

		else
		{
			res = -negMod;
		}
	}

	else
	{
		if (m > 0)
		{
			res = negMod;
		}

		else
		{
			res = -posMod;
		}
	}

	return res;
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


/* ======= Inclusion (convex only) algorithm ============================== */

/* Create an efficiency structure (see Preparata) for the convex polygon which
* allows binary searching to find which edge to test the point against.  This
* algorithm is O(log n).
*
* Call setup with 2D polygon _pgon_ with _numverts_ number of vertices,
* which returns a pointer to an inclusion anchor structure.
* Call testing procedure with a pointer to this structure and test point
* _point_, returns 1 if inside, 0 if outside.
* Call cleanup with pointer to inclusion anchor structure to free space.
*
* CONVEX must be defined for this test; it is not usable for general polygons.
*/


/* make inclusion wedge set */
/*pInclusionAnchor InclusionSetup(pgon, numverts)
{
	double	pgon[][2];
	int	numverts;

	int	pc, p1, p2, flip_edge;
	double	ax, ay, qx, qy, wx, wy, len;
	pInclusionAnchor pia;
	pInclusionSet	pis;

	// double the first edge to avoid needing modulo during test search
	pia = (pInclusionAnchor)malloc(sizeof(InclusionAnchor));
	MALLOC_CHECK(pia);
	pis = pia->pis =
		(pInclusionSet)malloc((numverts + 1) * sizeof(InclusionSet));
	MALLOC_CHECK(pis);

	pia->hi_start = numverts - 1;

	// get average point to make wedges from
	qx = qy = 0.0;
	for (p2 = 0; p2 < numverts; p2++) {
		qx += pgon[p2][X];
		qy += pgon[p2][Y];
	}
	pia->qx = qx /= (double)numverts;
	pia->qy = qy /= (double)numverts;

	// take cross product of vertex to find handedness
	pia->flip_edge = flip_edge =
		(pgon[0][X] - pgon[1][X]) * (pgon[1][Y] - pgon[2][Y]) >
		(pgon[0][Y] - pgon[1][Y]) * (pgon[1][X] - pgon[2][X]);


	ax = pgon[0][X] - qx;
	ay = pgon[0][Y] - qy;
	len = sqrt(ax * ax + ay * ay);
	if (len == 0.0) {
		fprintf(stderr, "sorry, polygon for inclusion test is defective\n");
		exit(1);
	}
	pia->ax = ax /= len;
	pia->ay = ay /= len;

	// loop through edges, and double last edge
	for (pc = p1 = 0, p2 = 1
		; pc <= numverts
		; pc++, p1 = p2, p2 = (++p2) % numverts, pis++) {

		// wedge border
		wx = pgon[p1][X] - qx;
		wy = pgon[p1][Y] - qy;
		len = sqrt(wx * wx + wy * wy);
		wx /= len;
		wy /= len;

		// cosine of angle from anchor border to wedge border
		pis->dot = ax * wx + ay * wy;
		// sign from cross product
		if ((ax * wy > ay * wx) == flip_edge) {
			pis->dot = -2.0 - pis->dot;
		}

		// edge
		pis->ex = pgon[p1][Y] - pgon[p2][Y];
		pis->ey = pgon[p2][X] - pgon[p1][X];
		pis->ec = pis->ex * pgon[p1][X] + pis->ey * pgon[p1][Y];

		// check sense and reverse plane eqns if need be
		if (flip_edge) {
			pis->ex = -pis->ex;
			pis->ey = -pis->ey;
			pis->ec = -pis->ec;
		}
	}
	// set first angle a little > 1.0 and last < -3.0 just to be safe.
	pia->pis[0].dot = -3.001;
	pia->pis[numverts].dot = 1.001;

	return(pia);
}

// Find wedge point is in by binary search, then test wedge
int InclusionTest(pia, point)
pInclusionAnchor	pia;
double	point[2];
{
	register double tx, ty, len, dot;
	int	inside_flag, lo, hi, ind;
	pInclusionSet	pis;

	tx = point[X] - pia->qx;
	ty = point[Y] - pia->qy;
	len = sqrt(tx * tx + ty * ty);
	// check if point is exactly at anchor point (which is inside polygon)
	if (len == 0.0) return(1);
	tx /= len;
	ty /= len;

	// get dot product for searching
	dot = pia->ax * tx + pia->ay * ty;
	if ((pia->ax * ty > pia->ay * tx) == pia->flip_edge) {
		dot = -2.0 - dot;
	}

	// binary search through angle list and find matching angle pair
	lo = 0;
	hi = pia->hi_start;
	while (lo <= hi) {
		ind = (lo + hi) / 2;
		if (dot < pia->pis[ind].dot) {
			hi = ind - 1;
		}
		else if (dot > pia->pis[ind + 1].dot) {
			lo = ind + 1;
		}
		else {
			goto Foundit;
		}
	}
	// should never reach here, but just in case...
	fprintf(stderr,
		"Hmmm, something weird happened - bad dot product %lg\n", dot);

Foundit:

	// test if the point is outside the wedge's exterior edge
	pis = &pia->pis[ind];
	inside_flag = (pis->ex * point[X] + pis->ey * point[Y] <= pis->ec);

	return(inside_flag);
}

void InclusionCleanup(p_inc_anchor)
pInclusionAnchor p_inc_anchor;
{
	free(p_inc_anchor->pis);
	free(p_inc_anchor);
}*/



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
double isLeft(cv::Point2d* P0, cv::Point2d* P1, cv::Point2d* P2)
{
	double eps = 1e-10;
	double val = (P1->x - P0->x) * (P2->y - P0->y) - (P2->x - P0->x) * (P1->y - P0->y);

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
bool winding_isInPolygon(cv::Point2d* P, std::vector<cv::Point2d*> V)
{
	size_t n = V.size() - 1;
	int    wn = 0;    // the  winding number counter

	// loop through all edges of the polygon
	for (size_t i = 0; i<n; i++) {   // edge from V[i] to  V[i+1]
		if (V[i]->y <= P->y) {          // start y <= P.y
			if (V[i + 1]->y  > P->y)      // an upward crossing
			if (isLeft(V[i], V[i + 1], P) > 0)  // P left of  edge
				++wn;            // have  a valid up intersect
		}
		else {                        // start y > P.y (no test needed)
			if (V[i + 1]->y <= P->y)     // a downward crossing
			if (isLeft(V[i], V[i + 1], P) < 0)  // P right of  edge
				--wn;            // have  a valid down intersect
		}
	}

	// if winding number > 0 or < 0, point is inside or on border
	return (wn > 0 || wn < 0);
}


// Calculates the absolute area of a triangle.
double triangleArea(cv::Point2d* a, cv::Point2d* b, cv::Point2d* c)
{
	double area = fabs((a->x * (b->y - c->y) + b->x * (c->y - a->y) + c->x * (a->y - b->y)) / 2.);

	return area;
}


/*
	Returns true if the point is inside the quadrilateral
	defined by the clockwise points q1 (top left) to q4,
	otherwise returns false.
	Points on the boundary count as inside.
*/
bool isInsideConvexQuadrilateral(cv::Point2d* p, cv::Point2d* q1, cv::Point2d* q2, cv::Point2d* q3, cv::Point2d* q4)
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
	double area1 = triangleArea(q1, q2, p);
	double area2 = triangleArea(q2, q3, p);
	double area3 = triangleArea(q3, q4, p);
	double area4 = triangleArea(q4, q1, p);

	// p lies on the boundary, so two triangles have zero area.
	// Count those points as inside the triangle.
	if (area1 == 0 || area2 == 0 || area3 == 0 || area4 == 0)
		return true;

	double tris_area = area1 + area2 + area3 + area4;

	// calculate area of quadrilateral. If the total area of
	// all triangles defined by (q_i, q_i+1, p) is larger than
	// the quadrilateral area, then p is outside the quadrilateral.

	// find lengths of sides:
	double l_a = sqrt(pow(q4->x - q3->x, 2) + pow(q4->y - q3->y, 2));
	double l_b = sqrt(pow(q3->x - q2->x, 2) + pow(q3->y - q2->y, 2));
	double l_c = sqrt(pow(q2->x - q1->x, 2) + pow(q2->y - q1->y, 2));
	double l_d = sqrt(pow(q1->x - q4->x, 2) + pow(q1->y - q4->y, 2));

	// semiperimeter of a quadrilateral:
	double semi = (l_a + l_b + l_c + l_d) / 2;

	// angles between a/d and b/c:
	cv::Vec2d a = cv::Vec2d(q3->x - q4->x, q3->y - q4->y);
	cv::Vec2d d = cv::Vec2d(q1->x - q4->x, q1->y - q4->y);
	cv::Vec2d b = cv::Vec2d(q3->x - q2->x, q3->y - q2->y);
	cv::Vec2d c = cv::Vec2d(q1->x - q2->x, q1->y - q2->y);

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
	Creates more training data from an image (and its ground truth label image) by applying
	elastic grid deformation by random vectors of max. strength <magnitude>

	The "iterations" variable controls how many distorted versions of the input
	image are created.
*/
std::vector<std::string> augmentImageAndLabel(std::string imagePath, std::string labelPath, std::string outpath, double magnitude)
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
	
	// perform elastic deformation
	cv::Mat elasticImage = image;
	cv::Mat elasticLabel = label;

	int gridSize = 1;
	int sigma = 20;
	int alpha = magnitude;
	elasticDeformation(&elasticImage, &elasticLabel, gridSize, sigma, alpha);


	// save augmented image
	//std::string imageName = imagePath.substr(imagePath.find_last_of("/\\") + 1);
	std::string imageName = imagePath.substr(imagePath.find_last_of("/\\") + 1);

	std::string replaceImage = imageName;
	std::string replaceLabel = imageName;

	std::string augName = replaceImage.replace(replaceImage.cend() - 4, replaceImage.cend(), "") + "_AUGMENTED.png";
	std::string augLabName = replaceLabel.replace(replaceLabel.cend() - 4, replaceLabel.cend(), "") + "_AUGMENTED_label.png";

	std::string imageOutputPath = outpath + augName;
	std::string labelOutputPath = outpath + augLabName;


	// set compression parameters because otherwise
	// writing PNGs crashes the program
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9); // maximum quality

	cv::imwrite(imageOutputPath, elasticImage, compression_params);
	cv::imwrite(labelOutputPath, elasticLabel, compression_params);
	

	// return paths of augmented image and label
	std::vector<std::string> augPaths = { imageOutputPath, labelOutputPath };
	 
	return augPaths;
}