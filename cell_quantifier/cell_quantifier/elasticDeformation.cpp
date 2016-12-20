#include "elasticDeformation.hpp"

/*
Implements elastic deformation of an image as described in
"Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis." by Patrice Y. Simard, Dave Steinkraus, John C. Platt.

Alpha controls the strength of the deformation, while sigma changes the smoothing
factor of the Gaussian distribution. The grid size determines how many
grides nodes there will be; gridSize = 3 would result in a 3x3 grid on the image.
*/
void elasticDeformation(cv::Mat* image, cv::Mat* label, int gridSize, double sigma, double alpha)
{
	// check if grid size is <= #pixels
	assert(gridSize <= image->rows && gridSize <= image->cols);

	// Calculate number of "full" cells plus "broken" last cell if applicable
	int numCellX = image->cols / gridSize;
	if ((image->cols % gridSize) != 0)
		numCellX++;

	int numCellY = image->rows / gridSize;
	if ((image->rows % gridSize) != 0)
		numCellY++;

	// create 32-bit float displacement fields for x- and y-directions
	cv::Mat xField = cv::Mat::zeros(numCellY + 1, numCellX + 1, CV_64FC1); // numCellX+1 for "broken" last cell
	cv::Mat yField = cv::Mat::zeros(numCellY + 1, numCellX + 1, CV_64FC1);

	// fill both fields with random uniformly distributed
	// numbers in the range [-1, 1]



	// TODO: CHANGE THIS SO IT BECOMES RANDOM AGAIN



	std::random_device rdev;
	uint64_t seed = 1; // (uint64_t(rdev()) << 32) | rdev(); 
	cv::RNG randomgen = cv::RNG::RNG(seed);

	randomgen.fill(xField, cv::RNG::UNIFORM, -1, 1, true);
	randomgen.fill(yField, cv::RNG::UNIFORM, -1, 1, true);

	// smooth displacement fields by Gaussian filtering
	cv::GaussianBlur(xField, xField, cv::Size(0, 0), sigma, sigma);
	cv::GaussianBlur(yField, yField, cv::Size(0, 0), sigma, sigma);


	// normalize field to norm 1
	//cv::normalize(xField, xField, 1.0);
	//cv::normalize(yField, yField, 1.0);

	// scale all displacement vectors by scalar alpha
	xField *= alpha;
	yField *= alpha;


	// apply elastic deformation to input and label images by deplacing pixels according to the
	// x- and y-values at their position in the displacement fields
	cv::Mat imageElastic = cv::Mat(image->size(), CV_64FC3);
	cv::Mat labelElastic = cv::Mat(label->size(), CV_64FC3);

	labelElastic = cv::Vec3f(0, 255, 0);

	//std::cout << "Channels = " << imageElastic.channels() << ", Type = " << imageElastic.type() << std::endl;
	std::cout << "cols = " << imageElastic.cols << ", rows = " << imageElastic.rows << std::endl << std::endl;

	// calculate grid point displacement coordinates
	for (int y = 0; y < numCellY; y++)
	{
		for (int x = 0; x < numCellX; x++)
		{
			// clamp coordinates to image boundaries to handle OOB shifts
			// and find pixel the displacement vector points to by rounding
			double xcoord, ycoord;

			// clamp image coordinates so shifted points don't 
			// lie outside the image boundaries
			xcoord = clamp(x * gridSize + xField.at<double>(cv::Point2d(x, y)), 0, image->cols - 1);
			ycoord = clamp(y * gridSize + yField.at<double>(cv::Point2d(x, y)), 0, image->rows - 1);

			// save shifted grid points' coordinates
			xField.at<double>(cv::Point2d(x, y)) = xcoord;
			yField.at<double>(cv::Point2d(x, y)) = ycoord;
		}
	}

	

	for (int y = 0; y < numCellY; y++)
	{
		for (int x = 0; x < numCellX; x++)
		{
			// set the four corner points of area to interpolate (order: top left to bottom left, CW)
			cv::Point2d q1, q2, q3, q4;

			// q1 is the top left corner of current quad
			// and is never OOB
			q1.x = xField.at<double>(y, x);
			q1.y = yField.at<double>(y, x);

			// top left corner grid point
			if (y == 0 && x == 0)
			{
				q2.x = xField.at<double>(y, x + 1);
				q2.y = yField.at<double>(y, x + 1);

				q3.x = xField.at<double>(y + 1, x + 1);
				q3.y = yField.at<double>(y + 1, x + 1);

				q4.x = xField.at<double>(y + 1, x);
				q4.y = yField.at<double>(y + 1, x);
			}

			// top right corner
			else if (y == 0 && x == numCellX - 1)
			{
				q2.x = image->cols - 1;
				q2.y = q1.y;

				q3.x = image->cols - 1;
				q3.y = clamp(q2.y + gridSize, 0, image->rows - 1);

				q4.x = q1.x;
				q4.y = clamp(q1.y + gridSize, 0, image->rows - 1);
			}

			// bottom left corner
			else if (y == numCellY - 1 && x == 0)
			{
				q2.x = xField.at<double>(y, x + 1);
				q2.y = yField.at<double>(y, x + 1);

				q3.x = q2.x;
				q3.y = image->rows - 1;

				q4.x = q1.x;
				q4.y = image->rows - 1;
			}

			// bottom right corner
			else if (y == numCellY - 1 && x == numCellX - 1)
			{
				q2.x = image->cols - 1;
				q2.y = q1.y;

				q3.x = image->cols - 1;
				q3.y = image->rows - 1;

				q4.x = q1.x;
				q4.y = image->rows - 1;
			}

			// bottom edge
			else if (y == numCellY - 1)
			{
				q2.x = xField.at<double>(y, x + 1);
				q2.y = yField.at<double>(y, x + 1);

				q3.x = q2.x;
				q3.y = image->rows - 1;

				q4.x = q1.x;
				q4.y = image->rows - 1;
			}

			// right edge
			else if (x == numCellX - 1)
			{
				q2.x = image->cols - 1;
				q2.y = q1.y;

				q3.x = image->cols - 1;
				q3.y = clamp(q2.y + gridSize, 0, image->rows - 1);

				q4.x = xField.at<double>(y + 1, x);
				q4.y = yField.at<double>(y + 1, x);
			}

			// inside the grid proper
			else
			{
				q2.x = xField.at<double>(y, x + 1);
				q2.y = yField.at<double>(y, x + 1);


				q3.x = xField.at<double>(y + 1, x + 1);
				q3.y = yField.at<double>(y + 1, x + 1);

				q4.x = xField.at<double>(y + 1, x);
				q4.y = yField.at<double>(y + 1, x);
			}


			// rasterize point coordinates (happens automatically I think)
			//q1.x = cvRound(q1.x); q2.x = cvRound(q2.x); q3.x = cvRound(q3.x); q4.x = cvRound(q4.x);
			//q1.y = cvRound(q1.y); q2.y = cvRound(q2.y); q3.y = cvRound(q3.y); q4.y = cvRound(q4.y);


//#define CIRCLES 1
#ifdef CIRCLES
			cv::namedWindow("progress", cv::WINDOW_NORMAL);

			// draw grid corners.
			std::cout << q1 << std::endl;
			std::cout << q2 << std::endl;
			std::cout << q3 << std::endl;
			std::cout << q4 << std::endl << std::endl;

			cv::circle(imageElastic, q1, 1, cv::Scalar(0, 0, 255), 2);
			cv::circle(imageElastic, q2, 1, cv::Scalar(255, 0, 0), 2);
			cv::circle(imageElastic, q3, 1, cv::Scalar(0, 255, 0), 2);
			cv::circle(imageElastic, q4, 1, cv::Scalar(255, 0, 255), 2);
#endif

#define INTERPOLATE 1
#ifdef INTERPOLATE
			std::vector<cv::Point2d> cell = { q1, q2, q3, q4, q1 }; // contour of quad

			// get MBR of current grid cell
			double minX = fmin(q1.x, fmin(q2.x, fmin(q3.x, q4.x)));
			double maxX = fmax(q1.x, fmax(q2.x, fmax(q3.x, q4.x)));

			double minY = fmin(q1.y, fmin(q2.y, fmin(q3.y, q4.y)));
			double maxY = fmax(q1.y, fmax(q2.y, fmax(q3.y, q4.y)));

			// for each point in the grid shape's MBR that inside the cell:
			// Interpolate values via inverse bilinear interpolation
			for (double grid_y = q1.y; grid_y < maxY; grid_y++)
			{
				for (double grid_x = q1.x; grid_x < maxX; grid_x++)
				{
					cv::Point2d p = cv::Point2d(grid_x, grid_y); // interpolation point
					cv::Vec3d pixelVal; // pixel color

					if (p.x == 140 && p.y == 149)
						std::cout << "GOT EM" << std::endl;

					// is the point inside the irregular grid cell? If no, don't interpolate
					// (Points on the boundary count as inside the cell)
					//if(!isInsideConvexQuadrilateral(p, q1, q2, q3, q4))
					if (!winding_isInPolygon(p, cell))
					{
						//std::cout << contour << std::endl;
						continue;
					}


					// inverse bilinear interpolation for original image:
					// get u and v parameters for interpolation
					cv::Vec2d inv = invBilinearInterpolation(p, q1, q2, q3, q4);

					// calculate bilinearly interpolated value
					if (inv == cv::Vec2d(-1.0, -1.0))
					{
						// if no solution: do nearest neighbor interpolation
						pixelVal = nearestNeighborInterpolation(*image, p, q1, q2, q3, q4);
					}

					else
					{
						pixelVal = (1 - inv[0]) * ((1 - inv[1])
								* image->at<cv::Vec3d>(q1) + inv[1]
								* image->at<cv::Vec3d>(q2)) + inv[0]
								* ((1 - inv[1]) * image->at<cv::Vec3d>(q3) + inv[1]
								* image->at<cv::Vec3d>(q4));
					}

					// apply chosen color to output image
					imageElastic.at<cv::Vec3d>(p) = pixelVal;

					// Nearest-neighbor interpolation for label image so labels
					// remain solid colors instead of becoming gradients
					pixelVal = nearestNeighborInterpolation(*label, p, q1, q2, q3, q4);
					labelElastic.at<cv::Vec3d>(p) = pixelVal;
				}
			}
#endif 

			//cv::imshow("progress", imageElastic);
			//cv::waitKey(0);
		}

		std::cout << "Interpolation: Row " << y << " of " << numCellY - 1 << " done" << std::endl;
	}

	// assign changes to input matrices
	*image = imageElastic;
	*label = labelElastic;

	imageElastic.convertTo(imageElastic, CV_8UC3);
	labelElastic.convertTo(labelElastic, CV_8UC3);

	cv::namedWindow("elasticImage", cv::WINDOW_NORMAL);
	cv::imshow("elasticImage", imageElastic);

	cv::namedWindow("elasticLabel", cv::WINDOW_NORMAL);
	cv::imshow("elasticLabel", labelElastic);
}