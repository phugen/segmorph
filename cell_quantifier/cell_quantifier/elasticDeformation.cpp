#include "elasticDeformation.hpp"

/*
Implements elastic deformation of an image as described in
"Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis." by Patrice Y. Simard, Dave Steinkraus, John C. Platt.

Alpha controls the strength of the deformation, while sigma changes the smoothing
factor of the Gaussian distribution. The grid size determines how many
grid nodes there will be; gridSize = 3 would result in a 3x3 grid on the image.
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
	std::random_device rdev;
	uint64_t seed = (uint64_t(rdev()) << 32) | rdev();
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

	labelElastic = cv::Vec3f(255, 255, 0);

	//std::cout << "Channels = " << imageElastic.channels() << ", Type = " << imageElastic.type() << std::endl;
	//std::cout << "cols = " << imageElastic.cols << ", rows = " << imageElastic.rows << std::endl << std::endl;

	// calculate grid point displacement coordinates
	for (int y = 0; y < numCellY; y++)
	{
		for (int x = 0; x < numCellX; x++)
		{
			double xcoord, ycoord;

			// non-clamped image coordinates: use readSafe / writeSafe
			// to handle OOB situations!
			xcoord = x * gridSize + xField.at<double>(cv::Point2d(x, y));
			ycoord = y * gridSize + yField.at<double>(cv::Point2d(x, y));

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


			// rasterize point coordinates
			//q1.x = cvRound(q1.x); q2.x = cvRound(q2.x); q3.x = cvRound(q3.x); q4.x = cvRound(q4.x);
			//q1.y = cvRound(q1.y); q2.y = cvRound(q2.y); q3.y = cvRound(q3.y); q4.y = cvRound(q4.y);


#define INTERPOLATE 1
#ifdef INTERPOLATE
			// get MBR of current grid cell
			double minX = fmin(q1.x, fmin(q2.x, fmin(q3.x, q4.x)));
			double maxX = fmax(q1.x, fmax(q2.x, fmax(q3.x, q4.x)));

			double minY = fmin(q1.y, fmin(q2.y, fmin(q3.y, q4.y)));
			double maxY = fmax(q1.y, fmax(q2.y, fmax(q3.y, q4.y)));

			// for each point in the grid shape's MBR that's inside the cell:
			// Interpolate values via inverse bilinear interpolation
			for (double grid_y = minY; grid_y < maxY; grid_y++)
			{
				for (double grid_x = minX; grid_x < maxX; grid_x++)
				{
					//cv::Point2d p = cv::Point2d(grid_x, grid_y); // interpolation point
					cv::Vec3d pixelVal; // pixel color

					// if grid only 1x1: don't interpolate
					if (gridSize == 1)
					{
 						imageElastic.at<cv::Vec3d>(y, x) = readSafe<cv::Vec3d>(image, grid_y, grid_x);
						labelElastic.at<cv::Vec3d>(y, x) = readSafe<cv::Vec3d>(label, grid_y, grid_x);

						continue;
					}

					cv::Point2d p = cv::Point2d(grid_x, grid_y);


//#define BARYCENTRIC
#ifdef BARYCENTRIC
					std::vector<cv::Point2d*> cell = { &q1, &q2, &q3, &q4, &q1 }; // contour of tri1

					// split quad into two triangles
					// TODO: split intelligently?
					std::vector<cv::Point2d*> tri1 = { &q1, &q2, &q3, &q1 };
					std::vector<cv::Point2d*> tri2 = { &q1, &q3, &q4, &q1 };

					// Is p inside triangle 1 or 2? If yes,
					// perform barycentric interpolation for p
					// (Points on the boundary count as inside the triangle)
					if (winding_isInPolygon(&p, tri1))
					{
						pixelVal = barycentricInterpolation(*image, &p, &q1, &q2, &q3);
					}

					else if (winding_isInPolygon(&p, tri2))
					{
						pixelVal = barycentricInterpolation(*image, &p, &q1, &q3, &q4);
					}

					// If not, don't interpolate
					else
					{
						continue;
					}
#endif



					
#ifdef BILINEAR
					if (!winding_isInPolygon(&p, cell))
					{
						continue;
					}

					// inverse bilinear interpolation for original image:
					// get u and v parameters for interpolation
					cv::Vec2d inv = invBilinearInterpolation(*image, p, q1, q2, q3, q4);

					// calculate bilinearly interpolated value
					if (inv == cv::Vec2d(-1.0, -1.0))
					{
						// if no solution: do nearest neighbor interpolation
						pixelVal = cv::Vec3d(0, 255, 0); // nearestNeighborInterpolation(*image, p, q1, q2, q3, q4);
					}

					else
					{
						pixelVal = (1 - inv[0]) * ((1 - inv[1])
							* readSafe<cv::Vec3d>(image, q1.y, q1.x) + inv[1]
							* readSafe<cv::Vec3d>(image, q2.y, q2.x)) + inv[0]
							* ((1 - inv[1]) * readSafe<cv::Vec3d>(image, q3.y, q3.x) + inv[1]
							* readSafe<cv::Vec3d>(image, q4.y, q4.x));
					}
#endif

					// apply chosen color to output image
					writeSafe<cv::Vec3d>(&imageElastic, grid_y, grid_x, pixelVal);

					// Nearest-neighbor interpolation for label image so labels
					// remain solid colors instead of becoming gradients
					pixelVal = nearestNeighborInterpolation(*label, &p, &q1, &q2, &q3, &q4);
					writeSafe<cv::Vec3d>(&labelElastic, grid_y, grid_x, pixelVal);
				}
			}
#endif 
		}

		//std::cout << "Interpolation: Row " << y << " of " << numCellY - 1 << " done" << std::endl;
	}

	//std::cout << std::endl;

	// assign changes to input matrices
	*image = imageElastic;
	*label = labelElastic;

	imageElastic.convertTo(imageElastic, CV_8UC3);
	labelElastic.convertTo(labelElastic, CV_8UC3);

	/*cv::namedWindow("elasticImage", cv::WINDOW_NORMAL);
	cv::imshow("elasticImage", imageElastic);

	cv::namedWindow("elasticLabel", cv::WINDOW_NORMAL);
	cv::imshow("elasticLabel", labelElastic);*/
}