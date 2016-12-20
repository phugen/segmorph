#include "kmeans.hpp"
#include <iostream>
#include <cstdint>

cv::Mat segmentKmeans(cv::Mat image, int k, int iterations)
{
	// convert input image to float depth
	cv::Mat image_32f, labels, samples;
	image.convertTo(image_32f, CV_32F);

	// convert data into column vector
	samples = image_32f.reshape(1, image.rows * image.cols); // sample matrix: 1 row per sample, 1 col per feature per sample (here: only 1 feature)

	// define colors for labels
	uchar* colors;
	colors = (uchar*)malloc(sizeof(uchar) * k);

	for (int i = 0; i < k; i++)
	{
		// spread available color space evenly
		// among labels
		colors[i] = (uchar) cvRound(255 / (i + 1));
	}

	// start K-means algorithm
	std::cout << "Starting k-means algorithm ... ";
	kmeans(samples, k, labels, cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 1E+10, 1E-300), iterations, cv::KMEANS_PP_CENTERS);
	std::cout << " done." << std::endl;

	// walk through image pixel by pixel and classify each pixel
	// as part of the back- or foreground based on k-means label
	std::cout << "Starting k-means classification ... ";

	cv::Mat segmented = cv::Mat(image.size(), CV_8U);

	int samplesProcessed = 0;
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++, samplesProcessed++)
		{
			// retrieve color associated with label of current pixel
			segmented.at<uchar>(cv::Point(x, y)) = colors[labels.at<int>(samplesProcessed, 0)];
		}
	}
	std::cout << " done." << std::endl;

	// show k-means result
	cv::namedWindow("K-Means", cv::WINDOW_NORMAL);
	cv::imshow("K-Means", segmented);

	free(colors);

	return segmented;
}