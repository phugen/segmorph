#include "otsu.hpp"
#include <iostream>

// Do binary segmentation of an image by using Otsu's thresholding
// technique. Foreground pixels are colored white.
cv::Mat segmentOtsu(cv::Mat image)
{
	cv::Mat matGrayscale = cv::Mat(image.size(), image.type());
	cv::Mat segmented = cv::Mat(image.size(), image.type());

	// convert to grayscale matrix
	cv::cvtColor(image, matGrayscale, cv::COLOR_BGR2GRAY, 0);


	// boost brightness of pixels with gray intensity >= cutoff
	// which is manually determined for cell material (TODO: do this automatically? Histogram analysis?)
	/*cv::Mat image_boosted = matGrayscale.clone();
	uchar cutoff = 10;

	for (int y = 0; y < matGrayscale.rows; y++)
	{
		for (int x = 0; x < matGrayscale.cols; x++)
		{
			uchar new_intensity;
			uchar intensity = matGrayscale.at<uchar>(y, x);
			//uchar blue = intensity[0];
			//uchar green = intensity[1];
			//uchar red = intensity[2];

			if (intensity >= cutoff)
			{
				// uchar new_blue;
				// uchar new_green;
				// uchar new_red;

				// prevent overflow if pixel too bright
				//blue + 10 < 256 ? new_blue = blue + 10 : new_blue = 255;
				//green + 10 < 256 ? new_green = green + 10 : new_green = 255;
				//red + 10 < 256 ? new_red = red + 10 : new_red = 255;
				intensity + 10 < 256 ? new_intensity = intensity + 10 : new_intensity = 255;

				//image_boosted.at<cv::Vec3b>(y, x) = cv::Vec3b(new_blue, new_green, new_red);
				image_boosted.at<uchar>(y, x) = new_intensity;
			}
		}
	}

	cv::namedWindow("Boosted", cv::WINDOW_NORMAL);
	cv::imshow("Boosted", image_boosted);

	cv::waitKey(0);*/



	// apply Otsu's threshold
	cv::threshold(matGrayscale, segmented, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	// show result
	cv::namedWindow("Otsu's threshold");
	cv::imshow("Otsu's threshold", segmented);


	return segmented;
}