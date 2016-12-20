#include "GMM.hpp"

// Finds <nGaussians> different regions in <image> using the grayscale image pixels
// as samples to train a Gaussian Mixture Model.
// Runs <iterations> times or until the difference to the last run is smaller than <EPS>.
cv::Mat segmentGMM(cv::Mat image, int nGaussians, int iterations, float eps)
{
	std::cout << std::endl << "Beginning GMM segmentation ... " << std::endl;

	cv::Mat image_boosted = cv::Mat(image.size(), image.type()); // image with added histogram equalization
	cv::Mat image_64f = cv::Mat(image.size(), CV_64FC1); // 64-bit version of original image
	cv::Mat samples = cv::Mat(image.rows * image.cols, 1, CV_64FC1); // matrix to contain samples from image; each row contains one sample: #pixels x 1 matrix.
	cv::Mat labels = cv::Mat(image.rows * image.cols, 1, CV_32SC1); // matrix to contain most likely label for each sample
	cv::Mat segmented = cv::Mat(image.size(), CV_8UC3); // output; segmentation into fore- and background
	cv::Mat eroded = cv::Mat(image.size(), CV_8UC3); // segmented matrix after erosion


	// transform input matrix to 64-bit double value matrix
	image.convertTo(image_64f, CV_64FC1, 1.0, 0.0);

	// extract pixels from 64-bit image and use them as samples
	// saved in a 1-channel column vector
	samples = image_64f.reshape(1, image.rows * image.cols);

	// define colors for labels
	uchar* colors;
	colors = (uchar*)malloc(sizeof(uchar)* nGaussians);

	for (int i = 0; i < nGaussians; i++)
	{
		// spread available color space evenly
		// among labels
		colors[i] = (uchar)cvRound(255 / (i + 1));
	}
	

	// create Gaussian Mixture Model
	cv::Ptr<cv::ml::EM> em_model = cv::ml::EM::create();

	em_model->setClustersNumber(nGaussians); // set number of gaussians / separate regions to find
	em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL); // matrix with positive diagonal elements
	em_model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, iterations, eps)); // stop after 300 iterations or when epsilon < 0.1
	

	std::cout << std::endl << "Starting  GMM training ... ";

	// train GMM on sample data
	em_model->trainEM(samples,
						cv::noArray(), // don't output additional likelihood values
						labels, // save most likely label for each pixel to #samples x 1 matrix
						cv::noArray()); // don't save additional posterior probablities

	std::cout << std::endl << "done." << std::endl;
	std::cout << std::endl << "Starting  GMM pixel classification ... ";

	// walk through image pixel by pixel and classify each pixel
	// as part of the back- or foreground gaussian distribution
	// classify every image pixel
	int samplesProcessed = 0;

	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			cv::Mat sample = cv::Mat(1, 1, CV_64FC1);
			sample.at<double>(0, 0) = image_64f.at<double>(y, x); // prepare one sample for classification

			/*int response = em_model->predict2(sample,
												cv::noArray()) // don't output posterior likelihoods for each component
												[1]; // first element is likelihood, second element is index*/

			segmented.at<cv::Vec3b>(cv::Point(x, y)) = colors[labels.at<int>(samplesProcessed, 0)];
			samplesProcessed++;
		}
	}

	std::cout << std::endl << "done." << std::endl;

	// show binary segmentation result
	cv::namedWindow("GMM Segmentation", cv::WINDOW_NORMAL);
	cv::imshow("GMM Segmentation", segmented);

	// remove small white artifacts by eroding the image
	/*int erosion_size = 1;
	cv::erode(segmented, eroded, getStructuringElement(cv::MORPH_RECT,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size)));*/
	cv::erode(segmented, eroded, cv::Mat());

	cv::namedWindow("Eroded", cv::WINDOW_NORMAL);
	cv::imshow("Eroded", eroded);

	free(colors);

	return eroded;
}
