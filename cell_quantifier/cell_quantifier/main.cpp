#include "IPANtest.hpp"
#include "SegmentationTest.hpp"
#include "auxiliary.hpp"
#include "GMM.hpp"
#include "Otsu.hpp"
#include "Kmeans.hpp"
#include "edge.hpp"
#include "graphCut.hpp"


int main(int argc, char* argv)
{
	std::vector<cv::Mat> grayscaleImages;
	cv::Mat matInput, matGrayscale, matHist, matHistNorm; // image matrix variables
	const std::string path = std::string("G:/CDEF_2013/CF/F_GNAGNA/Stuff/Studium/Master/WS_2016/Masterarbeit/Test_Data/"); // path to input images
	
	// load all images in folder specified by path
	std::vector<std::string> fileNames = listFilesInDirectory(path);


	// Create intensity value histogram
	// from all images
	/*int nimages = 1; // one image
	int dims = 1; // only one channel
	int channels[] = { 0 }; // which channel? -> intensity channel
	int histSize[] = { 256 }; // 256 bins
	float hranges[] = { 0, 256 }; // value range
	const float *ranges[] = { hranges };

	

	for (auto name = fileNames.begin(); name != fileNames.end(); name++)
	{
		std::cout << *name << std::endl;

		matInput = cv::imread(*name, CV_LOAD_IMAGE_COLOR);
		if (!matInput.data)
		{
			std::cout << "Could not open or find the image" << std::endl;
			cv::waitKey(0);
			exit(-1);
		}

		//cv::namedWindow("Initial image", cv::WINDOW_NORMAL);
		//cv::imshow("Initial image", matInput);
		//cv::waitKey(0);

		
		cv::cvtColor(matInput, matGrayscale, cv::COLOR_BGR2GRAY, 0); // convert to grayscale image
		cv::calcHist(&matGrayscale, nimages, channels, cv::Mat(), matHist, dims, histSize, ranges, true, true); // accumulation histogram
	}*/


	// histogram specs
	/*int hist_w = 512; // width
	int hist_h = 400; // height 
	int bin_w = cvRound((double) hist_w / histSize[0]);

	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

	// normalize the result to fit histogram width
	normalize(matHist, matHistNorm, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

	// draw histogram bars
	for (int i = 1; i < histSize[0]; i++)
	{
		float value = matHistNorm.at<float>(i);
		line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(matHistNorm.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(value)),
			cv::Scalar(0, 0, 0), 2, 8, 0);
	}

	// show accumulated histogram of all images
	cv::namedWindow("Accumulated histogram", cv::WINDOW_NORMAL);
	cv::imshow("Accumulated histogram", histImage);*/

	// TODO: automatic histogram analysis here? find order of curve / number of local maxima
	// and derive number of gaussians from it
	
	//cv::namedWindow("Original image", cv::WINDOW_NORMAL);
	//cv::imshow("Original image", matGrayscale);

	cv::Mat segmented;
	std::string GTPath = std::string("G:/CDEF_2013/CF/F_GNAGNA/Stuff/Studium/Master/WS_2016/Masterarbeit/Test_Data_Labelled/WT_(2)_label.bmp"); // path to ground truth for test image

	// GMM segmentation
	/*segmented = segmentGMM(matInput, 2, 300, 0.00001); // binary segmentation with two-gaussian GMM
	overlayFoundBorders(GTPath, segmented, std::string("GMM-2 overlay")); // show GMM borders on original image

	// binary segmentation by Otsu thresholding
	segmented = segmentOtsu(matInput); 
	overlayFoundBorders(GTPath, segmented, std::string("Otsu overlay"));

	// binary segmentation by k-means (k = 2)
	segmented = segmentKmeans(matInput, 5, 1); 
	overlayFoundBorders(GTPath, segmented, std::string("k-means (5) overlay"));

	// edge detection segmentation
	segmented = segmentEdge(matInput, 3);
	overlayFoundBorders(GTPath, segmented, std::string("Canny w/ Otsu thresh overlay"));*/


	// TODO: conservative segmentation, then use connected components' 
	// central points as starting points for active contour 

	
	// TODO: Max Flow Min Cut applied to images; images as a graph.
	// https://en.wikipedia.org/wiki/Graph_cuts_in_computer_vision : Standard vs. Iterated vs Dynamic?
	// see German max-flow min-cut article for nice proof of theorem
	//segmented = segmentGraphcut(matInput, 5);
	//overlayFoundBorders(GTPath, segmented, std::string("graphcut overlay"));

	std::string ipath = "G:/CDEF_2013/CF/F_GNAGNA/Stuff/Studium/Master/WS_2016/Masterarbeit/Test_Data/vincent_van_gogh.jpg";
	std::string labelPath = "G:/CDEF_2013/CF/F_GNAGNA/Stuff/Studium/Master/WS_2016/Masterarbeit/Test_Data/vincent_van_gogh_label.jpg";

	augmentImageAndLabel(ipath, labelPath, 0, 0);

	cv::waitKey(0);
	return 0;
}