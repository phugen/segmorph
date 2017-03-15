#include "IPANtest.hpp"
#include "SegmentationTest.hpp"
#include "auxiliary.hpp"
#include "GMM.hpp"
#include "Otsu.hpp"
#include "Kmeans.hpp"
#include "edge.hpp"
#include "graphCut.hpp"
#include "unet.hpp"



int main(int argc, char* argv)
{
	std::vector<cv::Mat> grayscaleImages;
	cv::Mat matInput, matGrayscale, matHist, matHistNorm; // image matrix variables
	const std::string path = std::string("G:/CDEF_2013/CF/F_GNAGNA/Stuff/Studium/Master/WS_2016/Masterarbeit/cell_quantifier/Network/cell_quantifier/training_2png/"); // path to input images
	
	// load all images in folder specified by path
	std::vector<std::string> fileNames = listFilesInDirectory(path);


	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9); // maximum quality


	// TODO: automatic histogram analysis here? find order of curve / number of local maxima
	// and derive number of gaussians from it
	
	//cv::namedWindow("Original image", cv::WINDOW_NORMAL);
	//cv::imshow("Original image", matGrayscale);

	cv::Mat segmented;
	//std::string GTPath = std::string("latex_test.png"); // path to ground truth for test image
	matInput = cv::imread("latex_test.png");

	if (matInput.data == NULL)
	{
		exit(-1);
	}

	// binary segmentation by k-means (k = 4)
	cv::Mat kmeans = segmentKmeans(matInput, 4, 1);
	cv::imwrite("latex_test_label_kmeans.png", kmeans, compression_params);
	//overlayFoundBorders(GTPath, segmented, std::string("k-means (5) overlay"));

	// GMM segmentation
	cv::Mat gmm = segmentGMM(matInput, 4, 300, 0.00001); // binary segmentation with two-gaussian GMM
	cv::imwrite("latex_test_label_GMM.png", gmm, compression_params);
	//overlayFoundBorders(GTPath, segmented, std::string("GMM-4 overlay")); // show GMM borders on original image

	// binary segmentation by Otsu thresholding
	/*segmented = segmentOtsu(matInput); 
	overlayFoundBorders(GTPath, segmented, std::string("Otsu overlay"));

	// edge detection segmentation
	segmented = segmentEdge(matInput, 3);
	overlayFoundBorders(GTPath, segmented, std::string("Canny w/ Otsu thresh overlay"));*/

	
	// TODO: Max Flow Min Cut applied to images; images as a graph.
	// https://en.wikipedia.org/wiki/Graph_cuts_in_computer_vision : Standard vs. Iterated vs Dynamic?
	// see German max-flow min-cut article for nice proof of theorem
	//segmented = segmentGraphcut(matInput, 5);
	//overlayFoundBorders(GTPath, segmented, std::string("graphcut overlay"));

	std::string outpath = "G:/CDEF_2013/CF/F_GNAGNA/Stuff/Studium/Master/WS_2016/Masterarbeit/cell_quantifier/Network/cell_quantifier/training_3augmented/";

	int index = 1;
	int imgno = fileNames.size() / 2;


	for (auto filename = fileNames.begin(); filename != fileNames.end(); filename++)
	{
		std::string checkme = *filename;

		// deal with labelled images inside implicitly 
		// in augmentImageAndLabel and therefore ignore them here
		if (checkme.find("label") == std::string::npos)
		{
			std::string replaceMe = *filename;
			std::string labelPath = replaceMe.insert(replaceMe.length() - 4, "_label");

			// perform data augmentation to increase number
			// of training samples
			//std::vector<std::string> augPaths = augmentImageAndLabel(*filename, labelPath, outpath, 1000);

			//std::cout << "Augmented image " << index << " / " << imgno << "!\n";
			index++;
		}
	}


	return 0;
}