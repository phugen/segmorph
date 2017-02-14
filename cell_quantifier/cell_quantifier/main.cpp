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

	std::string outpath = "G:/CDEF_2013/CF/F_GNAGNA/Stuff/Studium/Master/WS_2016/Masterarbeit/cell_quantifier/Network/cell_quantifier/training_3augmented/";

	for (auto filename = fileNames.begin(); filename != fileNames.end(); filename++)
	{
		// ignore labelled images
		if ((*filename).find("label") == std::string::npos)
		{
			std::string replaceMe = *filename;
			std::string labelPath = replaceMe.insert(replaceMe.length() - 4, "_label");

			// perform data augmentation to increase number
			// of training samples
			std::vector<std::string> augPaths = augmentImageAndLabel(*filename, labelPath, outpath, 1000);
		}
	}


	

	// train U-Net
	//trainUnet(images, labels);

	// segment cells using trained U-Net
	//segmentImage(images);

	//cv::waitKey(0);
	return 0;
}