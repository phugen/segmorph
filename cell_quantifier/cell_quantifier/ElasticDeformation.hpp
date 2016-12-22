#include "auxiliary.hpp"


void elasticDeformation(cv::Mat* image, cv::Mat* label, int gridSize, double sigma, double alpha);


// Accesses an OpenCV matrix safely, guaranteeing that the access
// will never be OOB. Returns a value depending on the access mode.  
// The mode can be set to ACCESS_MIRROR, ACCESS_NEUTRAL or ACCESS_CLAMP.
template <typename T> T readSafe(cv::Mat* mat, int y, int x, int mode)
{
	// coordinates within bounds, normal access
	if (y >= 0 && y < mat->rows &&
		x >= 0 && x < mat->cols)
	{
		return mat->at<T>(y, x);
	}

	// Torus-like access behavior.
	else if (mode == ACCESS_MIRROR)
	{
		double new_y = fmod_custom(y, mat->rows - 1);
		double new_x = fmod_custom(x, mat->cols - 1);

		return mat->at<T>(new_y, new_x);
	}


	// OOB = constant, neutral value
	else if (mode == ACCESS_NEUTRAL)
	{
		if (x < 0 || x > mat->cols - 1 || y < 0 || y > mat->rows - 1)
		{
			return NEUTRAL;
		}

		else
			return mat->at<T>(y, x);
	}

	// OOB = nearest, valid value
	else if (mode == ACCESS_CLAMP)
	{
		double new_y = clamp(y, 0, mat->rows - 1);
		double new_x = clamp(x, 0, mat->cols - 1);

		return mat->at<T>(new_y, new_x);
	}

	// unknown mode
	else
		return -1;
}


// Writes to an OpenCV matrix safely, guaranteeing that the access
// will never be OOB. Returns 0 upon success and -1 if the mode is unknown. 
// The mode can be set to ACCESS_MIRROR or ACCESS_CLAMP.
template <typename T> int writeSafe(cv::Mat* mat, int y, int x, T val, int mode)
{
	// Torus-like access behavior.
	if (mode == ACCESS_MIRROR)
	{
		double ynew = fmod_custom(y, mat->rows - 1);
		double xnew = fmod_custom(x, mat->cols - 1);

		mat->at<T>(ynew, xnew) = val;

		return 0;
	}


	// OOB = nearest, valid value
	else if (mode == ACCESS_CLAMP)
	{
		double ynew = clamp(y, 0, mat->rows - 1);
		double xnew = clamp(x, 0, mat->cols - 1);

		mat->at<T>(ynew, xnew) = val;
		return 0;
	}

	// unknown mode
	else
		return -1;
}