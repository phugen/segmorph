#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

cv::Vec3d nearestNeighborInterpolation(cv::Mat image, cv::Point2d p, cv::Point2d a, cv::Point2d b, cv::Point2d c, cv::Point2d d);
float bilinearInterpolation();
cv::Vec2d invBilinearInterpolation(cv::Point2d p, cv::Point2d a, cv::Point2d b, cv::Point2d c, cv::Point2d d);
