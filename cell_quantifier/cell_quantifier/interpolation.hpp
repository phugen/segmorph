#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

cv::Vec3d nearestNeighborInterpolation(cv::Mat image, cv::Point2d* p, cv::Point2d* a, cv::Point2d* b, cv::Point2d* c, cv::Point2d* d);
cv::Vec3d barycentricInterpolation(cv::Mat image, cv::Point2d* p, cv::Point2d* a, cv::Point2d* b, cv::Point2d* c);
float bilinearInterpolation();
cv::Vec2d invBilinearInterpolation(cv::Mat image, cv::Point2d* p, cv::Point2d* a, cv::Point2d* b, cv::Point2d* c, cv::Point2d* d);
