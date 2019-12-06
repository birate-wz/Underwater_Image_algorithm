#pragma once
#include<opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;
class SID    //single Image Dehazing
{
public:
	//SID() {};
	int **getMinChannel(cv::Mat img);
	int **getDarkChannel(int **img,int blockSize=3);
	cv::Mat getDarkImage(cv::Mat img, int blockSize=15);
	int getGlobalAtmosphericLightValue(int **darkChannel, cv::Mat img, float percent = 0.001); //估算全局大气值
	cv::Mat getRecoverScene(cv::Mat img, float omega = 0.95, float t0 = 0.1, int blockSize = 15, float percent = 0.001);


	Mat fastGuidedFilter(const cv::Mat &I_org, const cv::Mat &p_org, int r, double eps, int s);
	void get_darkchannel(const cv::Mat &input, cv::Mat &dark_channel, const int window_size);
	void get_atmosphere(const cv::Mat &image, const cv::Mat &dark_channel, cv::Vec3f &atmosphere);
	void get_esttransmap(const cv::Mat &image, const cv::Vec3f &atmosphere, const double &omega, const unsigned int win_size, cv::Mat &trans_est);
	void get_radiance(const cv::Mat&image, const cv::Mat&trans_map, const cv::Vec3f &atmosphere, cv::Mat &radiance);
	Mat getdarkprio2(Mat input_image, int win_size, double omega);
};


