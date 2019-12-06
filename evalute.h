#pragma once
#ifndef __evalute_h__

#define __evalute_h__


#include<opencv2/opencv.hpp>
#include<omp.h>  //多线程
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;





class  Evalue
{
public:
	unsigned long getImageQualityAssessment(Mat &img);
	double Tenengrad(Mat& roi);
	double Brenner(Mat& image);
	double SMD(Mat& image);
	double SMD2(Mat& image);
	double Enery(Mat &image);
	double Jpeg(Mat& image);
	double Jpeg2(Mat& image);
	double Tenengrad_Diff(Mat &image);
	double Laplacian(Mat &image);
	double variance(Mat &image);
	double DefRto(Mat frame);
	Mat addSaltNoise(const Mat srcImage, int n);  //椒盐噪声
	double generateGaussianNoise(double mu, double sigma);
	Mat addGaussianNoise(Mat &srcImag);
	double Entropy(Mat img);   //信息熵
	Scalar getMSSIM(Mat  inputimage1, Mat inputimage2);  //SSIM
};

#endif // !__evalute_h__
