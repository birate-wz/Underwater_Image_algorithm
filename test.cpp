#include<opencv2/opencv.hpp>
#include<omp.h>  //多线程
#include "evalute.h"
#include<iostream>
#include "msrcr.h"
#include <fstream>
#include "algor.h"
#include "SIHRUDCP.h"
#include <math.h>
#include <cv.h>
#include<time.h>
#include <ximgproc.hpp>
//#pragma omp parallel for

using namespace cv;
using namespace std;

Mat Hist(Mat& img) {  //直方图均衡算法
	Mat imgeRGB[3];
	split(img, imgeRGB);
	for (int i = 0; i < 3; i++) {
		equalizeHist(imgeRGB[i],imgeRGB[i]);
	}
	merge(imgeRGB,3,img);
	return img;
}
Mat HomoFilter(cv::Mat src) {  //同态滤波 效果太差
	src.convertTo(src, CV_64FC1);
	int rows = src.rows;
	int cols = src.cols;
	int m = rows % 2 == 1 ? rows + 1 : rows;
	int n = cols % 2 == 1 ? cols + 1 : cols;
	copyMakeBorder(src, src, 0, m - rows, 0, n - cols, BORDER_CONSTANT, Scalar::all(0));
	rows = src.rows;
	cols = src.cols;
	Mat dst(rows, cols, CV_64FC1);
	//1. ln
	for (int i = 0; i < rows; i++) {
		double *srcdata = src.ptr<double>(i);
		double *logdata = src.ptr<double>(i);
		for (int j = 0; j < cols; j++) {
			logdata[j] = log(srcdata[j] + 0.0001);
		}
	}
	//2. dct
	Mat mat_dct = Mat::zeros(rows, cols, CV_64FC1);
	dct(src, mat_dct);
	//3. 高斯同态滤波器
	Mat H_u_v;
	double gammaH = 1.5;
	double gammaL = 0.5;
	double C = 1;
	double  d0 = (src.rows / 2) * (src.rows / 2) + (src.cols / 2) * (src.cols / 2);
	double  d2 = 0;
	H_u_v = Mat::zeros(rows, cols, CV_64FC1);
	for (int i = 0; i < rows; i++) {
		double * dataH_u_v = H_u_v.ptr<double>(i);
		for (int j = 0; j < cols; j++) {
			d2 = pow(i, 2.0) + pow(j, 2.0);
			dataH_u_v[j] = (gammaH - gammaL) * (1 - exp(-C * d2 / d0)) + gammaL;
		}
	}
	H_u_v.ptr<double>(0)[0] = 1.1;
	mat_dct = mat_dct.mul(H_u_v);
	//4. idct
	idct(mat_dct, dst);
	//exp
	for (int i = 0; i < rows; i++) {
		double  *srcdata = dst.ptr<double>(i);
		double *dstdata = dst.ptr<double>(i);
		for (int j = 0; j < cols; j++) {
			dstdata[j] = exp(srcdata[j]);
		}
	}
	dst.convertTo(dst, CV_8UC1);
	return dst;
}
Mat Hvalue, Svalue, Ivalue;

Mat RGBtoHSI(Mat src) {
	Mat HSIImage = Mat(Size(src.cols, src.rows), CV_8UC3);

	vector<Mat> channels;
	split(HSIImage, channels);

	Hvalue = channels.at(0);
	Svalue = channels.at(1);
	Ivalue = channels.at(2);

	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			double H, S, I;
			int Bvalue = src.at<Vec3b>(i, j)[0];
			int Gvalue = src.at<Vec3b>(i, j)[1];
			int Rvalue = src.at<Vec3b>(i, j)[2];
			//求角度 根据公式
			double numerator = ((Rvalue - Gvalue) + (Rvalue - Bvalue)) / 2;
			double denominator = sqrt(pow((Rvalue - Gvalue), 2) + ((Rvalue - Bvalue)*((Gvalue - Bvalue))));
			if (denominator == 0) H = 0;
			else {
				double theta = acos(numerator / denominator) * 180 / 3.14;
				if (Bvalue <= Gvalue)
					H = theta;
				else
					H = 360 - theta;
			}
			Hvalue.at<uchar>(i, j) = (int)(H * 255 / 360); //将0-360 映射到0-255

														   //求S
			int minvalue = min(min(Bvalue, Gvalue), Rvalue);  //求三个通道的最小值
			numerator = minvalue * 3;
			denominator = Bvalue + Gvalue + Rvalue;
			if (denominator == 0) S = 0;
			else {
				S = 1 - numerator / denominator;
			}
			Svalue.at<uchar>(i, j) = (int)(S * 255);

			I = (Bvalue + Gvalue + Rvalue) / 3;    //增加I的值
			Ivalue.at<uchar>(i, j) = (int)I;
		}
	}
	merge(channels, HSIImage);
	return HSIImage;
}

//HSI转RGB空间
Mat HSItoRGB(Mat HSIImage,Mat src) {
	Mat RGBImage = Mat(Size(src.cols, src.rows), CV_8UC3);
	vector<Mat> channels;
	split(RGBImage, channels);
	Mat Rvalue = channels.at(0);
	Mat Gvalue = channels.at(1);
	Mat Bvalue = channels.at(2);

	int B, G, R;
	for (int i = 0; i < RGBImage.rows; ++i) {
		for (int j = 0; j < RGBImage.cols; ++j) {
			double  dH = HSIImage.at<Vec3b>(i, j)[0]; //色度
			double  dS = HSIImage.at<Vec3b>(i, j)[1]; //饱和度
			double  dI = HSIImage.at<Vec3b>(i, j)[2]; //亮度
			double dTempB, dTempG, dTempR;
			if (dH < 120 && dH >= 0) { //RG扇形
									   //将H转换为弧度显示
				dH = dH * 3.1415926 / 180;
				dTempB = dI*(1 - dS);
				dTempR = dI * (1 + (dS*cos(dH)) / cos(3.1415926 / 3 - dH));
				dTempG = (3 * dI - (dTempB + dTempR));
			}
			else if (dH < 240 && dH >= 120) { //GB扇形
				dH -= 120;
				//将H转为弧度显示
				dH = dH*3.1415926 / 180;
				dTempR = dI*(1 - dS);
				dTempG = dI*(1 + dS*cos(dH) / cos(3.1415926 / 3 - dH));
				dTempB = (3 * dI - (dTempR + dTempG));
			}
			else if (dH<360 && dH >= 240) { //BR扇形
				dH -= 240;
				//将H转为弧度显示
				dH = dH*3.1415926 / 180;
				dTempG = dI*(1 - dS);
				dTempB = dI*(1 + dS*cos(dH) / cos(3.1415926 / 3 - dH));
				dTempR = (3 * dI - (dTempB + dTempG));
			}
			B = (int)(dTempB  );
			G = (int)(dTempG  );
			R = (int)(dTempR );
			Bvalue.at<uchar>(i, j) = B;
			Gvalue.at<uchar>(i, j) = G;
			Rvalue.at<uchar>(i, j) = R;
		}
	}
	merge(channels, RGBImage);
	return RGBImage;
}
// 将HSI颜色空间的三个分量组合起来，便于显示
IplImage* catHSImage(CvMat* HSI_H, CvMat* HSI_S, CvMat* HSI_I)
{
	IplImage* HSI_Image = cvCreateImage(cvGetSize(HSI_H), IPL_DEPTH_8U, 3);

	for (int i = 0; i < HSI_Image->height; i++)
	{
		for (int j = 0; j < HSI_Image->width; j++)
		{
			double d = cvmGet(HSI_H, i, j);
			int b = (int)(d * 255 / 360);
			d = cvmGet(HSI_S, i, j);
			int g = (int)(d * 255);
			d = cvmGet(HSI_I, i, j);
			int r = (int)(d * 255);

			cvSet2D(HSI_Image, i, j, cvScalar(b, g, r));
		}
	}

	return HSI_Image;
}
//将RGB转换为HSI
CvMat* HSI_H, *HSI_S, *HSI_I;
void RGBTOHSI(IplImage* img) {
	// 三个HSI空间数据矩阵
	 HSI_H = cvCreateMat(img->height, img->width, CV_32FC1);
	 HSI_S = cvCreateMat(img->height, img->width, CV_32FC1);
	 HSI_I = cvCreateMat(img->height, img->width, CV_32FC1);
	// 原始图像数据指针, HSI矩阵数据指针
	uchar* data;

	// rgb分量
	unsigned char img_r, img_g, img_b;
	unsigned char min_rgb;  // rgb分量中的最小值
				   // HSI分量
	float fHue, fSaturation, fIntensity;

	for (int i = 0; i < img->height; i++)
	{
		for (int j = 0; j < img->width; j++)
		{
			data = cvPtr2D(img, i, j, 0);
			img_b = *data;
			data++;
			img_g = *data;
			data++;
			img_r = *data;

			// Intensity分量[0, 1]
			float temp = (float)((img_b + img_g + img_r) / 3) ;
			temp = temp*1.3 + 0.02;  //增强I的系数
			fIntensity = temp / 255;
			// 得到RGB分量中的最小值
			float fTemp = img_r < img_g ? img_r : img_g;
			min_rgb = fTemp < img_b ? fTemp : img_b;
			// Saturation分量[0, 1]
			fSaturation = 1 - (float)(3 * min_rgb) / (img_r + img_g + img_b);

			// 计算theta角
			float numerator = (img_r - img_g + img_r - img_b) / 2;
			float denominator = sqrt(
				pow((img_r - img_g), 2) + (img_r - img_b)*(img_g - img_b));

			// 计算Hue分量
			if (denominator != 0)
			{
				float theta = acos(numerator / denominator) * 180 / 3.14;

				if (img_b <= img_g)
				{
					fHue = theta;
				}
				else
				{
					fHue = 360 - theta;
				}
			}
			else
			{
				fHue = 0;
			}

			// 赋值
			cvmSet(HSI_H, i, j, fHue);
			cvmSet(HSI_S, i, j, fSaturation);
			cvmSet(HSI_I, i, j, fIntensity);
		}
	}
}
// 将HSI颜色模型的数据转换为RGB颜色模型的图像
IplImage* HSI2RGBImage(CvMat* HSI_H, CvMat* HSI_S, CvMat* HSI_I)
{
	IplImage * RGB_Image = cvCreateImage(cvGetSize(HSI_H), IPL_DEPTH_8U, 3);

	int iB, iG, iR;
	for (int i = 0; i < RGB_Image->height; i++)
	{
		for (int j = 0; j < RGB_Image->width; j++)
		{
			// 该点的色度H
			double dH = cvmGet(HSI_H, i, j);
			// 该点的色饱和度S
			double dS = cvmGet(HSI_S, i, j);
			// 该点的亮度
			double dI = cvmGet(HSI_I, i, j);

			double dTempB, dTempG, dTempR;
			// RG扇区
			if (dH < 120 && dH >= 0)
			{
				// 将H转为弧度表示
				dH = dH * 3.1415926 / 180;
				dTempB = dI * (1 - dS);
				dTempR = dI * (1 + (dS * cos(dH)) / cos(3.1415926 / 3 - dH));
				dTempG = (3 * dI - (dTempR + dTempB));
			}
			// GB扇区
			else if (dH < 240 && dH >= 120)
			{
				dH -= 120;

				// 将H转为弧度表示
				dH = dH * 3.1415926 / 180;

				dTempR = dI * (1 - dS);
				dTempG = dI * (1 + dS * cos(dH) / cos(3.1415926 / 3 - dH));
				dTempB = (3 * dI - (dTempR + dTempG));
			}
			// BR扇区
			else
			{
				dH -= 240;

				// 将H转为弧度表示
				dH = dH * 3.1415926 / 180;

				dTempG = dI * (1 - dS);
				dTempB = dI * (1 + (dS * cos(dH)) / cos(3.1415926 / 3 - dH));
				dTempR = (3 * dI - (dTempG + dTempB));
			}

			iB = dTempB * 255;
			iG = dTempG * 255;
			iR = dTempR * 255;

			cvSet2D(RGB_Image, i, j, cvScalar(iB, iG, iR));
		}
	}

	return RGB_Image;
}

/*CLAHE 算法*/
static void color_transfer_with_spilt(Mat& input, vector<Mat> &chls) {
	cvtColor(input,input,COLOR_BGR2YCrCb);
	split(input,chls);
}
static void color_retransfer_with_merge(Mat&output,vector<Mat>&chls) {
	merge(chls,output);
	cvtColor(output,output,COLOR_YCrCb2BGR);
}
Mat clahe_deal(Mat&src) {
	Mat ycrcb = src.clone();
	vector<Mat> channel;
	color_transfer_with_spilt(src, channel);

	Mat clahe_img;
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(4);
	clahe->setTilesGridSize(Size(8, 8)); //将图像分为8*8块
	clahe->apply(channel[0],clahe_img);
	channel[0].release();
	clahe_img.copyTo(channel[0]);
	color_retransfer_with_merge(ycrcb,channel);
	return ycrcb;
}
/*end*/
//gamma 变换
void MyGammaCorrection(Mat& src, Mat& dst, float fGamma)
{

	// build look up table  
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}

	dst = src.clone();
	const int channels = dst.channels();
	switch (channels)
	{
	case 1:
	{

		MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			//*it = pow((float)(((*it))/255.0), fGamma) * 255.0;  
			*it = lut[(*it)];

		break;
	}
	case 3: 
	{

		MatIterator_<Vec3b> it, end;
		for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
		{
			//(*it)[0] = pow((float)(((*it)[0])/255.0), fGamma) * 255.0;  
			//(*it)[1] = pow((float)(((*it)[1])/255.0), fGamma) * 255.0;  
			//(*it)[2] = pow((float)(((*it)[2])/255.0), fGamma) * 255.0;  
			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}
		break;
		}
	}
}
int main(int argc,char *argv[]){
	Mat img1, img2, imgdst, img3, img4,img5,img6;
	Msrcr msrcr;
	Evalue evalute;
	SID sid;
	char key;

	Mat imageGray;
	Mat imageSobel;
	ofstream file;
	file.open("data.txt", ios::trunc);   //写文件

	vector<double> sigema;
	vector<double> weight;
	for (int i = 0; i < 3; i++)
		weight.push_back(1. / 3);
	// 由于MSRCR.h中定义了宏USE_EXTRA_SIGMA，所以此处的vector<double> sigema并没有什么意义
	sigema.push_back(30);
	sigema.push_back(150);
	sigema.push_back(300);

	img1 = imread("C:\\Users\\wangzhen\\Desktop\\code\\datacollect\\image\\imgsrc.jpg");

	imshow("img1", img1);

	/*评价函数*/

	//cout << "getMSSIM" << evalute.getMSSIM(img1, img5) << endl;
	/**直方图均衡化*/
	/*stogram h;
	img2 = Hist(img1);
	imshow("Histogram1",img2);
	vector<cv::Mat> imgs = h.getHistogramImage(img2);
	imshow("B", imgs[0]);
	imshow("G", imgs[1]);
	imshow("R", imgs[2]);*/


	/* 同态滤波算法*/
	/*int originrows = img1.rows;
	int origincols = img1.cols;
	Mat dst(img1.rows, img1.cols, CV_8UC3);
	cvtColor(img1, img1, COLOR_BGR2YUV);
	vector <Mat> yuv;
	split(img1, yuv);
	Mat nowY = yuv[0];
	Mat newY = HomoFilter(nowY);
	Mat tempY(originrows, origincols, CV_8UC1);
	for (int i = 0; i < originrows; i++) {
		for (int j = 0; j < origincols; j++) {
			tempY.at<uchar>(i, j) = newY.at<uchar>(i, j);
		}
	}
	yuv[0] = tempY;
	merge(yuv, dst);
	cvtColor(dst, dst, COLOR_YUV2BGR);
	imshow("result", dst);*/

	//img2 = RGBtoHSI(img1);
	//imshow("HSI",img2);
	clock_t start, finish;
	double totaltime;   //开始时间
	
	    //  MSRCR+HSI增强
	msrcr.MultiScaleRetinexCR(img1,img2,weight,sigema,128,128);
	
	imshow("MSRCR", img2);
	IplImage* temp = &IplImage(img2);
	IplImage *img = cvCloneImage(temp);
	RGBTOHSI(img);       //RGB->HSI
	IplImage* HSI_Image = catHSImage(HSI_H, HSI_S, HSI_I);
	IplImage* RGB_Image = HSI2RGBImage(HSI_H, HSI_S, HSI_I);

	IplImage *img_rgb = HSI2RGBImage(HSI_H, HSI_S, HSI_I); //HSI->RGB
	img3 = cvarrToMat(img_rgb);
	//img4 = clahe_deal(img3);
	cvShowImage("HSI Color Model", HSI_Image);
    cvShowImage("RGB Color Model", RGB_Image);
	//imshow("CLAHE",img4);
	//finish = clock();     //结束时间
	//totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	//cout << "The run time is:" << totaltime << "s!" << endl;
	addWeighted(img2, 0.2, img3, 0.8, 0, img5);
	imwrite("MSRCR.jpg", img5);
	imshow("dst", img5);
	/*       SSR*/
	//msrcr.MultiScaleRetinex(img1,img2,weight,sigema);
	//msrcr.MultiScaleRetinexCR(img1, img2, weight, sigema);
	
	//float eps = 0.02 * 255 * 255;//eps的取值很关键（乘于255的平方）
	//cv::ximgproc::guidedFilter(img1, img1, img2,16, eps, -1);  //引导滤波算法
	//GaussianBlur(img1, img2, Size(15, 15), 11, 11);
//	cvtColor(img1, img2, COLOR_BGR2GRAY);
//	finish = clock();     //结束时间
//	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
//	cout << "The run time is:" << totaltime << "s!" << endl;*/

	
	//MyGammaCorrection(img1, img2, 1/1.5);
	//imshow("gamma",img2);
	//start = clock();
	img2 = sid.getdarkprio2(img1, 15, 0.9);
	//img3 = sid.getRecoverScene(img1);   //暗通道先验
	//finish = clock();     //结束时间
	
	imshow("1", img2);
	//imwrite("my.jpg", img2);
	//imshow("2", img3);
//	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
//	cout << "The run time is:" << totaltime << "s!" << endl;
	
	//imshow("拉伸后", img5);*/
	waitKey();
	return 0;
}