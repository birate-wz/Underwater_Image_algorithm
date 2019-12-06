#include "SIHRUDCP.h"
#include <ximgproc.hpp>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int rows, cols;
struct node
{
	int x, y, val;
	node() {};
	node(int _x, int _y, int _val) :x(_x), y(_y), val(_val) {}
	bool operator<(const node &rhs) {
		return  val>rhs.val;
	}
};

int ** SID::getMinChannel(cv::Mat img)   //求3个通道的最小值 存储在二维mat中
{
	rows = img.rows;
	cols = img.cols;

	if (img.channels() < 3) {
		printf("Input Error");
		exit(-1);
	}
	int **imageGray;
	imageGray = new int* [rows];
	for (int i = 0; i < rows; i++) {
		imageGray[i] = new int [cols];
	}
	
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int min = 255;
			for (int k = 0; k < 3; k++) {
				if (img.at<Vec3b>(i, j)[k] < min)
					min = img.at<Vec3b>(i, j)[k];
			}
			imageGray[i][j] = min;
		}
	}
	return imageGray;
}

int ** SID::getDarkChannel(int ** img, int blockSize)
{
	if (blockSize % 2 == 0 || blockSize < 3) {
		printf("blockSize is not odd or too small!");
		exit(-1);
	}
	int poolSize = (blockSize - 1) / 2;   //半径
	int newHight = rows + poolSize - 1;
	int newWidth = cols + poolSize - 1;
	int **imageMiddle;
	imageMiddle = new int *[newHight];
	for (int i = 0; i < newHight; i++)
		imageMiddle[i] = new int[newWidth];

	for (int i = 0; i < newHight; i++) {
		for (int j = 0; j < newWidth; j++) {
			if (i < rows && j < cols)
				imageMiddle[i][j] = img[i][j];
			else
				imageMiddle[i][j] = 255;
		}
	}
	int **imageDark;   //求暗通道最小值滤波
	imageDark = new int*[rows];
	for (int i = 0; i < rows; i++)
		imageDark[i] = new int[cols];

	for (int i = poolSize; i < newHight - poolSize; i++) {
		for (int j = poolSize; j < newWidth - poolSize; j++) {
			int min = 255;
			for (int k = i - poolSize; k < i + poolSize + 1; k++) {
				for (int l = j - poolSize; l < j + poolSize + 1; l++) {
					if (imageMiddle[k][l] < min)
						min = imageMiddle[k][l];
				}
			}
			imageDark[i - poolSize][j - poolSize] = min;
		}
	}

	return imageDark;
}
cv::Mat SID::getDarkImage(cv::Mat img,int blockSize) {   //得到暗通道 颜色的图像
	cv::Mat dark(img.rows, img.cols, CV_8UC3);
	int **imageGray = getMinChannel(img);
	int **imageDark = getDarkChannel(imageGray, blockSize);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			for (int channel = 0; channel < 3; channel++) {
				if (imageDark[i][j] > 255)
					dark.at<Vec3b>(i, j)[channel] = 255;
				else if(imageDark[i][j] < 0)
					dark.at<Vec3b>(i, j)[channel] = 0;
				else 
					dark.at<Vec3b>(i, j)[channel] = imageDark[i][j];
			}
		}
	}
	return dark;
}
int SID::getGlobalAtmosphericLightValue(int ** darkChannel, cv::Mat img, float percent)
{
	int size = rows*cols;
	vector<node> nodes;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			node temp;
			temp.x = i;
			temp.y = j;
			temp.val = darkChannel[i][j];
			nodes.push_back(temp);
		}
	}
	sort(nodes.begin(),nodes.end());
	int atmosphericLight = 0;
	if (int(percent*size) == 0) {  //求边缘值
		for (int i = 0; i < 3; i++) {
			if (img.at<Vec3b>(nodes[0].x, nodes[0].y)[i] > atmosphericLight)
				atmosphericLight = img.at<Vec3b>(nodes[0].x, nodes[0].y)[i];
		}
	}
	//获取暗通道在前0.1%的位置的像素点在原图像中的最高亮度值
	for (int i = 0; i < int(percent*size); i++) {
		for (int j = 0; j < 3; j++) {
			if (img.at<Vec3b>(nodes[i].x, nodes[i].y)[j] > atmosphericLight) {
				atmosphericLight = img.at<Vec3b>(nodes[i].x, nodes[i].y)[j];
			}
		}
	}
	return atmosphericLight;
}

cv::Mat SID::getRecoverScene(cv::Mat img, float omega , float t0 , int blockSize , float percent)
{
	int **imageGray = getMinChannel(img);
	int **imageDark = getDarkChannel(imageGray,blockSize);
	int atmosphericLight= getGlobalAtmosphericLightValue(imageDark, img);

	float **transmission;
	transmission = new float*[rows];
	for (int j = 0; j < rows; j++)
		transmission[j] = new float[cols+1];

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			transmission[i][j] = 1 - omega * imageDark[i][j] / atmosphericLight;
			if (transmission[i][j] < t0)
				transmission[i][j] = t0;
		}
	}
	//采用引导滤波 细化
	/*Mat guidimge(img.rows, img.cols, CV_8UC3);
	for (int channel = 0; channel < 3; channel++) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				guidimge.at<Vec3b>(i, j)[channel] = transmission[i][j];
			}
		}
	}
	Mat image_gray(img.rows, img.cols, CV_8UC1);
	cv::cvtColor(img, image_gray, CV_BGR2GRAY);
	Mat img2(img.rows, img.cols, CV_8UC3);
	float eps = 0.02 * 255 * 255;//eps的取值很关键（乘于255的平方）
	//img2=fastGuidedFilter(image_gray, guidimge, 15, 0.001, 1);
	cv::ximgproc::guidedFilter(guidimge, image_gray, img2, blockSize, 0.001, -1);  //引导滤波算法
	for (int channel = 0; channel < 3; channel++) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				transmission[i][j] = img2.at<Vec3b>(i, j)[channel];
				
			}
		}
	}*/
	//恢复原图像
	cv::Mat dst(img.rows, img.cols, CV_8UC3);
	for (int channel = 0; channel < 3; channel++) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				int temp = (img.at<Vec3b>(i, j)[channel] - atmosphericLight) / transmission[i][j] + atmosphericLight;
				if (temp > 255) temp = 255;
				if (temp < 0) temp = 0;
				dst.at<Vec3b>(i, j)[channel] = temp;
			}
		}
	}
	//delete transmission;
	return dst;
}



/*基于导向滤波的暗通道先验*/
void SID::get_darkchannel(const cv::Mat &input, cv::Mat &dark_channel, const int window_size)
{
	cv::Mat RGB_channel[3];//三通道，分别是B,G,R  OpenCV中顺序相反
	cv::split(input, RGB_channel);//将原图分离到三通道

	cv::Mat min_channel;//三个颜色通道中最小值
	min_channel = cv::min(cv::min(RGB_channel[2], RGB_channel[1]), RGB_channel[0]);

	cv::Matx<unsigned char, 15, 15> element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(window_size, window_size));
	//进行腐蚀操作
	cv::erode(min_channel, dark_channel, element);
}
void SID::get_atmosphere(const cv::Mat &image, const cv::Mat &dark_channel, cv::Vec3f &atmosphere)
{
	unsigned int m{ 0 }, n{ 0 };
	m = image.rows;
	n = image.cols;
	unsigned int n_pixels = n * m;
	unsigned int n_search_pixels = n_pixels * 0.01;
	cv::Mat dark_vec = dark_channel.reshape(0, 1);//将暗通道变为行向量形式
	cv::Mat image_vec = image.reshape(3, 1);//将原图转换为行向量

											//对暗通道图进行排序得到暗通道图索引
	cv::Mat indices(1, n_pixels, CV_8UC1);//存储将暗通道排序后得到的索引矩阵
	cv::sortIdx(dark_vec, indices, CV_SORT_DESCENDING);//对暗通道进行排序
	atmosphere = cv::Vec3f(0, 0, 0);//用于存储大气光值
	cv::Vec3f temp;
	unsigned int temp1;
	for (unsigned int i = 0; i < n_search_pixels; i++)//此处采用迭代，可能会降低运行效率
	{
		temp1 = indices.at<int>(i);
		temp = image_vec.at<cv::Vec3f>(temp1);
		atmosphere = atmosphere + temp;
	}
	atmosphere[0] /= n_search_pixels;//B通道大气光
	atmosphere[1] /= n_search_pixels;//G通道大气光
	atmosphere[2] /= n_search_pixels;//R通道大气光
}
void SID::get_esttransmap(const cv::Mat &image, const cv::Vec3f &atmosphere, const double &omega, const unsigned int win_size, cv::Mat &trans_est)
{
	unsigned int m{ 0 }, n{ 0 };
	m = image.rows;
	n = image.cols;
	//获得扩展大气光矩阵
	cv::Mat atmosphere_Matx(m, n, CV_32FC3, cv::Scalar(atmosphere[0], atmosphere[1], atmosphere[2]));
	cv::Mat dark_sub;
	get_darkchannel(image / atmosphere_Matx, dark_sub, win_size);
	trans_est = 1 - (omega*dark_sub);
}
void  SID::get_radiance(const cv::Mat&image, const cv::Mat&trans_map, const cv::Vec3f &atmosphere, cv::Mat &radiance)
{
	unsigned int m{ 0 }, n{ 0 };
	m = image.rows;
	n = image.cols;
	//得到大气光矩阵（将三个通道的大气光进行拷贝）
	cv::Mat rep_atmosphere(m, n, CV_32FC3, cv::Scalar(atmosphere[0], atmosphere[1], atmosphere[2]));

	cv::Mat max_transmap = cv::max(trans_map, 0.1);//设置大气光传递矩阵最小值为0.1（防止除零主导）

												   //将大气光传递矩阵复制到三个通道(默认三个通道采用同一大气光)
	vector<cv::Mat>trans_map_final;
	cv::Mat trans_3channel;
	trans_map_final.push_back(trans_map);
	trans_map_final.push_back(trans_map);
	trans_map_final.push_back(trans_map);
	cv::merge(trans_map_final, trans_3channel);

	//采用大气光照模型计算需要恢复的图像
	radiance = ((image - rep_atmosphere) / trans_3channel) + rep_atmosphere;
}
Mat SID::fastGuidedFilter(const cv::Mat &I_org, const cv::Mat &p_org, int r, double eps, int s)
{
	/*
	% GUIDEDFILTER   O(N) time implementation of guided filter.
	%
	%   - guidance image: I (should be a gray-scale/single channel image)
	%   - filtering input image: p (should be a gray-scale/single channel image)
	%   - local window radius: r
	%   - regularization parameter: eps
	*/

	cv::Mat I, _I;
	I_org.convertTo(_I, CV_32FC1, 1.0);

	resize(_I, I, I.size(), 1.0 / s, 1.0 / s, 1);


	cv::Mat p, _p;
	p_org.convertTo(_p, CV_32FC1, 1.0);
	//p = _p;
	resize(_p, p, p.size(), 1.0 / s, 1.0 / s, 1);

	//[hei, wid] = size(I);    
	int hei = I.rows;
	int wid = I.cols;

	r = (2 * r + 1) / s + 1;//因为opencv自带的boxFilter（）中的Size,比如9x9,我们说半径为4   

							//mean_I = boxfilter(I, r) ./ N;    
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, CV_32FC1, cv::Size(r, r));

	//mean_p = boxfilter(p, r) ./ N;    
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, CV_32FC1, cv::Size(r, r));

	//mean_Ip = boxfilter(I.*p, r) ./ N;    
	cv::Mat mean_Ip;
	cv::boxFilter(I.mul(p), mean_Ip, CV_32FC1, cv::Size(r, r));

	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.    
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//mean_II = boxfilter(I.*I, r) ./ N;    
	cv::Mat mean_II;
	cv::boxFilter(I.mul(I), mean_II, CV_32FC1, cv::Size(r, r));

	//var_I = mean_II - mean_I .* mean_I;    
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	//a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;       
	cv::Mat a = cov_Ip / (var_I + eps);

	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;    
	cv::Mat b = mean_p - a.mul(mean_I);

	//mean_a = boxfilter(a, r) ./ N;    
	cv::Mat mean_a;
	cv::boxFilter(a, mean_a, CV_32FC1, cv::Size(r, r));
	cv::Mat rmean_a;
	resize(mean_a, rmean_a, I_org.size(), 1);

	//mean_b = boxfilter(b, r) ./ N;    
	cv::Mat mean_b;
	cv::boxFilter(b, mean_b, CV_32FC1, cv::Size(r, r));
	cv::Mat rmean_b;
	resize(mean_b, rmean_b, I_org.size(), 1);

	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;    
	cv::Mat q = rmean_a.mul(_I) + rmean_b;

	return q;
}
Mat SID::getdarkprio2(Mat input_image,int win_size,double omega) {

	cv::Mat image_double;//获得double类型的图像，防止后续计算溢出
	input_image.convertTo(image_double, CV_32F, 1.0 / 255, 0);//转换为浮点图，方便后续计算而不溢出

															  /*************************************获取暗通道***********************************/
	cv::Mat dark_channel;
	get_darkchannel(image_double, dark_channel, win_size);
	/*********************************************************************************/

	/**************************由暗通道和原图计算大气光*********************************/
	cv::Vec3f atmosphere{ 0,0,0 };//用于存储大气光值的向量
	get_atmosphere(image_double, dark_channel, atmosphere);//通过函数获取大气光
														   /*********************************************************************************/

														   /*********************************计算光照传递评估图*******************************/
	cv::Mat trans_est;
	get_esttransmap(image_double, atmosphere, omega, win_size, trans_est);
	/*********************************************************************************/

	/***************************采用导向滤波改良传递评估图******************************/
	cv::Mat trans_map;//光照传输图矩阵
	cv::Mat image_gray;//原图对应的灰度图
	cv::cvtColor(image_double, image_gray, CV_BGR2GRAY);
	//采用导向滤波计算传递矩阵
	trans_map = fastGuidedFilter(image_gray, trans_est, 15, 0.001, 1);;
	/*********************************************************************************/

	/********************************获得无雾图像**************************************/
	cv::Mat radiance;
	get_radiance(image_double, trans_map, atmosphere, radiance);
	/*********************************************************************************/
	cv::Mat out_file;
	radiance.convertTo(out_file, CV_8U, 250.0);
	return radiance;
}

