#include "evalute.h"

using namespace cv;
using namespace std;

unsigned long Evalue::getImageQualityAssessment(Mat & img)
{
	unsigned long score = 0;
	for (int x = 0; x < img.rows; x++) {     //行数
		uchar* ptr = img.ptr<uchar>(x);
#pragma omp parallel for
		for (int y = 0; y < img.cols*img.channels(); y++) //列数
			score += ptr[y];
	}
	return score;
}

double Evalue::Tenengrad(Mat & roi)  //Tenengrad梯度函数
{
	Mat imageGray;
	cvtColor(roi, imageGray, CV_RGB2GRAY);
	Mat imageSobel;
	Sobel(imageGray, imageSobel, CV_16U, 1, 1);

	//图像的平均灰度
	double meanValue = 0.0;
	meanValue = mean(imageSobel)[0];
/*	Mat smooth_image;
	blur(roi, smooth_image, Size(3, 3));  //均值滤波
	Mat sobel_x, sobel_y;
	Sobel(smooth_image, sobel_x, CV_16U, 1, 0, 3); //获取x方向上的sobel算子
	Sobel(smooth_image, sobel_y, CV_16U, 0, 1, 3); // 获取y方向上的sobel算子

	convertScaleAbs(sobel_x, sobel_x);   //对x方向上增强
	convertScaleAbs(sobel_y, sobel_y);   //对y方向上增强

	Mat pow_x;
	Mat pow_y;
	pow(sobel_x, 2, pow_x);   //像素平方
	pow(sobel_y, 2, pow_x);

	Mat weighted_mat, result;
	addWeighted(pow_x,0.5,pow_y,0.5,0,weighted_mat);  //加权
	//weighted_mat = pow_x + pow_y;
	cv::sqrt(weighted_mat, weighted_mat);
	double score = getImageQualityAssessment(weighted_mat);
*/
	return meanValue;
}

double Evalue::Brenner(Mat & image)
{
	double score = 0.0;
	for (int x = 0; x < image.rows; x++) {     //行数
		uchar* ptr = image.ptr<uchar>(x);
#pragma omp parallel for
		for (int y = 0; y < (image.cols - 2)*image.channels(); y++) //列数
			score += abs(ptr[y + 2] - ptr[y])*abs(ptr[y + 2] - ptr[y]);
	}
	return score;
}

double Evalue::SMD(Mat & image)
{
	double score = 0.0;
	for (int i = 1; i < image.rows - 1; i++) {
		uchar *ptr = image.ptr<uchar>(i);
		uchar *ptr_1 = image.ptr<uchar>(i - 1);
#pragma omp parallel for
		for (int j = 0; j < (image.cols - 1)*image.channels(); j++) {
			score += abs(ptr[j] - ptr_1[j]) + abs(ptr[j] - ptr[j + 1]);
		}
	}
	return score;
}

double Evalue::SMD2(Mat & image)
{
	double score = 0.0;
	for (int i = 0; i < image.rows - 1; i++) {
		uchar *ptr = image.ptr<uchar>(i);
		uchar *ptr_1 = image.ptr<uchar>(i + 1);
#pragma omp parallel for
		for (int j = 0; j < (image.cols - 1)*image.channels(); j++) {
			score += abs(ptr[j] - ptr[j + 1])*abs(ptr[j] - ptr_1[j]);
		}
	}
	return score;
}

double Evalue::Enery(Mat & image)
{
	double score = 0;
	for (int i = 0; i < image.rows - 1; ++i)
	{
		uchar* ptr = image.ptr<uchar>(i);
		uchar* ptr_1 = image.ptr<uchar>(i + 1);
#pragma omp parallel for
		for (int j = 0; j < (image.cols - 1) * image.channels(); ++j)
		{
			score += pow(ptr[j + 1] - ptr[j], 2) + pow(ptr_1[j] - ptr[j], 2);
		}
	}
	return score;
}

double Evalue::Jpeg(Mat & image)
{
	//horizontal calculate
	auto b_h = 0.0;
	for (int i = 1; i < floor(image.rows / 8) - 1; ++i)
	{
		uchar* ptr = image.ptr<uchar>(8 * i);
		uchar* ptr_1 = image.ptr<uchar>(8 * i + 1);
#pragma omp parallel for
		for (int j = 1; j < image.cols; ++j)
		{
			b_h += abs(ptr_1[j] - ptr[j]);
		}
	}
	b_h *= 1 / (image.cols*(floor(image.rows / 8) - 1));

	auto a_h = 0.0;
	for (int i = 1; i < image.rows - 1; ++i)
	{
		uchar* ptr = image.ptr<uchar>(i);
		uchar* ptr_1 = image.ptr<uchar>(i + 1);
#pragma omp parallel for
		for (int j = 1; j < image.cols; ++j)
		{
			a_h += abs(ptr_1[j] - ptr[j]);
		}
	}
	a_h = (a_h * 8.0 / (image.cols * (image.rows - 1)) - b_h) / 7;

	auto z_h = 0.0;
	for (int i = 1; i < image.rows - 2; ++i)
	{
		uchar* ptr = image.ptr<uchar>(i);
		uchar* ptr_1 = image.ptr<uchar>(i + 1);
#pragma omp parallel for
		for (int j = 1; j < image.cols; ++j)
		{
			z_h += (ptr_1[j] - ptr[j]) * (ptr_1[j + 1] - ptr[j]) > 0 ? 0 : 1;
		}
	}
	z_h *= 1.0 / (image.cols* (image.rows - 2));

	//vertical calculate
	auto b_v = 0.0;
	for (int i = 1; i < image.rows; ++i)
	{
		uchar* ptr = image.ptr<uchar>(i);
#pragma omp parallel for
		for (int j = 1; j < floor(image.cols / 8) - 1; ++j)
		{
			b_v += abs(ptr[8 * j + 1] - ptr[8 * j]);
		}
	}
	b_v *= 1.0 / (image.rows*(floor(image.cols / 8) - 1));

	auto a_v = 0.0;
	for (int i = 1; i < image.rows; ++i)
	{
		uchar* ptr = image.ptr<uchar>(i);
#pragma omp parallel for
		for (int j = 1; j < image.cols - 1; ++j)
		{
			a_v += abs(ptr[j + 1] - ptr[j]);
		}
	}
	a_v = (a_v * 8.0 / (image.rows * (image.cols - 1)) - b_v) / 7;

	auto z_v = 0.0;
	for (int i = 1; i < image.rows; ++i)
	{
		uchar* ptr = image.ptr<uchar>(i);
#pragma omp parallel for
		for (int j = 1; j < image.cols - 2; ++j)
		{
			z_v += (ptr[j + 1] - ptr[j]) * (ptr[j + 2] - ptr[j + 1]) > 0 ? 0 : 1;
		}
	}
	z_v *= 1.0 / (image.rows* (image.cols - 2));

	////////////////////////////////////////////////////////////////////////////
	auto B = (b_v + b_h) / 2;
	auto A = (a_h + a_v) / 2;
	auto Z = (z_h + z_v) / 2;
	auto alpha = -245.9, beta = 261.9, gamma1 = -0.024, gamma2 = 0.016, gamma3 = 0.0064;

	auto S = alpha + beta*pow(B, gamma1)*pow(A, gamma2)*pow(Z, gamma3);
	return S;
}

double Evalue::Jpeg2(Mat & image)
{
	double s = Jpeg(image);
	double ss = 4.0 / (1.0 + exp(-1.0217*(s - 3))) + 1.0;
	return ss;
}

double Evalue::Tenengrad_Diff(Mat & image)
{
	Mat imageGray;
	Mat imageSobel;
	cvtColor(image, imageGray, CV_RGB2GRAY);

	Sobel(imageGray, imageSobel, CV_16U, 1, 1);

	//图像的平均灰度  
	double meanValue = 0.0;
	meanValue = mean(imageSobel)[0];
	//cout << "Tenengrad_Diff:" << meanValue << endl;
	return meanValue;
}

double Evalue::Laplacian(Mat & image)
{
	Mat imageGray;
	Mat imageLap;

	cvtColor(image, imageGray, CV_RGB2GRAY);
	cv::Laplacian(imageGray, imageLap, CV_16U);

	double meanValue = 0.0;
	meanValue = mean(imageLap)[0];
	//cout << "Laplacian:" << meanValue << endl;
	return meanValue;
}

double Evalue::variance(Mat & image)
{
	Mat imageGrey;

	cvtColor(image, imageGrey, CV_RGB2GRAY);
	Mat meanValueImage;
	Mat meanStdValueImage;

	meanStdDev(imageGrey, meanValueImage, meanStdValueImage); //求标准差
	double meanValue = 0.0;
	meanValue = meanStdValueImage.at<double>(0, 0);

	//cout << "方差:" << meanValue << endl;
	return meanValue;
}

/*
这个很容易理解，模糊的图像，就是像素点颜色的过渡非常平滑，没有颜色急剧的变化。反之锐化的图像就是像素点颜色变化很突然。
那么怎么反映两个像素点颜色的过渡呢，
sqrt((pow((double)(data[(i + 1)*step + j] - data[i*step + j]), 2) \
+ pow((double)(data[i*step + j + 1] - data[i*step + j]), 2)));
这里算的是像素点的平方差。
abs(data[(i + 1)*step + j] - data[i*step + j]) + abs(data[i*step + j + 1] - data[i*step + j]);
这里算的是差的绝对值。
为什么算两个，前者表现的是变化速度的快慢，后者是变化的大小。用这两个指标来评价比较准确。
DR = temp / num;
求一个平均数。
*/
double Evalue::DefRto(Mat frame)
{
	Mat gray;
	cvtColor(frame, gray, CV_BGR2GRAY);
	IplImage *img = &(IplImage(gray));
	double temp = 0;
	double DR = 0;
	int i, j;//循环变量 
	int height = img->height;
	int width = img->width;
	int step = img->widthStep / sizeof(uchar);
	uchar *data = (uchar*)img->imageData;
	double num = width*height;
	for (i = 0; i < height - 1; i++)
	{
		for (j = 0; j < width; j++)
		{
			temp += sqrt((pow((double)(data[(i + 1)*step + j] - data[i*step + j]), 2) \
				+ pow((double)(data[i*step + j + 1] - data[i*step + j]), 2)));
			temp += abs(data[(i + 1)*step + j] - data[i*step + j]) + abs(data[i*step + j + 1] - data[i*step + j]);
		}
	}
	DR = temp / num;
	return DR;
}
Mat Evalue::addSaltNoise(const Mat srcImage, int n)
{
	Mat dstImage = srcImage.clone();
	for (int k = 0; k < n; k++)
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;		//盐噪声
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 255;
			dstImage.at<Vec3b>(i, j)[1] = 255;
			dstImage.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int k = 0; k < n; k++)
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;		//椒噪声
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 0;
			dstImage.at<Vec3b>(i, j)[1] = 0;
			dstImage.at<Vec3b>(i, j)[2] = 0;
		}
	}
	return dstImage;
}
//生成高斯噪声
double Evalue::generateGaussianNoise(double mu, double sigma)
{
	//定义小值
	const double epsilon = numeric_limits<double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flag为假构造高斯随机变量X
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;
	//构造随机变量
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flag为真构造高斯随机变量
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z0*sigma + mu;
}

//为图像添加高斯噪声
Mat Evalue::addGaussianNoise(Mat &srcImag)
{
	Mat dstImage = srcImag.clone();
	int channels = dstImage.channels();
	int rowsNumber = dstImage.rows;
	int colsNumber = dstImage.cols*channels;
	//判断图像的连续性
	if (dstImage.isContinuous())
	{
		colsNumber *= rowsNumber;
		rowsNumber = 1;
	}
	for (int i = 0; i < rowsNumber; i++)
	{
		for (int j = 0; j < colsNumber; j++)
		{
			//添加高斯噪声
			int val = dstImage.ptr<uchar>(i)[j] +
				generateGaussianNoise(2, 0.8) * 32;
			if (val < 0)
				val = 0;
			if (val>255)
				val = 255;
			dstImage.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	return dstImage;
}

/* 信息熵  评价图像指标*/
double Evalue::Entropy(Mat img) {
	double temp[256] = { 0.0 };

	//计算每个像素的累计值

	for (int m = 0; m < img.rows; m++) {
		const uchar*t = img.ptr<uchar>(m);
		for (int n = 0; n < img.cols; n++) {
			int i = t[n];
			temp[i] = temp[i] + 1;
		}
	}

	//计算每个像素的 概率
	for (int i = 0; i < 256; i++) {
		temp[i] = temp[i] / (img.cols * img.rows);
	}

	double result = 0.0;
	//计算图像的信息熵
	for (int i = 0; i < 256; i++) {
		if (temp[i] == 0.0)
			result = result;
		else
			result = result - temp[i] * (log(temp[i]) / log(2.0));
	}
	return result;
}

// 两幅图像联合信息熵计算
/*double ComEntropy(Mat img1, Mat img2, double img1_entropy, double img2_entropy)
{
	double temp[256][256] = { 0.0 };

	// 计算联合图像像素的累积值
	for (int m1 = 0, m2 = 0; m1 < img1.rows, m2 < img2.rows; m1++, m2++)
	{    // 有效访问行列的方式
		const uchar* t1 = img1.ptr<uchar>(m1);
		const uchar* t2 = img2.ptr<uchar>(m2);
		for (int n1 = 0, n2 = 0; n1 < img1.cols, n2 < img2.cols; n1++, n2++)
		{
			int i = t1[n1], j = t2[n2];
			temp[i][j] = temp[i][j] + 1;
		}
	}

	// 计算每个联合像素的概率
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)

		{
			temp[i][j] = temp[i][j] / (img1.rows*img1.cols);
		}
	}

	double result = 0.0;
	//计算图像联合信息熵
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)

		{
			if (temp[i][j] == 0.0)
				result = result;
			else
				result = result - temp[i][j] * (log(temp[i][j]) / log(2.0));
		}
	}

	//得到两幅图像的互信息熵
	img1_entropy = Entropy(img1);
	img2_entropy = Entropy(img2);
	result = img1_entropy + img2_entropy - result;

	return result;

}*/

/*SSIM 结构相似性*/
Scalar Evalue::getMSSIM(Mat  inputimage1, Mat inputimage2)
{
	Mat i1 = inputimage1;
	Mat i2 = inputimage2;
	const double C1 = 6.5025, C2 = 58.5225;
	int d = CV_32F;
	Mat I1, I2;
	i1.convertTo(I1, d);
	i2.convertTo(I2, d);
	Mat I2_2 = I2.mul(I2);
	Mat I1_2 = I1.mul(I1);
	Mat I1_I2 = I1.mul(I2);
	Mat mu1, mu2;
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);
	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);
	Mat sigma1_2, sigma2_2, sigma12;
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;
	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;
	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);
	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);
	Mat ssim_map;
	divide(t3, t1, ssim_map);
	Scalar mssim = mean(ssim_map);
	return mssim;
}