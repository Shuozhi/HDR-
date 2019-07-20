#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>
using namespace cv;
using namespace cv::xfeatures2d;

//SIRF特征检测
Mat src;
int numFeatures = 120;
void trackBar(int, void*);
int main()
{
	src = imread("262A2645.tif");
	if (src.empty())
	{
		printf("can not load image \n");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);

	namedWindow("output", WINDOW_AUTOSIZE);
	createTrackbar("minHessian", "output", &numFeatures, 500, trackBar);

	waitKey(0);
	return 0;
}


void trackBar(int, void*)
{
	Mat dst;
	// SIRF特征检测
	Ptr<SIFT> detector = SIFT::create(numFeatures);
	std::vector<KeyPoint> keypoints;
	detector->detect(src, keypoints, Mat());
	// 绘制关键点
	drawKeypoints(src, keypoints, dst, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("output", dst);
}
