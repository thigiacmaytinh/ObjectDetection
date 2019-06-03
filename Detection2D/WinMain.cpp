// TGMTtemplate.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <tchar.h>
#include "TGMTConfig.h"
#include "TGMTcamera.h"
#include "TGMTobjDetect.h"
#include "TGMTdraw.h"
#include "TGMTdebugger.h"
#include "TGMTfile.h"


#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv::xfeatures2d;

cv::Mat g_mat;

cv::Mat ResizeByHeight(cv::Mat matInput, int height)
{
	cv::Mat matResult;
	float scaleWidth = matInput.cols * height / matInput.rows;
	cv::resize(matInput, matResult, cv::Size(scaleWidth, height));
	return matResult;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawAndShowKeypoint(cv::Mat matInput, std::vector<cv::KeyPoint> keypoints)
{
	
	for (int i = 0; i < keypoints.size(); i++)
	{
		cv::KeyPoint key = keypoints[i];
		TGMTshape::Circle circle(key.pt, 3);
		TGMTdraw::DrawCircle(g_mat, circle);		
	}

	ShowImage(g_mat, "draw keypoint");

	cv::imwrite("keypoint.jpg", g_mat);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SURFmatching(cv::Mat matScene, cv::Mat matObject)
{

	//-- Step 1: Detect the keypoints and extract descriptors using SURF
	int minHessian = 400;
	cv::Ptr<SURF> detector = SURF::create(minHessian);
	std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
	cv::Mat descriptors_object, descriptors_scene;
	detector->detectAndCompute(matObject, cv::Mat(), keypoints_object, descriptors_object);
	detector->detectAndCompute(matScene, cv::Mat(), keypoints_scene, descriptors_scene);

	DrawAndShowKeypoint(matScene, keypoints_scene);
	//-- Step 2: cv::Matching descriptor vectors using FLANN cv::Matcher
	cv::FlannBasedMatcher matcher;
	std::vector< cv::DMatch > matches;
	matcher.match(descriptors_object, descriptors_scene, matches);
	double max_dist = 0; double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);
	//-- Draw only "good" cv::Matches (i.e. whose distance is less than 3*min_dist )
	std::vector< cv::DMatch > good_Matches;
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_Matches.push_back(matches[i]);
		}
	}
	cv::Mat img_Matches;
	cv::drawMatches(matObject, keypoints_object, matScene, keypoints_scene,
		good_Matches, img_Matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


	//-- Localize the object
	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;
	for (size_t i = 0; i < good_Matches.size(); i++)
	{
		//-- Get the keypoints from the good cv::Matches
		obj.push_back(keypoints_object[good_Matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_Matches[i].trainIdx].pt);
	}
	cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC);
	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<cv::Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(matObject.cols, 0);
	obj_corners[2] = cvPoint(matObject.cols, matObject.rows); obj_corners[3] = cvPoint(0, matObject.rows);
	std::vector<cv::Point2f> scene_corners(4);
	cv::perspectiveTransform(obj_corners, scene_corners, H);
	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	cv::line(img_Matches, scene_corners[0] + cv::Point2f(matObject.cols, 0), scene_corners[1] + cv::Point2f(matObject.cols, 0), cv::Scalar(0, 255, 0), 4);
	cv::line(img_Matches, scene_corners[1] + cv::Point2f(matObject.cols, 0), scene_corners[2] + cv::Point2f(matObject.cols, 0), cv::Scalar(0, 255, 0), 4);
	cv::line(img_Matches, scene_corners[2] + cv::Point2f(matObject.cols, 0), scene_corners[3] + cv::Point2f(matObject.cols, 0), cv::Scalar(0, 255, 0), 4);
	cv::line(img_Matches, scene_corners[3] + cv::Point2f(matObject.cols, 0), scene_corners[0] + cv::Point2f(matObject.cols, 0), cv::Scalar(0, 255, 0), 4);
	//-- Show detected cv::Matches



	img_Matches = ResizeByHeight(img_Matches, 800);
	cv::imshow("Good cv::Matches & Object detection", img_Matches);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

int _tmain(int argc, _TCHAR* argv[])
{
	cv::Mat matScene = cv::imread(argv[1]);
	g_mat = matScene;
	cv::cvtColor(matScene, matScene, CV_BGR2GRAY);

	cv::Mat matObject = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	SURFmatching(matScene, matObject);
	

	cv::waitKey();
	getchar();
	return 0;
}

