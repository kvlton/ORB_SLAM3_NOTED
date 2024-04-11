#include "utils/fisheye_correct.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <fstream>

using namespace std;

static const float PI = 3.14159265;
inline float degreeToRadian(float d){ return (d / 180.f)* PI; }
inline float radianToDegree(float r){ return (r / PI)* 180.f; } // 弧度->角度

FisheyeCorrect::FisheyeCorrect(const string& strSettingFile)
{
  /// 1.读取参数
  cv::FileStorage fSettings(strSettingFile,cv::FileStorage::READ);

  K_original = cv::Mat::eye(3,3,CV_32F);
  K_correct = K_original.clone();

  K_original.at<float>(0,0) = fSettings["Fisheye.fx"];
  K_original.at<float>(1,1) = fSettings["Fisheye.fy"];
  K_original.at<float>(0,2) = fSettings["Fisheye.cx"];
  K_original.at<float>(1,2) = fSettings["Fisheye.cy"];

  K_correct.at<float>(0,0) = fSettings["Camera.fx"];
  K_correct.at<float>(1,1) = fSettings["Camera.fx"];
  K_correct.at<float>(0,2) = fSettings["Camera.cx"];
  K_correct.at<float>(1,2) = fSettings["Camera.cy"];

  W_original = fSettings["Fisheye.width"];
  H_original = fSettings["Fisheye.height"];
  W_correct = fSettings["Camera.width"];
  H_correct = fSettings["Camera.height"];


  correct_choice = fSettings["correct_choice"];
  if(correct_choice == 0) /// 畸变表
  {
    // 畸变表
    string fisheye_distortion_file;
    fSettings["fisheye_distortion_file"] >> fisheye_distortion_file;
    ifstream instream(fisheye_distortion_file.c_str());
    if (!instream.is_open()){
        cout << "open distortion file failed: " << fisheye_distortion_file << endl;
        exit(-1);
    }

    distortion_list.reserve(900); // 0 ~ 90 度
    float current_distortion;
    while (instream >> current_distortion)
    {
        distortion_list.push_back(current_distortion);
    }
    instream.close();

    // 像素大小
    pixel_size = fSettings["pixel_size"];
  }
  else if(correct_choice == 1) /// 畸变系数
  {
    D.create(1,4,CV_32F);
    D.at<float>(0,0) = fSettings["DistCoeff.k1"];
    D.at<float>(0,1) = fSettings["DistCoeff.k2"];
    D.at<float>(0,2) = fSettings["DistCoeff.k3"];
    D.at<float>(0,3) = fSettings["DistCoeff.k4"];
  }

  fSettings.release();

  /// 2.生成矫正映射
  generate_undistort_map();

}

/// 生成 矫正映射
void FisheyeCorrect::generate_undistort_map()
{
    const float &centerX = K_correct.at<float>(0,2);
    const float &centerY = K_correct.at<float>(1,2);
    const cv::Point2f original_center(K_original.at<float>(0,2),K_original.at<float>(1,2));

    //    correction_map.create(h_rect,w_rect,CV_32FC2);
    correction_mapX.release();
    correction_mapY.release();
    correction_mapX = cv::Mat::ones(H_correct,W_correct,CV_32FC1)*(-1);
    correction_mapY = cv::Mat::ones(H_correct,W_correct,CV_32FC1)*(-1);

    const float focal_in_pixel = K_correct.at<float>(0,0);
    float fisheye_pt[3] = {0,0,focal_in_pixel};
    float norm = 0;

    if(correct_choice == 0) /// 使用 畸变表 矫正
    {
        for (int h = 0; h < H_correct; ++h)
        {
            fisheye_pt[1] = h - centerY;
            for (int w = 0; w < W_correct; ++w)
            {
                fisheye_pt[0] = w - centerX;
                norm = fisheye_pt[0] * fisheye_pt[0] + fisheye_pt[1] * fisheye_pt[1] + focal_in_pixel * focal_in_pixel;
                float cos_value = focal_in_pixel / sqrtf(norm);
                float degree = radianToDegree(acosf(cos_value)); // 偏离光心的角度
                if (degree > 100)
                    continue;

                // 查表
                int position_floor = floor(degree * 10);
                int position_ceil = ceil(degree * 10);
                float radius_in_fisheye_floor = distortion_list[position_floor];
                float radius_in_fisheye_ceil  = distortion_list[position_ceil];
                float radius_in_fisheye;
                if (radius_in_fisheye_ceil == radius_in_fisheye_floor)
                    radius_in_fisheye = radius_in_fisheye_ceil;
                else
                {
                    radius_in_fisheye = radius_in_fisheye_floor
                                        + (radius_in_fisheye_ceil - radius_in_fisheye_floor)
                                          * (degree * 10 - position_floor) / (position_ceil - position_floor);
                }
                radius_in_fisheye /= pixel_size;

                float distance_to_original_axies = sqrtf(fisheye_pt[0] * fisheye_pt[0] + fisheye_pt[1] * fisheye_pt[1]);
                float x = fisheye_pt[0] * (radius_in_fisheye / distance_to_original_axies);
                float y = fisheye_pt[1] * (radius_in_fisheye / distance_to_original_axies);

                correction_mapX.at<float>(h, w) = x + original_center.x;
                correction_mapY.at<float>(h, w) = y + original_center.y;

//            std::cout <<"radius fisheye: " << radius_in_fisheye << " , radius correct: " << distance_to_original_axies
//                      <<" ,from: (" << h <<","<< w <<") --> ("
//                      << x + original_center.x <<" , " << y + original_center.y << std::endl;
            }
        }
    }
    else if(correct_choice == 1) /// 使用 畸变系数 矫正
    {
        cv::Size correctSize;
        correctSize.width = W_correct,correctSize.height = H_correct;
        cv::fisheye::initUndistortRectifyMap(K_original, D, cv::Mat::eye(3,3,CV_32F), // 直接调用ＯpenCV的接口
                                             K_correct,correctSize, CV_32FC1,
                                             correction_mapX, correction_mapY);
    }

//    std::cout <<"correction map: " << std::endl
//                                   << correction_map << std::endl;
//    return map.clone();
}

void FisheyeCorrect::correctImage(const cv::Mat fisheyeImage, cv::Mat &correctImage)
{
    if(correction_mapX.empty() || correction_mapY.empty() || fisheyeImage.empty())
        return;

    // assert(fisheyeImage.type() == CV_8UC1);

    correctImage.release();
    cv::remap(fisheyeImage,correctImage,correction_mapX,correction_mapY,cv::INTER_LINEAR);
}
