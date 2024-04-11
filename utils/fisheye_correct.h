#ifndef FISHEYE_AS_MONO_FISHEYECORRECT_H
#define FISHEYE_AS_MONO_FISHEYECORRECT_H

#include <opencv2/core/core.hpp>
#include <vector>

using namespace std;

class FisheyeCorrect
{
public:
    FisheyeCorrect(const string& strSettingFile);

    void generate_undistort_map();

    void correctImage(const cv::Mat fisheyeImage,cv::Mat &correctImage);

private:
    int correct_choice; // 0:畸变表  1:畸变系数

    cv::Mat K_original, K_correct;
    int W_original, H_original, W_correct, H_correct;

    // 畸变表
    std::vector<float> distortion_list;
    float pixel_size;

    // 畸变系数
    cv::Mat D;

    // 矫正映射
    cv::Mat correction_mapX, correction_mapY;

};


#endif //FISHEYE_AS_MONO_FISHEYECORRECT_H
