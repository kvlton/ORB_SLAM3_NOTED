#ifndef VIDEOPROCESS_H
#define VIDEOPROCESS_H

#include <string.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "utils/fisheye_correct.h"

using namespace std;

class VideoProcess
{
public:
  VideoProcess(const string&  strSettingFile);
  ~VideoProcess();

  bool getImageStamped(cv::Mat& image, double& nFrameStamp);

  void set_start_frame_time(double value);
  void set_final_frame_time(double value);
  
  bool isCompleted();
  
private:
  void ReadImageStamps(const string& strInfoFile);

  cv::VideoCapture video_;
  int mbCorrected;
  FisheyeCorrect* mpFisheyeCorrect;

  vector<double> mvFrameTimes; // 视频时间戳
  unsigned long int mnFrameId;
  
  double start_frame_time; // 视频开始播放的时间戳
  double final_frame_time;
  
  bool mbCompleted;

};

#endif // VIDEOPROCESS_H
