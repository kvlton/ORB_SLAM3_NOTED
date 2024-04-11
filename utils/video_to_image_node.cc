#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <utils/video_process.h>

using namespace std;

VideoProcess* video_process_;

int main(int argc, char** argv)
{
  if(argc < 2) {
      cout <<"usage : " << argv[0] <<" config_yaml_file " << endl;
      return 1;
  }
  
  int bRGB;
  int nFrameId;
  double start_frame_time;
  double final_frame_time;
  std::string image_save_path;

  std::string strSettingFile = argv[1];
  cv::FileStorage fSettings(strSettingFile, cv::FileStorage::READ);
  nFrameId = fSettings["start_frame_index"];
  bRGB = fSettings["Camera.RGB"];
  start_frame_time = fSettings["start_frame_time"];
  final_frame_time = fSettings["final_frame_time"];
  fSettings["image_save_path"] >> image_save_path;
  fSettings.release();
  
  video_process_ = new VideoProcess(strSettingFile);
  video_process_->set_start_frame_time(start_frame_time);
  video_process_->set_final_frame_time(final_frame_time);
  
  string image_path = image_save_path + "/image_0";
  string command = "mkdir -p " + image_path;
  system(command.c_str());
  
  string times_file = image_save_path+"/times.txt";
  ofstream f_times;
  if(nFrameId == 0) {
      f_times.open(times_file.c_str());      
  } 
  else {
      f_times.open(times_file.c_str(), ios::app);
  }
  f_times << fixed;
  
  while(true)
  {
    if(video_process_->isCompleted()) {
      cout<<"Video has completed playing."<<endl;
      break;
    }
    
    cv::Mat image;
    double timestamp;
    video_process_->getImageStamped(image, timestamp);
    
    if(image.empty())
        continue;
    
    cv::Mat imGray;
    if(image.channels()==3)
    {
        if(bRGB)
            cvtColor(image,imGray,CV_RGB2GRAY);
        else
            cvtColor(image,imGray,CV_BGR2GRAY);
    }
    else if(image.channels()==4)
    {
        if(bRGB)
            cvtColor(image,imGray,CV_RGBA2GRAY);
        else
            cvtColor(image,imGray,CV_BGRA2GRAY);
    }
    
    stringstream ss;
    ss << setfill('0') << setw(6) << nFrameId;
    string image_file = image_path + "/" + ss.str() + ".png";
    cv::imwrite(image_file, imGray);
    
    f_times << timestamp <<std::endl;
    
    nFrameId ++;
  }
  
  printf("Image size: %d.\n", nFrameId);
  
  return 0;
}
