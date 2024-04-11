#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <utils/video_process.h>
#include <System.h>

using namespace std; 

VideoProcess* video_process_;
ORB_SLAM3::System* system_;

int main(int argc, char** argv)
{
  if(argc < 2) {
      cout <<"usage : " << argv[0] <<" config_yaml_file " << endl;
      return 1;
  }

  std::string strSettingFile = argv[1];
  std::string strVocFile;
  cv::FileStorage fSettings(strSettingFile, cv::FileStorage::READ);
  fSettings["vocabulary_file"] >> strVocFile;
  float fps = fSettings["Camera.fps"];
  fSettings.release();

  video_process_ = new VideoProcess(strSettingFile);
  system_ = new ORB_SLAM3::System(strVocFile, strSettingFile, ORB_SLAM3::System::MONOCULAR, true);

  while(true)
  {
    cv::Mat image;
    double timestamp;
    if(!video_process_->getImageStamped(image, timestamp)) {
      cout<<"Video has completed playing."<<endl;
      break;
    }
    
    if(image.empty()) continue;
    
    std::cout<<"Current frame timestamp: "<<timestamp<<std::endl;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    system_->TrackMonocular(image, timestamp, vector<ORB_SLAM3::IMU::Point>(), to_string(timestamp)+".png");               /// slam 系统入口
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    double ttrack = chrono::duration_cast<chrono::duration<double>>(t2-t1).count();
////    cout<<"Current frame track costs time: "<<ttrack<<" seconds."<<endl;

    double T = 1.0/fps;
    if(ttrack<T) {
        usleep((T-ttrack)*1e6);
    }
    
    system_->Shutdown();

  }

  return 0;
}
