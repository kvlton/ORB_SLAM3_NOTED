# include "utils/video_process.h"


VideoProcess::VideoProcess(const string& strSettingFile):mnFrameId(0), start_frame_time(0.0),
  final_frame_time(DBL_MAX), mbCompleted(0)
{
  cv::FileStorage fSettings(strSettingFile,cv::FileStorage::READ);

  /// 1.加载视频流
  string video_path;
  fSettings["video_path"] >> video_path;
  if(!video_.open(video_path)) {
    printf("read %s error!", video_path.c_str());
    exit(-1);
  }

  /// 2.读取视频流的时间戳
  string frame_info_file;
  fSettings["frame_info_file"] >> frame_info_file;
  ReadImageStamps(frame_info_file);

  /// 3.鱼眼矫正
  mbCorrected = fSettings["undistort_flag"];
  if(mbCorrected == 1)
    mpFisheyeCorrect = new FisheyeCorrect(strSettingFile);

  fSettings.release();
}

VideoProcess::~VideoProcess()
{
  video_.release();

  if(mpFisheyeCorrect) {
      delete mpFisheyeCorrect;
      mpFisheyeCorrect = nullptr;
  }
}

/// 读取视频流的时间戳
void VideoProcess::ReadImageStamps(const string& strInfoFile)
{
  ifstream infile(strInfoFile.c_str(), ios::binary);
  if(infile.fail()) {
    cout<<"read file failed: "<<strInfoFile.c_str()<<endl;
    exit(-1);
  }

  unsigned int flag = 0;
  unsigned long int index = 0, time = 0;
  mvFrameTimes.clear();

  while(!infile.eof()) {
      infile.read((char*)&flag,sizeof(flag));
      infile.read((char*)&index,sizeof(index));
      infile.read((char*)&time,sizeof(time));
      if(flag)
        mvFrameTimes.push_back(time*1.0e-6); // 时间戳
  }
  infile.close();
}

void VideoProcess::set_start_frame_time(double value)
{
    start_frame_time = value;
}

void VideoProcess::set_final_frame_time(double value)
{
    final_frame_time = value;
}

/// 逐帧获取图像和时间戳
bool VideoProcess::getImageStamped(cv::Mat& image, double& nFrameStamp)
{
  if(mnFrameId >= mvFrameTimes.size() || nFrameStamp >= final_frame_time) {
    cout<<"Video has completed playing."<<endl;
    
    mbCompleted = true;
    return false;
  }

  // 视频不一定是从0开始播放的
  while(mvFrameTimes[mnFrameId] < start_frame_time)
  {
    video_ >> image;
    mnFrameId++;

    if(image.empty()) {
      cout<<"Total time of video is less than "<<start_frame_time<<endl;
      return false;
    }
  }

  // 读取当前帧视频和时间戳
  video_ >> image;
  if(image.empty()) {
    cout<<"Video has completed playing."<<endl;
    return false;
  }

  nFrameStamp = mvFrameTimes[mnFrameId];

  // 鱼眼矫正
  if(mbCorrected == 1) {
    mpFisheyeCorrect->correctImage(image, image);
//    cout<<"Current fisheye image is corrected."<<endl;
  }

  mnFrameId ++;

  return true;
}

bool VideoProcess::isCompleted()
{
    return mbCompleted;
}


