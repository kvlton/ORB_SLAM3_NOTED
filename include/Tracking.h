/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

#include"Viewer.h"
#include"FrameDrawer.h"
#include"Atlas.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include "ImuTypes.h"

#include "GeometricCamera.h"

#include <mutex>
#include <unordered_set>

namespace ORB_SLAM3
{

class Viewer;
class FrameDrawer;
class Atlas;
class LocalMapping;
class LoopClosing;
class System;

class Tracking
{  

public:
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Atlas* pAtlas,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, const string &_nameSeq=std::string());

    ~Tracking();

    // Parse the config file
    bool ParseCamParamFile(cv::FileStorage &fSettings);
    bool ParseORBParamFile(cv::FileStorage &fSettings);
    bool ParseIMUParamFile(cv::FileStorage &fSettings);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp, string filename);
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename);
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename);
    // cv::Mat GrabImageImuMonocular(const cv::Mat &im, const double &timestamp);

    void GrabImuData(const IMU::Point &imuMeasurement);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);
    void SetStepByStep(bool bSet);

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    /// 改变追踪模式（定位模式，综合模式）
    void InformOnlyTracking(const bool &flag);

    void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame);
    KeyFrame* GetLastKeyFrame()
    {
        return mpLastKeyFrame;
    }

    void CreateMapInAtlas();
    std::mutex mMutexTracks;

    //--
    void NewDataset();
    int GetNumberDataset();
    int GetMatchesInliers();
public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        RECENTLY_LOST=3,
        LOST=4,
        OK_KLT=5
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    Frame mLastFrame;

    cv::Mat mImGray;

    // Initialization Variables (Monocular)
    /// 单目的初始化参数
    std::vector<int> mvIniLastMatches;       // 未使用？
    std::vector<int> mvIniMatches;           // 初始化匹配点的索引
    std::vector<cv::Point2f> mvbPrevMatched; // 初始化匹配点的2d坐标
    std::vector<cv::Point3f> mvIniP3D;       // 初始化匹配点的3d坐标, 初始坐标系
    Frame mInitialFrame;                     // 初始化帧（第一帧）

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    /// 用来保存每一帧的跟踪结果
    list<cv::Mat> mlRelativeFramePoses; // 相对于参考关键帧的位姿
    list<KeyFrame*> mlpReferences;      // 参考关键帧
    list<double> mlFrameTimes;          // 时间戳
    list<bool> mlbLost;                 // 是否跟丢

    // frames with estimated pose
    int mTrackedFr;
    bool mbStep;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;                // 当前帧的追踪模式

    void Reset(bool bLocMap = false);
    void ResetActiveMap(bool bLocMap = false);

    float mMeanTrack;
    bool mbInitWith3KFs;
    double t0; // time-stamp of first read frame
    double t0vis; // time-stamp of first inserted keyframe
    double t0IMU; // time-stamp of IMU initialization


    vector<MapPoint*> GetLocalMapMPS();


    //TEST--
    bool mbNeedRectify;
    //cv::Mat M1l, M2l;
    //cv::Mat M1r, M2r;

    bool mbWriteStats;

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    void CreateNewMapPoints();
    cv::Mat ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2);
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();
    bool PredictStateIMU();

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    bool TrackLocalMap_old();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // Perform preintegration from last frame
    void PreintegrateIMU();

    // Reset IMU biases and compute frame velocity
    void ResetFrameIMU();
    void ComputeGyroBias(const vector<Frame*> &vpFs, float &bwx,  float &bwy, float &bwz);
    void ComputeVelocitiesAccBias(const vector<Frame*> &vpFs, float &bax,  float &bay, float &baz);


    bool mbMapUpdated;

    // Imu preintegration from last frame
    IMU::Preintegrated *mpImuPreintegratedFromLastKF;

    // Queue of IMU measurements between frames
    std::list<IMU::Point> mlQueueImuData;

    // Vector of IMU measurements from previous to current frame (to be filled by PreintegrateIMU)
    std::vector<IMU::Point> mvImuFromLastFrame;
    std::mutex mMutexImuQueue;

    // Imu calibration parameters
    IMU::Calib *mpImuCalib;

    // Last Bias Estimation (at keyframe creation)
    IMU::Bias mLastBias;

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO; /// 定位模式下使用，表示上一帧的跟踪效果不好，下一帧偏向于重定位

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    Initializer* mpInitializer;
    bool mbSetInit;

    //Local Map
    KeyFrame* mpReferenceKF;                  // 局部地图跟踪会改变参考关键帧
    std::vector<KeyFrame*> mvpLocalKeyFrames; // 局部地图关键帧
    std::vector<MapPoint*> mvpLocalMapPoints; // 局部地图地图点
    
    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;
    bool bStepByStep;

    //Atlas
    Atlas* mpAtlas;

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    int mnFirstImuFrameId;
    int mnFramesToResetIMU;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth; // 近点的深度阈值（近点认为是可信，可以单帧插入地图点，远点需要匹配插入）

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor; // 深度图的比例因子

    //Current matches in frame
    int mnMatchesInliers; // 当前帧优化后内点数（匹配到的特征点数）

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;
    double mTimeStampLost;
    double time_recently_lost;

    unsigned int mnFirstFrameId;
    unsigned int mnInitialFrameId;
    unsigned int mnLastInitFrameId;

    bool mbCreatedMap;


    //Motion Model
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    // 上一帧新添的地图点（用来在后面删除这些新添加的地图点）
    list<MapPoint*> mlpTemporalPoints;

    //int nMapChangeIndex;

    int mnNumDataset;

    ofstream f_track_stats;

    ofstream f_track_times;
    double mTime_PreIntIMU;
    double mTime_PosePred;
    double mTime_LocalMapTrack;
    double mTime_NewKF_Dec;

    GeometricCamera* mpCamera, *mpCamera2;

    int initID, lastID;

    cv::Mat mTlr;

public:
    cv::Mat mImRight;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
