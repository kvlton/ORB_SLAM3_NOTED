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


#ifndef MAPPOINT_H
#define MAPPOINT_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

#include<opencv2/core/core.hpp>
#include<mutex>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>

namespace ORB_SLAM3
{

class KeyFrame;
class Map;
class Frame;

class MapPoint
{
    template<class Archive>
    void serializeMatrix(Archive &ar, cv::Mat& mat, const unsigned int version)
    {
        int cols, rows, type;
        bool continuous;

        if (Archive::is_saving::value) {
            cols = mat.cols; rows = mat.rows; type = mat.type();
            continuous = mat.isContinuous();
        }

        ar & cols & rows & type & continuous;
        if (Archive::is_loading::value)
            mat.create(rows, cols, type);

        if (continuous) {
            const unsigned int data_size = rows * cols * mat.elemSize();
            ar & boost::serialization::make_array(mat.ptr(), data_size);
        } else {
            const unsigned int row_size = cols*mat.elemSize();
            for (int i = 0; i < rows; i++) {
                ar & boost::serialization::make_array(mat.ptr(i), row_size);
            }
        }
    }

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & mnId;
        ar & mnFirstKFid;
        ar & mnFirstFrame;
        ar & nObs;
        // Variables used by the tracking
        ar & mTrackProjX;
        ar & mTrackProjY;
        ar & mTrackDepth;
        ar & mTrackDepthR;
        ar & mTrackProjXR;
        ar & mTrackProjYR;
        ar & mbTrackInView;
        ar & mbTrackInViewR;
        ar & mnTrackScaleLevel;
        ar & mnTrackScaleLevelR;
        ar & mTrackViewCos;
        ar & mTrackViewCosR;
        ar & mnTrackReferenceForFrame;
        ar & mnLastFrameSeen;

        // Variables used by local mapping
        ar & mnBALocalForKF;
        ar & mnFuseCandidateForKF;

        // Variables used by loop closing and merging
        ar & mnLoopPointForKF;
        ar & mnCorrectedByKF;
        ar & mnCorrectedReference;
        serializeMatrix(ar,mPosGBA,version);
        ar & mnBAGlobalForKF;
        ar & mnBALocalForMerge;
        serializeMatrix(ar,mPosMerge,version);
        serializeMatrix(ar,mNormalVectorMerge,version);

        // Protected variables
        serializeMatrix(ar,mWorldPos,version);
        //ar & BOOST_SERIALIZATION_NVP(mBackupObservationsId);
        ar & mBackupObservationsId1;
        ar & mBackupObservationsId2;
        serializeMatrix(ar,mNormalVector,version);
        serializeMatrix(ar,mDescriptor,version);
        ar & mBackupRefKFId;
        ar & mnVisible;
        ar & mnFound;

        ar & mbBad;
        ar & mBackupReplacedId;

        ar & mfMinDistance;
        ar & mfMaxDistance;

    }


public:
    MapPoint();

    MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
    MapPoint(const double invDepth, cv::Point2f uv_init, KeyFrame* pRefKF, KeyFrame* pHostKF, Map* pMap);
    MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);

    void SetWorldPos(const cv::Mat &Pos);

    cv::Mat GetWorldPos();

    cv::Mat GetNormal();
    KeyFrame* GetReferenceKeyFrame();

    std::map<KeyFrame*,std::tuple<int,int>> GetObservations();
    int Observations();

    void AddObservation(KeyFrame* pKF,int idx);
    void EraseObservation(KeyFrame* pKF);

    std::tuple<int,int> GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint* pMP);    
    MapPoint* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();
    void SetNormalVector(cv::Mat& normal);

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, KeyFrame*pKF);
    int PredictScale(const float &currentDist, Frame* pF);

    Map* GetMap();
    void UpdateMap(Map* pMap);

    void PrintObservations();

    void PreSave(set<KeyFrame*>& spKF,set<MapPoint*>& spMP);
    void PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int, MapPoint*>& mpMPid);

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;

    // Variables used by the tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackDepth;
    float mTrackDepthR;
    float mTrackProjXR;
    float mTrackProjYR;
    bool mbTrackInView, mbTrackInViewR;     // 是否在跟踪范围内
    int mnTrackScaleLevel, mnTrackScaleLevelR;
    float mTrackViewCos, mTrackViewCosR;
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnLastFrameSeen;      // 最后一次观测到该地图点的普通帧

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF; // 防止重复添加

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;     // 用来寻找闭环帧附近的地图点，防止重复添加
    long unsigned int mnCorrectedByKF;      // 用来表示该地图点已经被矫正过了
    long unsigned int mnCorrectedReference; // 被矫正时，所在的关键帧
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;      // 表示是否已经在全局BA中被矫正了
    long unsigned int mnBALocalForMerge;

    // Variable used by merging
    cv::Mat mPosMerge;          // Merge后的 3d坐标
    cv::Mat mNormalVectorMerge; // Merge后的 平均观测方向


    // Fopr inverse depth optimization
    double mInvDepth;
    double mInitU;
    double mInitV;
    KeyFrame* mpHostKF;

    static std::mutex mGlobalMutex;

    unsigned int mnOriginMapId;

protected:    

     // Position in absolute coordinates
     cv::Mat mWorldPos;

     // Keyframes observing the point and associated index in keyframe
     std::map<KeyFrame*,std::tuple<int,int> > mObservations;
     // For save relation without pointer, this is necessary for save/load function
     std::map<long unsigned int, int> mBackupObservationsId1;
     std::map<long unsigned int, int> mBackupObservationsId2;

     // Mean viewing direction
     cv::Mat mNormalVector; // 平均观测方向

     // Best descriptor to fast matching
     cv::Mat mDescriptor;

     // Reference KeyFrame
     KeyFrame* mpRefKF;
     long unsigned int mBackupRefKFId;

     // Tracking counters
     int mnVisible;
     int mnFound;

     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;           // 表示该地图点被替换、删除
     MapPoint* mpReplaced; // 用来代替 当前地图点
     // For save relation without pointer, this is necessary for save/load function
     long long int mBackupReplacedId;

     // Scale invariance distances
     float mfMinDistance; // 能观测到该地图点的最近距离
     float mfMaxDistance; // 能观测到该地图点的最远距离

     Map* mpMap;

     std::mutex mMutexPos;
     std::mutex mMutexFeatures;
     std::mutex mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
