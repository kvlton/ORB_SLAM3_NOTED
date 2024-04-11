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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM3
{

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint():
    mnFirstKFid(0), mnFirstFrame(0), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL))
{
    mpReplaced = static_cast<MapPoint*>(NULL);
}

MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
    mnOriginMapId(pMap->GetId())
{
    Pos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);

    mbTrackInViewR = false;
    mbTrackInView = false;

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    // 防止地图点ID冲突（地图点可以由 跟踪线程 和 局部地图线程 两个线程插入）
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

MapPoint::MapPoint(const double invDepth, cv::Point2f uv_init, KeyFrame* pRefKF, KeyFrame* pHostKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
    mnOriginMapId(pMap->GetId())
{
    mInvDepth=invDepth;
    mInitU=(double)uv_init.x;
    mInitV=(double)uv_init.y;
    mpHostKF = pHostKF;

    mNormalVector = cv::Mat::zeros(3,1,CV_32F);

    // Worldpos is not set
    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap), mnOriginMapId(pMap->GetId())
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow;
    if(pFrame -> Nleft == -1 || idxF < pFrame -> Nleft){
        Ow = pFrame->GetCameraCenter();
    }
    else{
        cv::Mat Rwl = pFrame -> mRwc;
        cv::Mat tlr = pFrame -> mTlr.col(3);
        cv::Mat twl = pFrame -> mOw;

        Ow = Rwl * tlr + twl;
    }
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector/cv::norm(mNormalVector);

    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);
    const int level = (pFrame -> Nleft == -1) ? pFrame->mvKeysUn[idxF].octave
                                              : (idxF < pFrame -> Nleft) ? pFrame->mvKeys[idxF].octave
                                                                         : pFrame -> mvKeysRight[idxF].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat MapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

cv::Mat MapPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

/// 添加能观测到该地图点的关键帧，idx为地图点在关键帧里的索引
void MapPoint::AddObservation(KeyFrame* pKF, int idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    tuple<int,int> indexes;

    if(mObservations.count(pKF)){
        indexes = mObservations[pKF];
    }
    else{
        indexes = tuple<int,int>(-1,-1);
    }

    if(pKF -> NLeft != -1 && idx >= pKF -> NLeft){
        get<1>(indexes) = idx;
    }
    else{
        get<0>(indexes) = idx;
    }

    mObservations[pKF]=indexes;

    if(!pKF->mpCamera2 && pKF->mvuRight[idx]>=0)
        nObs+=2;
    else
        nObs++;
}

/// 删除 与关键帧的 观测关系
void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            //int idx = mObservations[pKF];
            tuple<int,int> indexes = mObservations[pKF];
            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

            if(leftIndex != -1){
                if(!pKF->mpCamera2 && pKF->mvuRight[leftIndex]>=0)
                    nObs-=2;
                else
                    nObs--;
            }
            if(rightIndex != -1){
                nObs--;
            }

            mObservations.erase(pKF);

            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if(nObs<=2)
                bBad=true; // 地图点的质量不好了
        }
    }

    if(bBad)
        SetBadFlag();      // 删除该地图点
}


std::map<KeyFrame*, std::tuple<int,int>>  MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

/// 删除该地图点
void MapPoint::SetBadFlag()
{
    map<KeyFrame*, tuple<int,int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true; // 被删除
        obs = mObservations;
        mObservations.clear();
    }
    for(map<KeyFrame*, tuple<int,int>>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        int leftIndex = get<0>(mit -> second), rightIndex = get<1>(mit -> second);
        if(leftIndex != -1){
            pKF->EraseMapPointMatch(leftIndex); // 从关键帧上，删除观测关系
        }
        if(rightIndex != -1){
            pKF->EraseMapPointMatch(rightIndex);
        }
    }

    mpMap->EraseMapPoint(this); // 从地图上，删除该地图点
}

MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

/// 替换该地图点
void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,tuple<int,int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true; // 被替换
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    /// 遍历 观测到 旧地图点 的关键帧
    for(map<KeyFrame*,tuple<int,int>>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;

        tuple<int,int> indexes = mit -> second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

        // 新地图点 是否 在关键帧的观测关系中
        if(!pMP->IsInKeyFrame(pKF))
        {
            // 如果 不在，就替换 观测关系
            
            if(leftIndex != -1){
                pKF->ReplaceMapPointMatch(leftIndex, pMP);
                pMP->AddObservation(pKF,leftIndex);
            }
            if(rightIndex != -1){
                pKF->ReplaceMapPointMatch(rightIndex, pMP);
                pMP->AddObservation(pKF,rightIndex);
            }
        }
        else
        {
            // 如果 在，就删除 旧观测关系（关键帧与旧地图点的关系）
            
            if(leftIndex != -1){
                pKF->EraseMapPointMatch(leftIndex);
            }
            if(rightIndex != -1){
                pKF->EraseMapPointMatch(rightIndex);
            }
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this); // 删除 旧地图点
}

bool MapPoint::isBad()
{
    unique_lock<mutex> lock1(mMutexFeatures,std::defer_lock);
    unique_lock<mutex> lock2(mMutexPos,std::defer_lock);
    lock(lock1, lock2);

    return mbBad;
}

void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

/// 从众多观测到该地图点的特征点中挑选区分度最高的描述子
void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    /// 1.获取所有观测到该地图点的关键帧，及所在的特征点索引
    map<KeyFrame*,tuple<int,int>> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mObservations;
    }

    if(observations.empty())
        return;

    /// 2.获取对应的特征点描述子
    vDescriptors.reserve(observations.size());

    for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad()){
            tuple<int,int> indexes = mit -> second;
            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

            if(leftIndex != -1){
                vDescriptors.push_back(pKF->mDescriptors.row(leftIndex));
            }
            if(rightIndex != -1){
                vDescriptors.push_back(pKF->mDescriptors.row(rightIndex));
            }
        }
    }

    if(vDescriptors.empty())
        return;

    /// 3.Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    /// 3.寻找中值距离最小的描述子，作为所有描述子的代表
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        // 当前描述子，距离其它描述子的距离的中值
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end()); // 排序
        int median = vDists[0.5*(N-1)];    // 中值

        if(median<BestMedian) // 中值最小的描述子，区分度最高
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

tuple<int,int> MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return tuple<int,int>(-1,-1);
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

/// 更新平均观测方向以及可观测距离
void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,tuple<int,int>> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;
        observations=mObservations; // 观测到该地图点的关键帧
        pRefKF=mpRefKF;             // 生成该地图点的参考帧
        Pos = mWorldPos.clone();    // 地图点的3d坐标
    }

    if(observations.empty())
        return;

    /// 1.平均观测方向
    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
    for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        tuple<int,int> indexes = mit -> second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

        if(leftIndex != -1){
            cv::Mat Owi = pKF->GetCameraCenter();
            cv::Mat normali = mWorldPos - Owi;
            normal = normal + normali/cv::norm(normali); // 当前关键帧的观测方向
            n++;
        }
        if(rightIndex != -1){
            cv::Mat Owi = pKF->GetRightCameraCenter();
            cv::Mat normali = mWorldPos - Owi;
            normal = normal + normali/cv::norm(normali); // 当前关键帧的观测方向
            n++;
        }
    }

    /// 2.可观测距离
    cv::Mat PC = Pos - pRefKF->GetCameraCenter();
    const float dist = cv::norm(PC);

    tuple<int ,int> indexes = observations[pRefKF];
    int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
    int level;
    if(pRefKF -> NLeft == -1){
        level = pRefKF->mvKeysUn[leftIndex].octave; // 地图点所在的金字塔层数
    }
    else if(leftIndex != -1){
        level = pRefKF -> mvKeys[leftIndex].octave;
    }
    else{
        level = pRefKF -> mvKeysRight[rightIndex - pRefKF -> NLeft].octave;
    }

    //const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level]; // 当前层的尺度
    const int nLevels = pRefKF->mnScaleLevels;                     // 金字塔的层数 8

    {
        unique_lock<mutex> lock3(mMutexPos);
        
        /// 特征点的距离越近，金字塔的层数越高
        mfMaxDistance = dist*levelScaleFactor;              // 能观测到地图点的最远距离（金字塔的最底层）
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1]; // 最近距离（金字塔的最高层）
        mNormalVector = normal/n; // 平均观测方向
    }
}

void MapPoint::SetNormalVector(cv::Mat& normal)
{
    unique_lock<mutex> lock3(mMutexPos);
    mNormalVector = normal;
}

float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}

/// 预估地图点在关键帧上的层数，用于地图点融合
int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    // 距离越近，金字塔层数越高
    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

/// 根据地图点的距离，预测在金字塔的层数
int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    // 距离越近，金字塔层数越高
    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}

void MapPoint::PrintObservations()
{
    cout << "MP_OBS: MP " << mnId << endl;
    for(map<KeyFrame*,tuple<int,int>>::iterator mit=mObservations.begin(), mend=mObservations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKFi = mit->first;
        tuple<int,int> indexes = mit->second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
        cout << "--OBS in KF " << pKFi->mnId << " in map " << pKFi->GetMap()->GetId() << endl;
    }
}

Map* MapPoint::GetMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

void MapPoint::UpdateMap(Map* pMap)
{
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

void MapPoint::PreSave(set<KeyFrame*>& spKF,set<MapPoint*>& spMP)
{
    mBackupReplacedId = -1;
    if(mpReplaced && spMP.find(mpReplaced) != spMP.end())
        mBackupReplacedId = mpReplaced->mnId;

    mBackupObservationsId1.clear();
    mBackupObservationsId2.clear();
    // Save the id and position in each KF who view it
    for(std::map<KeyFrame*,std::tuple<int,int> >::const_iterator it = mObservations.begin(), end = mObservations.end(); it != end; ++it)
    {
        KeyFrame* pKFi = it->first;
        if(spKF.find(pKFi) != spKF.end())
        {
            mBackupObservationsId1[it->first->mnId] = get<0>(it->second);
            mBackupObservationsId2[it->first->mnId] = get<1>(it->second);
        }
        else
        {
            EraseObservation(pKFi);
        }
    }

    // Save the id of the reference KF
    if(spKF.find(mpRefKF) != spKF.end())
    {
        mBackupRefKFId = mpRefKF->mnId;
    }
}

void MapPoint::PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int, MapPoint*>& mpMPid)
{
    mpRefKF = mpKFid[mBackupRefKFId];
    if(!mpRefKF)
    {
        cout << "MP without KF reference " << mBackupRefKFId << "; Num obs: " << nObs << endl;
    }
    mpReplaced = static_cast<MapPoint*>(NULL);
    if(mBackupReplacedId>=0)
    {
       map<long unsigned int, MapPoint*>::iterator it = mpMPid.find(mBackupReplacedId);
       if (it != mpMPid.end())
        mpReplaced = it->second;
    }

    mObservations.clear();

    for(map<long unsigned int, int>::const_iterator it = mBackupObservationsId1.begin(), end = mBackupObservationsId1.end(); it != end; ++it)
    {
        KeyFrame* pKFi = mpKFid[it->first];
        map<long unsigned int, int>::const_iterator it2 = mBackupObservationsId2.find(it->first);
        std::tuple<int, int> indexes = tuple<int,int>(it->second,it2->second);
        if(pKFi)
        {
           mObservations[pKFi] = indexes;
        }
    }

    mBackupObservationsId1.clear();
    mBackupObservationsId2.clear();
}

} //namespace ORB_SLAM
