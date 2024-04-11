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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "ImuTypes.h"
#include<mutex>

namespace ORB_SLAM3
{

long unsigned int KeyFrame::nNextId=0;

KeyFrame::KeyFrame():
        mnFrameId(0),  mTimeStamp(0), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
        mfGridElementWidthInv(0), mfGridElementHeightInv(0),
        mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mnBALocalForMerge(0),
        mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnMergeQuery(0), mnMergeWords(0), mnBAGlobalForKF(0),
        fx(0), fy(0), cx(0), cy(0), invfx(0), invfy(0), mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0), mPlaceRecognitionScore(0),
        mbf(0), mb(0), mThDepth(0), N(0), mvKeys(static_cast<vector<cv::KeyPoint> >(NULL)), mvKeysUn(static_cast<vector<cv::KeyPoint> >(NULL)),
        mvuRight(static_cast<vector<float> >(NULL)), mvDepth(static_cast<vector<float> >(NULL)), /*mDescriptors(NULL),*/
        /*mBowVec(NULL), mFeatVec(NULL),*/ mnScaleLevels(0), mfScaleFactor(0),
        mfLogScaleFactor(0), mvScaleFactors(0), mvLevelSigma2(0),
        mvInvLevelSigma2(0), mnMinX(0), mnMinY(0), mnMaxX(0),
        mnMaxY(0), /*mK(NULL),*/  mPrevKF(static_cast<KeyFrame*>(NULL)), mNextKF(static_cast<KeyFrame*>(NULL)), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
        mbToBeErased(false), mbBad(false), mHalfBaseline(0), mbCurrentPlaceRecognition(false), mbHasHessian(false), mnMergeCorrectedForKF(0),
        NLeft(0),NRight(0), mnNumberOfOpt(0)
{

}

KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    bImu(pMap->isImuInitialized()), mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mnBALocalForMerge(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0), mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0), mPlaceRecognitionScore(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mK(F.mK), mPrevKF(NULL), mNextKF(NULL), mpImuPreintegrated(F.mpImuPreintegrated),
    mImuCalib(F.mImuCalib), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mDistCoef(F.mDistCoef), mbNotErase(false), mnDataset(F.mnDataset),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap), mbCurrentPlaceRecognition(false), mNameFile(F.mNameFile), mbHasHessian(false), mnMergeCorrectedForKF(0),
    mpCamera(F.mpCamera), mpCamera2(F.mpCamera2),
    mvLeftToRightMatch(F.mvLeftToRightMatch),mvRightToLeftMatch(F.mvRightToLeftMatch),mTlr(F.mTlr.clone()),
    mvKeysRight(F.mvKeysRight), NLeft(F.Nleft), NRight(F.Nright), mTrl(F.mTrl), mnNumberOfOpt(0)
{

    imgLeft = F.imgLeft.clone();
    imgRight = F.imgRight.clone();

    mnId=nNextId++;

    mGrid.resize(mnGridCols);
    if(F.Nleft != -1)  mGridRight.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        if(F.Nleft != -1) mGridRight[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++){
            mGrid[i][j] = F.mGrid[i][j];
            if(F.Nleft != -1){
                mGridRight[i][j] = F.mGridRight[i][j];
            }
        }
    }



    if(F.mVw.empty())
        Vw = cv::Mat::zeros(3,1,CV_32F);
    else
        Vw = F.mVw.clone();

    mImuBias = F.mImuBias;
    SetPose(F.mTcw);

    mnOriginMapId = pMap->GetId();
}

void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw;
    if (!mImuCalib.Tcb.empty())
        Owb = Rwc*mImuCalib.Tcb.rowRange(0,3).col(3)+Ow;


    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    Cw = Twc*center;
}

void KeyFrame::SetVelocity(const cv::Mat &Vw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Vw_.copyTo(Vw);
}


cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}

cv::Mat KeyFrame::GetImuPosition()
{
    unique_lock<mutex> lock(mMutexPose);
    return Owb.clone();
}

cv::Mat KeyFrame::GetImuRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.rowRange(0,3).colRange(0,3)*mImuCalib.Tcb.rowRange(0,3).colRange(0,3);
}

cv::Mat KeyFrame::GetImuPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc*mImuCalib.Tcb;
}

cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}

cv::Mat KeyFrame::GetVelocity()
{
    unique_lock<mutex> lock(mMutexPose);
    return Vw.clone();
}

/// 与 关键帧pKF 建立共视关系（共视数为weight）
void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    // 重新对共视帧进行排序
    UpdateBestCovisibles();
}

/// 按照共视数，对共视帧进行排序（从大到小）
void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        if(!vPairs[i].second->isBad())
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }
    }

    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end()); // 排好序的共视帧
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());                    // 排好序的共视数
}

set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

/// 获取与 该关键帧 存在共视关系 的所有关键帧
vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames; // 已经排好序的
}

/// 获取与 该关键帧 存在共视关系 的前N个关键帧
vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else // 共视帧已经排好序了，取前N个即可
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
    {
        return vector<KeyFrame*>();
    }

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);

    if(it==mvOrderedWeights.end() && mvOrderedWeights.back() < w)
    {
        return vector<KeyFrame*>();
    }
    else
    {
        int n = it-mvOrderedWeights.begin();
        //cout << "n = " << n << endl;
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

/// 获取与 关键帧pKF 的共视数
int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

int KeyFrame::GetNumberMPs()
{
    unique_lock<mutex> lock(mMutexFeatures);
    int numberMPs = 0;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        numberMPs++;
    }
    return numberMPs;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

void KeyFrame::EraseMapPointMatch(const int &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

/// 删除 与 地图点pMP 的观测关系
void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    tuple<size_t,size_t> indexes = pMP->GetIndexInKeyFrame(this);
    size_t leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
    if(leftIndex != -1)
        mvpMapPoints[leftIndex]=static_cast<MapPoint*>(NULL);
    if(rightIndex != -1)
        mvpMapPoints[rightIndex]=static_cast<MapPoint*>(NULL);
}

/// 用 新的地图点pMP，替代 在idx位置 的旧地图点
void KeyFrame::ReplaceMapPointMatch(const int &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

/// 关键帧中，被观测次数超过 minObs次 的地图点数量
int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

/// 更新 Covisibility graph, essential graph 和 spanning tree
void KeyFrame::UpdateConnections(bool upParent)
{
    map<KeyFrame*,int> KFcounter; // 其他关键帧，与当前关键帧的共视数

    vector<MapPoint*> vpMP;       // 当前关键帧的所有地图点

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    /// 1.遍历当前关键帧的所有地图点，统计 其他关键帧 与 当前关键帧 的共视数
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations(); // 观测到该地图点的所有关键帧

        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId || mit->first->isBad() || mit->first->GetMap() != mpMap)
                continue;
            KFcounter[mit->first]++; // 统计共视数

        }
    }

    // This should not happen
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL; // 与 当前关键帧 共视数最大的关键帧
    int th = 15;

    /// 2.建立共视边，并且寻找共视程度最高的关键帧 作为父关键帧
    vector<pair<int,KeyFrame*> > vPairs; // 共视数放前面，便于排序（共视数大于15）
    vPairs.reserve(KFcounter.size());
    if(!upParent)
        cout << "UPDATE_CONN: current KF " << mnId << endl;
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(!upParent)
            cout << "  UPDATE_CONN: KF " << mit->first->mnId << " ; num matches: " << mit->second << endl;
        if(mit->second>nmax)
        {
            nmax=mit->second;
            pKFmax=mit->first; // 共视程度最高的关键帧
        }
        if(mit->second>=th)    // 共视数大于15, 建立共视边
        {
            vPairs.push_back(make_pair(mit->second,mit->first));
            (mit->first)->AddConnection(this,mit->second);
        }
    }

    // 如果共视数均小于15, 则使用共视数最多的建立一个共视边
    if(vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    /// 3.对共视边进行排序，按照共视数从大到小的顺序
    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = KFcounter;                                     // 共视边（未排序）
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end()); // 排好序的共视帧（从大到小）
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());                    // 排好序的共视数（从大到小）

//        if(mbFirstConnection && mnId!=mpMap->GetInitKFid())
//        {
//            mpParent = mvpOrderedConnectedKeyFrames.front();
//            mpParent->AddChild(this);
//            mbFirstConnection = false;
//        }

        /// 如果当前关键帧刚加入，将共视程度最高的关键帧，作为 父关键帧
        if(mbFirstConnection && mnId!=mpMap->GetInitKFid())
        {
            /*if(!mpParent || mpParent->GetParent() != this)
            {
                KeyFrame* pBestParent = static_cast<KeyFrame*>(NULL);
                for(KeyFrame* pKFi : mvpOrderedConnectedKeyFrames)
                {
                    if(pKFi->GetParent() || pKFi->mnId == mpMap->GetInitKFid())
                    {
                        pBestParent = pKFi;
                        break;
                    }
                }
                if(!pBestParent)
                {
                    cout << "It can't be a covisible KF without Parent" << endl << endl;
                    return;
                }
                mpParent = pBestParent;
                mpParent->AddChild(this);
                mbFirstConnection = false;
            }*/
            // cout << "udt.conn.id: " << mnId << endl;
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
//    if(!mpParent && mpParent != this)
//        mpParent->EraseChild(this);
    if(pKF == this)
    {
        cout << "ERROR: Change parent KF, the parent and child are the same KF" << endl;
        throw std::invalid_argument("The parent and child can not be the same");
    }

    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::SetFirstConnection(bool bFirst)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbFirstConnection=bFirst;
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::AddMergeEdge(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspMergeEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetMergeEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspMergeEdges;
}

/// 正在被某线程使用，设置 不可以被删除
void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

/// 使用完毕，设置 可以被删除
void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag(); /// 进行删除
    }
}

/// 从地图中删除该关键帧
void KeyFrame::SetBadFlag()
{   
    // std::cout << "Erasing KF..." << std::endl;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==mpMap->GetInitKFid())
        {
            //std::cout << "KF.BADFLAG-> KF 0!!" << std::endl;
            return;
        }
        else if(mbNotErase)      /// 该关键帧正在被某线程使用，不可以被删除
        {
            //std::cout << "KF.BADFLAG-> mbNotErase!!" << std::endl;
            mbToBeErased = true; /// 使用完毕后，再进行删除
            return;
        }
        if(!mpParent)
        {
            //cout << "KF.BADFLAG-> There is not parent, but it is not the first KF in the map" << endl;
            //cout << "KF.BADFLAG-> KF: " << mnId << "; first KF: " << mpMap->GetInitKFid() << endl;
        }
    }
    //std::cout << "KF.BADFLAG-> Erasing KF..." << std::endl;

    /// 1.让 其他共视帧 删除 与自己的共视关系
    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
    {
        mit->first->EraseConnection(this);
    }
    //std::cout << "KF.BADFLAG-> Connection erased..." << std::endl;

    /// 2.让 地图点 删除 与自己的观测关系
    for(size_t i=0; i<mvpMapPoints.size(); i++)
    {
        if(mvpMapPoints[i])
        {
            mvpMapPoints[i]->EraseObservation(this);
            // nDeletedPoints++;
        }
    }
    // cout << "nDeletedPoints: " << nDeletedPoints << endl;
    //std::cout << "KF.BADFLAG-> Observations deleted..." << std::endl;

    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        /// 3.清空自己与其它关键帧之间的联系
        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        /// 4.Update Spanning Tree
        set<KeyFrame*> sParentCandidates;       // 父关键帧 候选列表，用来给 子关键帧 寻找 新的父关键帧
        if(mpParent)
            sParentCandidates.insert(mpParent); // 当前关键帧（this）的父关键帧（祖关键帧），作为候选
        //std::cout << "KF.BADFLAG-> Initially there are " << sParentCandidates.size() << " candidates" << std::endl;

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        /// 4.1给 子关键帧 寻找 新的父关键帧（每次循环只给一个 子关键帧，确定它的 父关键帧，类似于最小生成树的 让树长大）
        while(!mspChildrens.empty())
        {
            bool bContinue = false; // 如果本次循环，没有找到任何一组结果（子关键帧、父关键帧），则退出循环

            // 单次循环找到的一组结果
            int max = -1;
            KeyFrame* pC; // 子关键帧
            KeyFrame* pP; // 父关键帧

            // 遍历所有的子关键帧
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                KeyFrame* pKF = *sit; // 子关键帧
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                // 因为共视关系已经清除了，所以 子关键帧 的共视帧列表里面没有 当前关键帧（this）
                // 遍历 子关键帧 的共视帧列表（有序的）
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    // 遍历 父关键帧 候选列表，第一轮只有 祖关键帧
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        if(vpConnected[i]->mnId == (*spcit)->mnId) // 候选关键帧 是 子关键帧的共视帧
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max) // 选择共视程度最高的一组结果（子关键帧、父关键帧）
                            {
                                pC = pKF;            // 当前子关键帧
                                pP = vpConnected[i]; // 对应的父关键帧
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }
            //std::cout << "KF.BADFLAG-> Find most similar children" << std::endl;

            if(bContinue)
            {
                if(pC->mnId == pP->mnId)
                {
                    /*cout << "ERROR: The parent and son can't be the same KF. ID: " << pC->mnId << endl;
                    cout << "Current KF: " << mnId << endl;
                    cout << "Parent of the map: " << endl;*/
                }
                pC->ChangeParent(pP);         // 子关键帧、父关键帧
                sParentCandidates.insert(pC); // 已经找到 父关键帧 的子关键帧，也作为 候选关键帧，继续循环
                mspChildrens.erase(pC);
            }
            else
                break; // 本次循环，没有找到任何一组结果（子关键帧、父关键帧），退出循环
        }
        //std::cout << "KF.BADFLAG-> Apply change of parent to children" << std::endl;

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        /// 4.2将 祖关键帧 设为 剩余子关键帧的 父关键帧
        if(!mspChildrens.empty())
        {
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }
        }
        //std::cout << "KF.BADFLAG-> Apply change to its parent" << std::endl;

        /// 5.删除 自己 与 父关键帧 的链接
        if(mpParent){
            mpParent->EraseChild(this);
            mTcp = Tcw*mpParent->GetPoseInverse();
        }
        else
        {
            //cout << "Error: KF haven't got a parent, it is imposible reach this code point without him" << endl;
        }
        mbBad = true;
    }

    /// 6.将自己 从地图和关键帧数据库中 删除
    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}

/// 寻找在一定范围内的特征点（圆形范围内）
vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = (NLeft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}

/// 计算深度中位数，用于尺度归一化
float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths; // 深度列表，用于排序之后寻找中位数
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2,3); // 深度
    for(int i=0; i<N; i++)
    {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw)+zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());  // 排序

    return vDepths[(vDepths.size()-1)/q]; // 返回深度的中位数
}

void KeyFrame::SetNewBias(const IMU::Bias &b)
{
    unique_lock<mutex> lock(mMutexPose);
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

cv::Mat KeyFrame::GetGyroBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return (cv::Mat_<float>(3,1) << mImuBias.bwx, mImuBias.bwy, mImuBias.bwz);
}

cv::Mat KeyFrame::GetAccBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return (cv::Mat_<float>(3,1) << mImuBias.bax, mImuBias.bay, mImuBias.baz);
}

IMU::Bias KeyFrame::GetImuBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return mImuBias;
}

Map* KeyFrame::GetMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

void KeyFrame::UpdateMap(Map* pMap)
{
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

void KeyFrame::PreSave(set<KeyFrame*>& spKF,set<MapPoint*>& spMP, set<GeometricCamera*>& spCam)
{
    // Save the id of each MapPoint in this KF, there can be null pointer in the vector
    mvBackupMapPointsId.clear();
    mvBackupMapPointsId.reserve(N);
    for(int i = 0; i < N; ++i)
    {

        if(mvpMapPoints[i] && spMP.find(mvpMapPoints[i]) != spMP.end()) // Checks if the element is not null
            mvBackupMapPointsId.push_back(mvpMapPoints[i]->mnId);
        else // If the element is null his value is -1 because all the id are positives
            mvBackupMapPointsId.push_back(-1);
    }
    //cout << "KeyFrame: ID from MapPoints stored" << endl;
    // Save the id of each connected KF with it weight
    mBackupConnectedKeyFrameIdWeights.clear();
    for(std::map<KeyFrame*,int>::const_iterator it = mConnectedKeyFrameWeights.begin(), end = mConnectedKeyFrameWeights.end(); it != end; ++it)
    {
        if(spKF.find(it->first) != spKF.end())
            mBackupConnectedKeyFrameIdWeights[it->first->mnId] = it->second;
    }
    //cout << "KeyFrame: ID from connected KFs stored" << endl;
    // Save the parent id
    mBackupParentId = -1;
    if(mpParent && spKF.find(mpParent) != spKF.end())
        mBackupParentId = mpParent->mnId;
    //cout << "KeyFrame: ID from Parent KF stored" << endl;
    // Save the id of the childrens KF
    mvBackupChildrensId.clear();
    mvBackupChildrensId.reserve(mspChildrens.size());
    for(KeyFrame* pKFi : mspChildrens)
    {
        if(spKF.find(pKFi) != spKF.end())
            mvBackupChildrensId.push_back(pKFi->mnId);
    }
    //cout << "KeyFrame: ID from Children KFs stored" << endl;
    // Save the id of the loop edge KF
    mvBackupLoopEdgesId.clear();
    mvBackupLoopEdgesId.reserve(mspLoopEdges.size());
    for(KeyFrame* pKFi : mspLoopEdges)
    {
        if(spKF.find(pKFi) != spKF.end())
            mvBackupLoopEdgesId.push_back(pKFi->mnId);
    }
    //cout << "KeyFrame: ID from Loop KFs stored" << endl;
    // Save the id of the merge edge KF
    mvBackupMergeEdgesId.clear();
    mvBackupMergeEdgesId.reserve(mspMergeEdges.size());
    for(KeyFrame* pKFi : mspMergeEdges)
    {
        if(spKF.find(pKFi) != spKF.end())
            mvBackupMergeEdgesId.push_back(pKFi->mnId);
    }
    //cout << "KeyFrame: ID from Merge KFs stored" << endl;

    //Camera data
    mnBackupIdCamera = -1;
    if(mpCamera && spCam.find(mpCamera) != spCam.end())
        mnBackupIdCamera = mpCamera->GetId();
    //cout << "KeyFrame: ID from Camera1 stored; " << mnBackupIdCamera << endl;

    mnBackupIdCamera2 = -1;
    if(mpCamera2 && spCam.find(mpCamera2) != spCam.end())
        mnBackupIdCamera2 = mpCamera2->GetId();
    //cout << "KeyFrame: ID from Camera2 stored; " << mnBackupIdCamera2 << endl;

    //Inertial data
    mBackupPrevKFId = -1;
    if(mPrevKF && spKF.find(mPrevKF) != spKF.end())
        mBackupPrevKFId = mPrevKF->mnId;
    //cout << "KeyFrame: ID from Prev KF stored" << endl;
    mBackupNextKFId = -1;
    if(mNextKF && spKF.find(mNextKF) != spKF.end())
        mBackupNextKFId = mNextKF->mnId;
    //cout << "KeyFrame: ID from NextKF stored" << endl;
    if(mpImuPreintegrated)
        mBackupImuPreintegrated.CopyFrom(mpImuPreintegrated);
    //cout << "KeyFrame: Imu Preintegrated stored" << endl;
}

void KeyFrame::PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int, MapPoint*>& mpMPid, map<unsigned int, GeometricCamera*>& mpCamId){
    // Rebuild the empty variables

    // Pose
    SetPose(Tcw);

    // Reference reconstruction
    // Each MapPoint sight from this KeyFrame
    mvpMapPoints.clear();
    mvpMapPoints.resize(N);
    for(int i=0; i<N; ++i)
    {
        if(mvBackupMapPointsId[i] != -1)
            mvpMapPoints[i] = mpMPid[mvBackupMapPointsId[i]];
        else
            mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
    }

    // Conected KeyFrames with him weight
    mConnectedKeyFrameWeights.clear();
    for(map<long unsigned int, int>::const_iterator it = mBackupConnectedKeyFrameIdWeights.begin(), end = mBackupConnectedKeyFrameIdWeights.end();
        it != end; ++it)
    {
        KeyFrame* pKFi = mpKFid[it->first];
        mConnectedKeyFrameWeights[pKFi] = it->second;
    }

    // Restore parent KeyFrame
    if(mBackupParentId>=0)
        mpParent = mpKFid[mBackupParentId];

    // KeyFrame childrens
    mspChildrens.clear();
    for(vector<long unsigned int>::const_iterator it = mvBackupChildrensId.begin(), end = mvBackupChildrensId.end(); it!=end; ++it)
    {
        mspChildrens.insert(mpKFid[*it]);
    }

    // Loop edge KeyFrame
    mspLoopEdges.clear();
    for(vector<long unsigned int>::const_iterator it = mvBackupLoopEdgesId.begin(), end = mvBackupLoopEdgesId.end(); it != end; ++it)
    {
        mspLoopEdges.insert(mpKFid[*it]);
    }

    // Merge edge KeyFrame
    mspMergeEdges.clear();
    for(vector<long unsigned int>::const_iterator it = mvBackupMergeEdgesId.begin(), end = mvBackupMergeEdgesId.end(); it != end; ++it)
    {
        mspMergeEdges.insert(mpKFid[*it]);
    }

    //Camera data
    if(mnBackupIdCamera >= 0)
    {
        mpCamera = mpCamId[mnBackupIdCamera];
    }
    if(mnBackupIdCamera2 >= 0)
    {
        mpCamera2 = mpCamId[mnBackupIdCamera2];
    }

    //Inertial data
    if(mBackupPrevKFId != -1)
    {
        mPrevKF = mpKFid[mBackupPrevKFId];
    }
    if(mBackupNextKFId != -1)
    {
        mNextKF = mpKFid[mBackupNextKFId];
    }
    mpImuPreintegrated = &mBackupImuPreintegrated;


    // Remove all backup container
    mvBackupMapPointsId.clear();
    mBackupConnectedKeyFrameIdWeights.clear();
    mvBackupChildrensId.clear();
    mvBackupLoopEdgesId.clear();

    UpdateBestCovisibles();

    //ComputeSceneMedianDepth();
}

bool KeyFrame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos();
    cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = Tcw.rowRange(0, 3).col(3);

    // 3D in camera coordinates
    cv::Mat Pc = Rcw*P+tcw;
    float &PcX = Pc.at<float>(0);
    float &PcY= Pc.at<float>(1);
    float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    // cout << "c";

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if(mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    float u_distort = x_distort * fx + cx;
    float v_distort = y_distort * fy + cy;

    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

bool KeyFrame::ProjectPointUnDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 
    cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
    // 3D in camera coordinates
    cv::Mat Pc = Rcw*P+tcw;
    float &PcX = Pc.at<float>(0);
    float &PcY= Pc.at<float>(1);
    float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    // cout << "c";

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    kp = cv::Point2f(u, v);

    return true;
}

cv::Mat KeyFrame::GetRightPose() {
    unique_lock<mutex> lock(mMutexPose);

    cv::Mat Rrl = mTlr.rowRange(0,3).colRange(0,3).t();
    cv::Mat Rlw = Tcw.rowRange(0,3).colRange(0,3).clone();
    cv::Mat Rrw = Rrl * Rlw;

    cv::Mat tlw = Tcw.rowRange(0,3).col(3).clone();
    cv::Mat trl = - Rrl * mTlr.rowRange(0,3).col(3);

    cv::Mat trw = Rrl * tlw + trl;

    cv::Mat Trw;
    cv::hconcat(Rrw,trw,Trw);

    return Trw.clone();
}

cv::Mat KeyFrame::GetRightPoseInverse() {
    unique_lock<mutex> lock(mMutexPose);
    cv::Mat Rrl = mTlr.rowRange(0,3).colRange(0,3).t();
    cv::Mat Rlw = Tcw.rowRange(0,3).colRange(0,3).clone();
    cv::Mat Rwr = (Rrl * Rlw).t();

    cv::Mat Rwl = Tcw.rowRange(0,3).colRange(0,3).t();
    cv::Mat tlr = mTlr.rowRange(0,3).col(3);
    cv::Mat twl = GetCameraCenter();

    cv::Mat twr = Rwl * tlr + twl;

    cv::Mat Twr;
    cv::hconcat(Rwr,twr,Twr);

    return Twr.clone();
}

cv::Mat KeyFrame::GetRightPoseInverseH() {
    unique_lock<mutex> lock(mMutexPose);
    cv::Mat Rrl = mTlr.rowRange(0,3).colRange(0,3).t();
    cv::Mat Rlw = Tcw.rowRange(0,3).colRange(0,3).clone();
    cv::Mat Rwr = (Rrl * Rlw).t();

    cv::Mat Rwl = Tcw.rowRange(0,3).colRange(0,3).t();
    cv::Mat tlr = mTlr.rowRange(0,3).col(3);
    cv::Mat twl = Ow.clone();

    cv::Mat twr = Rwl * tlr + twl;

    cv::Mat Twr;
    cv::hconcat(Rwr,twr,Twr);
    cv::Mat h(1,4,CV_32F,cv::Scalar(0.0f)); h.at<float>(3) = 1.0f;
    cv::vconcat(Twr,h,Twr);

    return Twr.clone();
}

cv::Mat KeyFrame::GetRightCameraCenter() {
    unique_lock<mutex> lock(mMutexPose);
    cv::Mat Rwl = Tcw.rowRange(0,3).colRange(0,3).t();
    cv::Mat tlr = mTlr.rowRange(0,3).col(3);
    cv::Mat twl = Ow.clone();

    cv::Mat twr = Rwl * tlr + twl;

    return twr.clone();
}

cv::Mat KeyFrame::GetRightRotation() {
    unique_lock<mutex> lock(mMutexPose);
    cv::Mat Rrl = mTlr.rowRange(0,3).colRange(0,3).t();
    cv::Mat Rlw = Tcw.rowRange(0,3).colRange(0,3).clone();
    cv::Mat Rrw = Rrl * Rlw;

    return Rrw.clone();

}

cv::Mat KeyFrame::GetRightTranslation() {
    unique_lock<mutex> lock(mMutexPose);
    cv::Mat Rrl = mTlr.rowRange(0,3).colRange(0,3).t();
    cv::Mat tlw = Tcw.rowRange(0,3).col(3).clone();
    cv::Mat trl = - Rrl * mTlr.rowRange(0,3).col(3);

    cv::Mat trw = Rrl * tlw + trl;

    return trw.clone();
}

void KeyFrame::SetORBVocabulary(ORBVocabulary* pORBVoc)
{
    mpORBvocabulary = pORBVoc;
}

void KeyFrame::SetKeyFrameDatabase(KeyFrameDatabase* pKFDB)
{
    mpKeyFrameDB = pKFDB;
}

} //namespace ORB_SLAM
