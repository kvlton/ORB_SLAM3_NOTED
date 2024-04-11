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


#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM3
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;       // 误匹配的阈值
const int ORBmatcher::HISTO_LENGTH = 30; // 角度直方图，用于剔除误匹配

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

/// 1.对当前帧视野范围内的局部地图点，通过投影进行特征匹配，用于局部地图跟踪（当前帧，局部地图点，扩大搜索范围的倍数）
int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
{
    // 匹配到的特征数量
    int nmatches=0, left = 0, right = 0;

    // 是否扩大搜索范围
    const bool bFactor = th!=1.0;

    // 遍历局部地图中的地图点
    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];
        if(!pMP->mbTrackInView && !pMP->mbTrackInViewR) // 是否在视野范围内
            continue;

        if(bFarPoints && pMP->mTrackDepth>thFarPoints)
            continue;

        if(pMP->isBad())
            continue;

        if(pMP->mbTrackInView)
        {
            // 预测所在的金字塔层数
            const int &nPredictedLevel = pMP->mnTrackScaleLevel;

            // The size of the window will depend on the viewing direction
            // 搜索范围
            float r = RadiusByViewingCos(pMP->mTrackViewCos);

            if(bFactor)
                r*=th;

            // 搜索范围覆盖的格子
            const vector<size_t> vIndices =
                    F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

            if(!vIndices.empty()){
                // 寻找和局部地图点最优、次优的匹配
                const cv::Mat MPdescriptor = pMP->GetDescriptor();

                int bestDist=256;  // 最优
                int bestLevel= -1;
                int bestDist2=256; // 次优
                int bestLevel2 = -1;
                int bestIdx =-1 ;

                // Get best and second matches with near keypoints
                for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                {
                    // 当前特征点ID
                    const size_t idx = *vit;

                    // 已经匹配到局部地图点了，在右图的位置不在搜索范围内
                    if(F.mvpMapPoints[idx])
                        if(F.mvpMapPoints[idx]->Observations()>0)
                            continue;

                    if(F.Nleft == -1 && F.mvuRight[idx]>0)
                    {
                        const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                        if(er>r*F.mvScaleFactors[nPredictedLevel])
                            continue;
                    }

                    // 当前特征点与局部地图点的距离
                    const cv::Mat &d = F.mDescriptors.row(idx);

                    const int dist = DescriptorDistance(MPdescriptor,d);

                    if(dist<bestDist)
                    {
                        bestDist2=bestDist; // 次优距离
                        bestDist=dist;      // 最优距离
                        bestLevel2 = bestLevel;
                        bestLevel = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                    : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                      : F.mvKeysRight[idx - F.Nleft].octave;
                        bestIdx=idx;
                    }
                    else if(dist<bestDist2)
                    {
                        bestLevel2 = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                     : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                       : F.mvKeysRight[idx - F.Nleft].octave;
                        bestDist2=dist;     // 次优距离
                    }
                }

                // Apply ratio to second match (only if best and second are in the same scale level)
                // 剔除误匹配（最优距离的阈值，最优与次优的差距）
                if(bestDist<=TH_HIGH) // 最优距离超过100表示为误匹配
                {
                    // 最优与次优差距不大，表示匹配的不好
                    if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                        continue;

                    if(bestLevel!=bestLevel2 || bestDist<=mfNNratio*bestDist2){
                        F.mvpMapPoints[bestIdx]=pMP;

                        if(F.Nleft != -1 && F.mvLeftToRightMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                            F.mvpMapPoints[F.mvLeftToRightMatch[bestIdx] + F.Nleft] = pMP;
                            nmatches++;
                            right++;
                        }

                        nmatches++;
                        left++;
                    }
                }
            }
        }

        if(F.Nleft != -1 && pMP->mbTrackInViewR){
            const int &nPredictedLevel = pMP->mnTrackScaleLevelR;
            if(nPredictedLevel != -1){
                float r = RadiusByViewingCos(pMP->mTrackViewCosR);

                const vector<size_t> vIndices =
                        F.GetFeaturesInArea(pMP->mTrackProjXR,pMP->mTrackProjYR,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel,true);

                if(vIndices.empty())
                    continue;

                const cv::Mat MPdescriptor = pMP->GetDescriptor();

                int bestDist=256;
                int bestLevel= -1;
                int bestDist2=256;
                int bestLevel2 = -1;
                int bestIdx =-1 ;

                // Get best and second matches with near keypoints
                for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                {
                    const size_t idx = *vit;

                    if(F.mvpMapPoints[idx + F.Nleft])
                        if(F.mvpMapPoints[idx + F.Nleft]->Observations()>0)
                            continue;


                    const cv::Mat &d = F.mDescriptors.row(idx + F.Nleft);

                    const int dist = DescriptorDistance(MPdescriptor,d);

                    if(dist<bestDist)
                    {
                        bestDist2=bestDist;
                        bestDist=dist;
                        bestLevel2 = bestLevel;
                        bestLevel = F.mvKeysRight[idx].octave;
                        bestIdx=idx;
                    }
                    else if(dist<bestDist2)
                    {
                        bestLevel2 = F.mvKeysRight[idx].octave;
                        bestDist2=dist;
                    }
                }

                // Apply ratio to second match (only if best and second are in the same scale level)
                if(bestDist<=TH_HIGH)
                {
                    if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                        continue;

                    if(F.Nleft != -1 && F.mvRightToLeftMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                        F.mvpMapPoints[F.mvRightToLeftMatch[bestIdx]] = pMP;
                        nmatches++;
                        left++;
                    }


                    F.mvpMapPoints[bestIdx + F.Nleft]=pMP;
                    nmatches++;
                    right++;
                }
            }
        }
    }
    
    // 返回最终匹配到的特征点数
    return nmatches;
}

/// 局部地图跟踪，投影匹配的范围
float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5; // 视角接近0度
    else
        return 4.0;
}

/// 判断匹配到的两个关键点是否满足对极约束
bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2, const bool b1)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    if(!b1)
        return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
    else
        return dsqr<6.63*pKF2->mvLevelSigma2[kp2.octave];
}

bool ORBmatcher::CheckDistEpipolarLine2(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF2, const float unc)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    if(unc==1.f)
        return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
    else
        return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave]*unc;
}

/// 1.根据词袋，对当前帧和参考帧进行特征匹配（对参考帧的地图点进行匹配）
int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    // 关键帧的地图点
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

    // 当前帧匹配到的地图点，未被匹配到的为NULL
    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

    // 关键帧特征点的正向索引
    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches=0; // 匹配到的特征数量

    // 建立角度变化直方图，用于后面剔除误匹配
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    // 根据当前帧和关键帧的正向索引map，进行特征匹配
    while(KFit != KFend && Fit != Fend)
    {
        // 正向索引节点相同，即特征都在同一颗子树下
        if(KFit->first == Fit->first)
        {
            // 当前子树下，特征ID列表
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;

            // 遍历关键帧的特征
            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            {
                // 当前关键帧特征，是否是好的地图点
                const unsigned int realIdxKF = vIndicesKF[iKF];

                MapPoint* pMP = vpMapPointsKF[realIdxKF];

                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;

                // 当前关键帧特征的描述子
                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);

                // 遍历当前帧的特征，寻找与关键帧特征最优和次优的匹配
                int bestDist1=256; // 最优距离
                int bestIdxF =-1 ; // 最优特征ID
                int bestDist2=256; // 次优距离

                int bestDist1R=256;
                int bestIdxFR =-1 ;
                int bestDist2R=256;

                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
                    if(F.Nleft == -1){
                        // 当前特征的ID
                        const unsigned int realIdxF = vIndicesF[iF];

                        // 当前特征已经被匹配过了，不再匹配
                        if(vpMapPointMatches[realIdxF])
                            continue;

                        // 当前特征的描述子
                        const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                        // 当前特征与关键帧特征的距离
                        const int dist =  DescriptorDistance(dKF,dF);

                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1; // 次优距离
                            bestDist1=dist;      // 最优距离
                            bestIdxF=realIdxF;   // 最优特征ID
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;      // 次优距离
                        }
                    }
                    else{
                        const unsigned int realIdxF = vIndicesF[iF];

                        if(vpMapPointMatches[realIdxF])
                            continue;

                        const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                        const int dist =  DescriptorDistance(dKF,dF);

                        if(realIdxF < F.Nleft && dist<bestDist1){
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdxF=realIdxF;
                        }
                        else if(realIdxF < F.Nleft && dist<bestDist2){
                            bestDist2=dist;
                        }

                        if(realIdxF >= F.Nleft && dist<bestDist1R){
                            bestDist2R=bestDist1R;
                            bestDist1R=dist;
                            bestIdxFR=realIdxF;
                        }
                        else if(realIdxF >= F.Nleft && dist<bestDist2R){
                            bestDist2R=dist;
                        }
                    }

                }

                // 剔除误匹配（最优距离的阈值，最优和次优的差距，特征点的角度变化应该一致）
                if(bestDist1<=TH_LOW) // 最优距离超过50表示为误匹配
                {
                    // 最优匹配和次优匹配，相差越大表示匹配越好
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        // 当前帧匹配到的地图点
                        vpMapPointMatches[bestIdxF]=pMP;

                        const cv::KeyPoint &kp =
                                (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                            : pKF -> mvKeys[realIdxKF];
                        
                        // 匹配特征的角度变化直方图，用于剔除误匹配
                        if(mbCheckOrientation)
                        {
                            cv::KeyPoint &Fkp =
                                    (!pKF->mpCamera2 || F.Nleft == -1) ? F.mvKeys[bestIdxF] :
                                    (bestIdxF >= F.Nleft) ? F.mvKeysRight[bestIdxF - F.Nleft]
                                                          : F.mvKeys[bestIdxF];

                            // 匹配特征 的角度变化（关键帧->当前帧）
                            float rot = kp.angle-Fkp.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }

                    if(bestDist1R<=TH_LOW)
                    {
                        if(static_cast<float>(bestDist1R)<mfNNratio*static_cast<float>(bestDist2R) || true)
                        {
                            vpMapPointMatches[bestIdxFR]=pMP;

                            const cv::KeyPoint &kp =
                                    (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                    (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                                : pKF -> mvKeys[realIdxKF];

                            if(mbCheckOrientation)
                            {
                                cv::KeyPoint &Fkp =
                                        (!F.mpCamera2) ? F.mvKeys[bestIdxFR] :
                                        (bestIdxFR >= F.Nleft) ? F.mvKeysRight[bestIdxFR - F.Nleft]
                                                               : F.mvKeys[bestIdxFR];

                                float rot = kp.angle-Fkp.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdxFR);
                            }
                            nmatches++;
                        }
                    }
                }

            }

            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
            // map内部是有序的
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }

    // 根据角度变化直方图，剔除误匹配
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        // 计算直方图中，频数最高的几个组（一般会提取出头尾两个分组 -30->30度）
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            
            // 当特征的角度变化没有在这个最高范围，则认为是误匹配，剔除
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

/// 4.闭环检测中，通过sim3投影匹配，在候选帧附近寻找更多的匹配点
/// 参数：关键帧，sim3变换，待匹配的所有地图点，匹配上的地图点，阈值
int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints,
                                   vector<MapPoint*> &vpMatched, int th, float ratioHamming)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw: Scw = s * Tcw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL)); // 已经匹配上的地图点列表，有些地图点是空的（未匹配上的 外点）

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    /// 遍历 待匹配的地图点 列表
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP]; // 地图点

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP)) // 地图点已经匹配上了其他的特征点
            continue;

        /// Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos(); // 3d世界坐标

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;       // 3d相机坐标

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        /// Project into Image
        const float x = p3Dc.at<float>(0);
        const float y = p3Dc.at<float>(1);
        const float z = p3Dc.at<float>(2);

        const cv::Point2f uv = pKF->mpCamera->project(cv::Point3f(x,y,z)); // 2d像素坐标

        // Point must be inside the image
        if(!pKF->IsInImage(uv.x,uv.y))
            continue;

        /// Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        /// Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        /// 预估地图点在关键帧上的层数，并寻找可能的匹配点
        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x,uv.y,radius);

        if(vIndices.empty())
            continue;

        /// 寻找最优匹配
        
        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor(); // 地图点描述子

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx]) // 特征点已经匹配上了其他的地图点
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave; // 特征点描述子

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF); // 描述子之间的距离

            if(dist<bestDist)
            {
                bestDist = dist; // 最优距离
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW*ratioHamming) // 最优距离超过50 表示为误匹配
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    return nmatches;
}

int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, const std::vector<KeyFrame*> &vpPointsKFs,
                       std::vector<MapPoint*> &vpMatched, std::vector<KeyFrame*> &vpMatchedKF, int th, float ratioHamming)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];
        KeyFrame* pKFi = vpPointsKFs[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW*ratioHamming)
        {
            vpMatched[bestIdx] = pMP;
            vpMatchedKF[bestIdx] = pKFi;
            nmatches++;
        }

    }

    return nmatches;
}

/// 单目初始化的特征匹配
int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches=0; // 匹配到的特征点数
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

    // 建立角度变化直方图，用于后面剔除误匹配
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1); // 防止单个特征点 重复匹配

    // 遍历 第一张图的关键点
    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        if(level1>0) /// 只对第0层的特征点进行匹配
            continue;

        // 第二张图 在搜索范围内的 关键点
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

        if(vIndices2.empty())
            continue;

        // 寻找最优、次优的匹配
        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;  // 最优
        int bestDist2 = INT_MAX; // 次优
        int bestIdx2 = -1;

        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            // 两个描述子之间的距离
            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist; // 次优
                bestDist=dist;      // 最优
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;     // 次优
            }
        }

        // 剔除误匹配：最优距离的阈值，最优和次优的差距，特征点的角度变化
        if(bestDist<=TH_LOW) // 最优距离超过50表示为误匹配
        {
            // 最优匹配和次优匹配，相差越大表示匹配越好
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                // 匹配到的特征点，已经匹配过了
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                
                // 匹配成功
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

                // 匹配特征的角度变化直方图，用于剔除误匹配
                if(mbCheckOrientation)
                {
                    // 匹配特征 的角度变化
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    
                    // bug
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

    // 根据角度变化直方图，剔除误匹配的特征点
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        // 计算直方图中，频数最高的几个组（一般会提取出头尾两个分组 -30->30度）
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            
            // 当特征点的角度变化没有在这个最高范围，则认为是误匹配，剔除
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}

/// 2.在闭环检测中，当前关键帧pKF1 与 候选关键帧pKF2，通过词袋进行特征匹配（地图点之间的匹配）
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    // 关键帧1
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    // 关键帧2
    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    // 匹配结果
    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    // 建立角度变化直方图，用于后面剔除误匹配
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

     // 匹配到的特征数量
    int nmatches = 0;

    // 正向索引的迭代器
    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        // 正向索引节点相同，即特征都在同一颗子树下
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1]; // 特征点1
                if(pKF1 -> NLeft != -1 && idx1 >= pKF1 -> mvKeysUn.size()){
                    continue;
                }

                MapPoint* pMP1 = vpMapPoints1[idx1];  // 地图点1
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1); // 描述子1

                // 遍历 关键帧2 的特征，寻找最优和次优的匹配
                int bestDist1=256; // 最优距离
                int bestIdx2 =-1 ; // 最优特征Id，特征点2 的Id
                int bestDist2=256; // 次优距离
                
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2]; // 特征点2

                    if(pKF2 -> NLeft != -1 && idx2 >= pKF2 -> mvKeysUn.size()){
                        continue;
                    }

                    MapPoint* pMP2 = vpMapPoints2[idx2];  // 地图点2

                    if(vbMatched2[idx2] || !pMP2) // 已经被匹配上了，或者没有地图点
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2); // 描述子2

                    // 当前特征与关键帧特征的距离
                    int dist = DescriptorDistance(d1,d2);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1; // 次优距离
                        bestDist1=dist;      // 最优距离
                        bestIdx2=idx2;       // 最优Id
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;      // 次优距离
                    }
                }

                // 剔除误匹配（最优距离的阈值，最优和次优的差距，特征点的角度变化应该一致）
                if(bestDist1<TH_LOW)
                {
                    // 最优匹配和次优匹配，相差越大表示匹配越好
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        // 地图点1 与 地图点2 匹配上了
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2]=true;

                        // 匹配特征的角度变化直方图，用于剔除误匹配
                        if(mbCheckOrientation)
                        {
                            // 匹配特征 的角度变化（关键帧->当前帧）
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    // 根据角度变化直方图，剔除误匹配
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        // 计算直方图中，频数最高的几个组（一般会提取出头尾两个分组 -30->30度）
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

/// 通过三角化得到的基础矩阵F，进行特征匹配，并判断是否满足对极约束
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
{   
    // 使用字典加快匹配
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter(); // 相机1的光心位置
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w*Cw+t2w;

    cv::Point2f ep = pKF2->mpCamera->project(C2); // 相机2的极点位置

    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    cv::Mat R12;
    cv::Mat t12;

    cv::Mat Rll,Rlr,Rrl,Rrr;
    cv::Mat tll,tlr,trl,trr;

    GeometricCamera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;

    if(!pKF1->mpCamera2 && !pKF2->mpCamera2){
        R12 = R1w*R2w.t();
        t12 = -R1w*R2w.t()*t2w+t1w;
    }
    else{
        Rll = pKF1->GetRotation() * pKF2->GetRotation().t();
        Rlr = pKF1->GetRotation() * pKF2->GetRightRotation().t();
        Rrl = pKF1->GetRightRotation() * pKF2->GetRotation().t();
        Rrr = pKF1->GetRightRotation() * pKF2->GetRightRotation().t();

        tll = pKF1->GetRotation() * (-pKF2->GetRotation().t() * pKF2->GetTranslation()) + pKF1->GetTranslation();
        tlr = pKF1->GetRotation() * (-pKF2->GetRightRotation().t() * pKF2->GetRightTranslation()) + pKF1->GetTranslation();
        trl = pKF1->GetRightRotation() * (-pKF2->GetRotation().t() * pKF2->GetTranslation()) + pKF1->GetRightTranslation();
        trr = pKF1->GetRightRotation() * (-pKF2->GetRightRotation().t() * pKF2->GetRightTranslation()) + pKF1->GetRightTranslation();
    }

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);
    vector<int> vMatches12(pKF1->N,-1);

    // 建立角度变化直方图，用于后面剔除误匹配
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it!=f1end && f2it!=f2end)
    {
        // 正向索引节点相同，即特征都在同一颗子树下
        if(f1it->first == f2it->first)
        {
            /// 遍历当前子树下，关键帧1的特征ID列表
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1]; // 关键帧1的特征ID
                
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1); // 对应的地图点
                
                // If there is already a MapPoint skip
                if(pMP1)
                {
                    continue;
                }

                const bool bStereo1 = (!pKF1->mpCamera2 && pKF1->mvuRight[idx1]>=0);

                if(bOnlyStereo)
                    if(!bStereo1)
                        continue;

                // 对应的特征点
                const cv::KeyPoint &kp1 = (pKF1 -> NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                : (idx1 < pKF1 -> NLeft) ? pKF1 -> mvKeys[idx1]
                                                                                         : pKF1 -> mvKeysRight[idx1 - pKF1 -> NLeft];

                const bool bRight1 = (pKF1 -> NLeft == -1 || idx1 < pKF1 -> NLeft) ? false
                                                                                   : true;
                //if(bRight1) continue;
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1); // 对应的描述子
                
                int bestDist = TH_LOW; // 最优距离
                int bestIdx2 = -1;
                
                /// 遍历当前子树下，关键帧2的特征ID列表
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2]; // 关键帧2的特征ID
                    
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2); // 对应的地图点
                    
                    // If we have already matched or there is a MapPoint skip
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = (!pKF2->mpCamera2 &&  pKF2->mvuRight[idx2]>=0);

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2); // 对应的描述子
                    
                    /// 计算两个描述子之间的距离
                    const int dist = DescriptorDistance(d1,d2);
                    
                    if(dist>TH_LOW || dist>bestDist)
                        continue;

                    // 对应的特征点
                    const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                    : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                             : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
                    const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                                       : true;
                    /// 投影在第二张图上的位置，距离极点的距离
                    if(!bStereo1 && !bStereo2 && !pKF1->mpCamera2)
                    {
                        const float distex = ep.x-kp2.pt.x;
                        const float distey = ep.y-kp2.pt.y;
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                        {
                            continue; // 距离相机2的极点过近
                        }
                    }

                    if(pKF1->mpCamera2 && pKF2->mpCamera2){
                        if(bRight1 && bRight2){
                            R12 = Rrr;
                            t12 = trr;

                            pCamera1 = pKF1->mpCamera2;
                            pCamera2 = pKF2->mpCamera2;
                        }
                        else if(bRight1 && !bRight2){
                            R12 = Rrl;
                            t12 = trl;

                            pCamera1 = pKF1->mpCamera2;
                            pCamera2 = pKF2->mpCamera;
                        }
                        else if(!bRight1 && bRight2){
                            R12 = Rlr;
                            t12 = tlr;

                            pCamera1 = pKF1->mpCamera;
                            pCamera2 = pKF2->mpCamera2;
                        }
                        else{
                            R12 = Rll;
                            t12 = tll;

                            pCamera1 = pKF1->mpCamera;
                            pCamera2 = pKF2->mpCamera;
                        }

                    }
                    
                    /// 判断是否满足对极约束
                    if(pCamera1->epipolarConstrain(pCamera2,kp1,kp2,R12,t12,pKF1->mvLevelSigma2[kp1.octave],pKF2->mvLevelSigma2[kp2.octave])||bCoarse) // MODIFICATION_2
                    {
                        bestIdx2 = idx2; // 最优匹配
                        bestDist = dist;
                    }
                }
                
                /// 在关键帧2下，找到了匹配程度最高的特征点
                if(bestIdx2>=0)
                {
                    const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                    : (bestIdx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[bestIdx2]
                                                                                                 : pKF2 -> mvKeysRight[bestIdx2 - pKF2 -> NLeft];
                    vMatches12[idx1]=bestIdx2;
                    nmatches++;

                    // 匹配特征的角度变化直方图，用于剔除误匹配
                    if(mbCheckOrientation)
                    {
                        // 匹配特征 的角度变化
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    // 根据角度变化直方图，剔除误匹配
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        // 计算直方图中，频数最高的几个组（一般会提取出头尾两个分组 -30->30度）
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    /// 返回匹配结果
    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i])); // 分别在前后两帧上的索引
    }

    return nmatches;
}

    int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                           vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, vector<cv::Mat> &vMatchedPoints)
    {
        // 使用字典加快匹配
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

        //Compute epipole in second image
        cv::Mat Cw = pKF1->GetCameraCenter(); // 相机1的光心位置
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();
        cv::Mat C2 = R2w*Cw+t2w;

        cv::Point2f ep = pKF2->mpCamera->project(C2); // 相机2的极点位置

        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();

        GeometricCamera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;
        cv::Mat Tcw1,Tcw2;

        // Find matches between not tracked keypoints
        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node

        int nmatches=0;
        vector<bool> vbMatched2(pKF2->N,false);
        vector<int> vMatches12(pKF1->N,-1);

        vector<cv::Mat> vMatchesPoints12(pKF1 -> N);

        // 建立角度变化直方图，用于后面剔除误匹配
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f/HISTO_LENGTH;

        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
        int right = 0;
        while(f1it!=f1end && f2it!=f2end)
        {
            // 正向索引节点相同，即特征都在同一颗子树下
            if(f1it->first == f2it->first)
            {
                /// 遍历当前子树下，关键帧1的特征ID列表
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1]; // 关键帧1的特征ID

                    MapPoint* pMP1 = pKF1->GetMapPoint(idx1); // 对应的地图点

                    // If there is already a MapPoint skip
                    if(pMP1)
                        continue;

                    // 对应的特征点
                    const cv::KeyPoint &kp1 = (pKF1 -> NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                    : (idx1 < pKF1 -> NLeft) ? pKF1 -> mvKeys[idx1]
                                                                                             : pKF1 -> mvKeysRight[idx1 - pKF1 -> NLeft];

                    const bool bRight1 = (pKF1 -> NLeft == -1 || idx1 < pKF1 -> NLeft) ? false
                                                                                       : true;


                    const cv::Mat &d1 = pKF1->mDescriptors.row(idx1); // 对应的描述子

                    int bestDist = TH_LOW; // 最优距离
                    int bestIdx2 = -1;

                    cv::Mat bestPoint;

                    /// 遍历当前子树下，关键帧2的特征ID列表
                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        size_t idx2 = f2it->second[i2]; // 关键帧2的特征ID

                        MapPoint* pMP2 = pKF2->GetMapPoint(idx2); // 对应的地图点

                        // If we have already matched or there is a MapPoint skip
                        if(vbMatched2[idx2] || pMP2)
                            continue;

                        const cv::Mat &d2 = pKF2->mDescriptors.row(idx2); // 对应的描述子

                        /// 计算两个描述子之间的距离
                        const int dist = DescriptorDistance(d1,d2);

                        if(dist>TH_LOW || dist>bestDist){
                            continue;
                        }

                        // 对应的特征点
                        const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                        : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                                 : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
                        const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                                           : true;

                        if(bRight1){
                            Tcw1 = pKF1->GetRightPose();
                            pCamera1 = pKF1->mpCamera2;
                        } else{
                            Tcw1 = pKF1->GetPose();
                            pCamera1 = pKF1->mpCamera;
                        }

                        if(bRight2){
                            Tcw2 = pKF2->GetRightPose();
                            pCamera2 = pKF2->mpCamera2;
                        } else{
                            Tcw2 = pKF2->GetPose();
                            pCamera2 = pKF2->mpCamera;
                        }

                        cv::Mat x3D;
                        if(pCamera1->matchAndtriangulate(kp1,kp2,pCamera2,Tcw1,Tcw2,pKF1->mvLevelSigma2[kp1.octave],pKF2->mvLevelSigma2[kp2.octave],x3D)){
                            bestIdx2 = idx2;
                            bestDist = dist;
                            bestPoint = x3D;
                        }

                    }

                    if(bestIdx2>=0)
                    {
                        const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                        : (bestIdx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[bestIdx2]
                                                                                                     : pKF2 -> mvKeysRight[bestIdx2 - pKF2 -> NLeft];
                        vMatches12[idx1]=bestIdx2;
                        vMatchesPoints12[idx1] = bestPoint;
                        nmatches++;
                        if(bRight1) right++;

                        if(mbCheckOrientation)
                        {
                            float rot = kp1.angle-kp2.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vMatches12[rotHist[i][j]]=-1;
                    nmatches--;
                }
            }

        }

        vMatchedPairs.clear();
        vMatchedPairs.reserve(nmatches);

        for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
        {
            if(vMatches12[i]<0)
                continue;
            vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
            vMatchedPoints.push_back(vMatchesPoints12[i]);
        }
        return nmatches;
    }

/// 1.地图点 与 关键帧 进行特征匹配 并 地图点融合
int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th, const bool bRight)
{
    cv::Mat Rcw,tcw, Ow;
    GeometricCamera* pCamera;

    if(bRight){
        Rcw = pKF->GetRightRotation();
        tcw = pKF->GetRightTranslation();
        Ow = pKF->GetRightCameraCenter();

        pCamera = pKF->mpCamera2;
    }
    else{
        Rcw = pKF->GetRotation();
        tcw = pKF->GetTranslation();
        Ow = pKF->GetCameraCenter(); // 光心位置

        pCamera = pKF->mpCamera;
    }

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    int nFused=0;

    const int nMPs = vpMapPoints.size();

    // For debbuging
    int count_notMP = 0, count_bad=0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0, count_dist = 0, count_normal=0, count_notidx = 0, count_thcheck = 0;
    
    /// 遍历所有的地图点
    for(int i=0; i<nMPs; i++)
    {
        MapPoint* pMP = vpMapPoints[i];

        if(!pMP)
        {
            count_notMP++;
            continue;
        }

        /*if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;*/
        if(pMP->isBad())
        {
            count_bad++;
            continue;
        }
        else if(pMP->IsInKeyFrame(pKF))
        {
            count_isinKF++;
            continue;
        }


        cv::Mat p3Dw = pMP->GetWorldPos(); // 3d世界坐标
        cv::Mat p3Dc = Rcw*p3Dw + tcw;     // 3d相机坐标


        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
        {
            count_negdepth++;
            continue;
        }

        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0);
        const float y = p3Dc.at<float>(1);
        const float z = p3Dc.at<float>(2);

        // 2d像素坐标
        const cv::Point2f uv = pCamera->project(cv::Point3f(x,y,z));

        // Point must be inside the image
        if(!pKF->IsInImage(uv.x,uv.y))
        {
            count_notinim++;
            continue; 
        }

        const float ur = uv.x-bf*invz;

        /// 判断 地图点的深度 是否在可探测范围
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
        {
            count_dist++;
            continue; // 不在可探测范围v
        }

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
        {
            count_normal++;
            continue;
        }

        /// 在关键帧上，寻找 有可能是该地图点 的所有特征点

        int nPredictedLevel = pMP->PredictScale(dist3D,pKF); // 预估地图点在关键帧上的层数

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel]; // 层数越高，半径越大

        // 关键帧在一定范围内的特征点
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x,uv.y,radius,bRight);

        if(vIndices.empty())
        {
            count_notidx++;
            continue;
        }

        // Match to the most similar keypoint in the radius
        /// 地图点 与 特征点 进行匹配

        const cv::Mat dMP = pMP->GetDescriptor(); // 地图点的描述子

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            size_t idx = *vit;
            
            // 特征点
            const cv::KeyPoint &kp = (pKF -> NLeft == -1) ? pKF->mvKeysUn[idx]
                                                          : (!bRight) ? pKF -> mvKeys[idx]
                                                                      : pKF -> mvKeysRight[idx];

            const int &kpLevel= kp.octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel) // 预估的层数
                continue;

            if(pKF->mvuRight[idx]>=0) // 双目 或 RGBD
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = uv.x-kpx;
                const float ey = uv.y-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            else // 单目
            {
                // 投影误差
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = uv.x-kpx;
                const float ey = uv.y-kpy;
                const float e2 = ex*ex+ey*ey;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }

            if(bRight) idx += pKF->NLeft;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx); // 特征点的描述子

            const int dist = DescriptorDistance(dMP,dKF); // 地图点 与 特征点 的距离

            if(dist<bestDist)
            {
                bestDist = dist; // 最优距离
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        /// 地图点 与 匹配上的特征点 融合
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF) // 特征点 已经存在地图点，两个地图点融合
            {
                if(!pMPinKF->isBad())
                {
                    // 选择被观测数多的地图点
                    if(pMPinKF->Observations()>pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else // 特征点 不存在地图点，直接新添
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
        else
            count_thcheck++;

    }

    /*cout << "count_notMP = " << count_notMP << endl;
    cout << "count_bad = " << count_bad << endl;
    cout << "count_isinKF = " << count_isinKF << endl;
    cout << "count_negdepth = " << count_negdepth << endl;
    cout << "count_notinim = " << count_notinim << endl;
    cout << "count_dist = " << count_dist << endl;
    cout << "count_normal = " << count_normal << endl;
    cout << "count_notidx = " << count_notidx << endl;
    cout << "count_thcheck = " << count_thcheck << endl;
    cout << "tot fused points: " << nFused << endl;*/
    return nFused; // 返回融合的数量
}

/// 2.地图点 与 关键帧 通过 sim3投影匹配，并融合地图点
int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image
        const float x = p3Dc.at<float>(0);
        const float y = p3Dc.at<float>(1);
        const float z = p3Dc.at<float>(2);

        const cv::Point2f uv = pKF->mpCamera->project(cv::Point3f(x,y,z));

        // Point must be inside the image
        if(!pKF->IsInImage(uv.x,uv.y))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x,uv.y,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF) // 匹配上的特征点，已经存在地图点
            {
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF; // 直接替换
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

/// 闭环检测中，通过sim3变换，进行特征匹配寻找更多的特征点（地图点之间的匹配）
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size(); // 关键帧1 的地图点

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size(); // 关键帧2 的地图点

    /// 记录之前通过词袋匹配上的特征点
    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;        // 关键帧1
            int idx2 = get<0>(pMP->GetIndexInKeyFrame(pKF2));
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true; // 关键帧2
        }
    }

    /// 关键帧1 与 关键帧2 分别进行特征匹配，最后再检测 两次匹配的一致性
    vector<int> vnMatch1(N1,-1); // 关键帧1 的匹配结果
    vector<int> vnMatch2(N2,-1); // 关键帧2 的匹配结果


    // Transform from KF1 to KF2 and search
    /// 遍历 关键帧1 的地图点，在 关键帧2 上寻找匹配的地图点
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1]; // 地图点1

        if(!pMP || vbAlreadyMatched1[i1]) // 不是地图点，或者已经被匹配上了
            continue;

        if(pMP->isBad())
            continue;

        // 地图点1 的坐标
        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        // 投影到 关键帧2 的2d像素坐标
        const float u = fx*x+cx; 
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        // 判断 地图点1 是否在相机2 的可观测范围
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        // 预估 地图点1 在关键帧2上的层数，并寻找可能的匹配点
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor(); // 描述子1

        // 遍历 关键帧2 的地图点，寻找最优匹配
        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx]; // 关键点2

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx); // 描述子2

            // 两个描述子之间的距离
            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH) // 最优距离超过100 表示为误匹配
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    /// 遍历 关键帧2 的地图点，在 关键帧1 上寻找匹配的地图点
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

    /// 2.将上一帧的地图点投影到当前帧，用于跟踪上一帧（当前帧，上一帧，搜索范围，是否为单目）
    int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
    {
        int nmatches = 0; // 匹配到的特征数量

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        // 坐标变换关系
        const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
        const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

        const cv::Mat twc = -Rcw.t()*tcw;

        const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
        const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);

        const cv::Mat tlc = Rlw*twc+tlw; // 当前帧，相对于上一帧的平移

        // 平移的方向（向前移动还是向后移动）
        const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;
        const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;

        // 遍历上一帧的地图点
        for(int i=0; i<LastFrame.N; i++)
        {
            MapPoint* pMP = LastFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!LastFrame.mvbOutlier[i])
                {
                    /// 当前地图点
                    
                    // Project
                    // 将地图点，投影到当前帧
                    cv::Mat x3Dw = pMP->GetWorldPos(); // 3D世界坐标
                    cv::Mat x3Dc = Rcw*x3Dw+tcw;       // 3D相机坐标（当前帧）

                    const float xc = x3Dc.at<float>(0);
                    const float yc = x3Dc.at<float>(1);
                    const float invzc = 1.0/x3Dc.at<float>(2);

                    if(invzc<0)
                        continue;

                    cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dc); // 2D像素坐标（当前帧）

                    if(uv.x<CurrentFrame.mnMinX || uv.x>CurrentFrame.mnMaxX)
                        continue;
                    if(uv.y<CurrentFrame.mnMinY || uv.y>CurrentFrame.mnMaxY)
                        continue;

                    // 根据地图点，在上一帧的金字塔层数，在当前帧投影位置附近，寻找特征点
                    int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                     : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                    // Search in a window. Size depends on scale
                    float radius = th*CurrentFrame.mvScaleFactors[nLastOctave]; // 层数越高，搜索范围越大

                    vector<size_t> vIndices2;

                    if(bForward)       // 向前移动，关键点变清楚了，能找到关键点的层数变高了
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave);
                    else if(bBackward) // 向后移动，关键点变模糊了，能找到关键点的层数变低了
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, 0, nLastOctave);
                    else               // 没有明显的前后移动，则在附近三层金字塔内寻找
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave-1, nLastOctave+1);

                    if(vIndices2.empty())
                        continue;

                    // 寻找和地图点最优的匹配
                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                    {
                        const size_t i2 = *vit; // 当前特征点ID

                        // 已经匹配到地图点了，在右图的位置不在搜索范围内
                        if(CurrentFrame.mvpMapPoints[i2])
                            if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                                continue;

                        if(CurrentFrame.Nleft == -1 && CurrentFrame.mvuRight[i2]>0)
                        {
                            const float ur = uv.x - CurrentFrame.mbf*invzc;
                            const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                            if(er>radius)
                                continue;
                        }

                        // 当前特征点与地图点的距离
                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                        const int dist = DescriptorDistance(dMP,d);

                        if(dist<bestDist)
                        {
                            bestDist=dist; // 最优距离
                            bestIdx2=i2;
                        }
                    }

                    // 剔除误匹配（最优距离的阈值，特征点的角度变化应该一致）
                    if(bestDist<=TH_HIGH)
                    {
                        CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                        nmatches++;

                        // 匹配特征的角度变化直方图，用于剔除误匹配
                        if(mbCheckOrientation)
                        {
                            cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                        : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                                : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                            cv::KeyPoint kpCF = (CurrentFrame.Nleft == -1) ? CurrentFrame.mvKeysUn[bestIdx2]
                                                                           : (bestIdx2 < CurrentFrame.Nleft) ? CurrentFrame.mvKeys[bestIdx2]
                                                                                                             : CurrentFrame.mvKeysRight[bestIdx2 - CurrentFrame.Nleft];
                            // 匹配特征 的角度变化（上一帧->当前帧）
                            float rot = kpLF.angle-kpCF.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
                    }
                    if(CurrentFrame.Nleft != -1){
                        cv::Mat x3Dr = CurrentFrame.mTrl.colRange(0,3).rowRange(0,3) * x3Dc + CurrentFrame.mTrl.col(3);

                        cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dr);

                        int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                         : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                        // Search in a window. Size depends on scale
                        float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                        vector<size_t> vIndices2;

                        if(bForward)
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave, -1,true);
                        else if(bBackward)
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, 0, nLastOctave, true);
                        else
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave-1, nLastOctave+1, true);

                        const cv::Mat dMP = pMP->GetDescriptor();

                        int bestDist = 256;
                        int bestIdx2 = -1;

                        for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                        {
                            const size_t i2 = *vit;
                            if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft])
                                if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft]->Observations()>0)
                                    continue;

                            const cv::Mat &d = CurrentFrame.mDescriptors.row(i2 + CurrentFrame.Nleft);

                            const int dist = DescriptorDistance(dMP,d);

                            if(dist<bestDist)
                            {
                                bestDist=dist;
                                bestIdx2=i2;
                            }
                        }

                        if(bestDist<=TH_HIGH)
                        {
                            CurrentFrame.mvpMapPoints[bestIdx2 + CurrentFrame.Nleft]=pMP;
                            nmatches++;
                            if(mbCheckOrientation)
                            {
                                cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                            : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                                    : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                                cv::KeyPoint kpCF = CurrentFrame.mvKeysRight[bestIdx2];

                                float rot = kpLF.angle-kpCF.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdx2  + CurrentFrame.Nleft);
                            }
                        }

                    }
                }
            }
        }

        //Apply rotation consistency
        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            // 计算直方图中，频数最高的几个组（一般会提取出头尾两个分组 -30->30度）
            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                // 当特征的角度变化没有在这个最高范围，则认为是误匹配，剔除
                if(i!=ind1 && i!=ind2 && i!=ind3)
                {
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                        nmatches--;
                    }
                }
            }
        }

        return nmatches;
    }

/// 3.对当前帧和候选帧，通过投影，进行特征匹配，用于重定位
/// 参数：当前帧 关键帧 已经匹配到的地图点 搜索范围 误匹配阈值
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
{
    int nmatches = 0; // 匹配到的特征数量

    // 坐标变换关系
    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw; // twc，相机光心在世界的位置

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    // 遍历候选帧的地图点
    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            // 如果当前地图点，已经被匹配过了，则跳过
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
                // 将地图点，投影到当前帧
                cv::Mat x3Dw = pMP->GetWorldPos(); // 3D世界坐标
                cv::Mat x3Dc = Rcw*x3Dw+tcw;       // 3D相机坐标（当前帧）

                const cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dc); // 2D像素坐标（当前帧）

                if(uv.x<CurrentFrame.mnMinX || uv.x>CurrentFrame.mnMaxX)
                    continue;
                if(uv.y<CurrentFrame.mnMinY || uv.y>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                // 计算地图点到光心的距离，并判断是否在尺度不变范围
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO);

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                // 根据地图点的距离，预测在金字塔的层数
                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                // Search in a window
                // 根据预测的金字塔层数，在地图点的投影位置附近，寻找特征点
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x, uv.y, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                // 寻找和地图点最优的匹配
                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit; // 当前特征点ID
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue; // 已经匹配到地图点了

                    // 当前特征点与地图点的距离
                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist; // 最优距离
                        bestIdx2=i2;
                    }
                }

                // 剔除误匹配（最优距离的阈值，特征点的角度变化应该一致）
                if(bestDist<=ORBdist)
                {
                    // 当前帧匹配到的地图点
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    // 匹配特征的角度变化直方图，用于剔除误匹配
                    if(mbCheckOrientation)
                    {
                        // 匹配特征 的角度变化
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        // 计算直方图中，频数最高的几个组（一般会提取出头尾两个分组 -30->30度）
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                    nmatches--;
                }
            }
        }
    }

    // 返回最终匹配到的特征点数
    return nmatches;
}

/// 计算直方图中，频数最高的三个组
void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
/// 计算两个ORB描述子之间的汉明距离
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    // 8*32=256
    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb; // 异或（32位）
        
        // 计算v中1的位数（平行算法）
        v = v - ((v >> 1) & 0x55555555);                            // 01--------->16
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);           // 0011------->8
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24; // 00001111--->4     00000001--->移动到25-32位    再向右移动24位即得结果
    }

    return dist;
}

} //namespace ORB_SLAM
