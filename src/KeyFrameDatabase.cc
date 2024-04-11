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


#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM3
{

KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}

/// 添加关键帧
void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);

    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF); // 单词的关键帧列表
}

/// 删除关键帧
void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    // 遍历关键帧的单词
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        // 当前单词对应的关键帧列表
        list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

        // 将关键帧，从当前关键帧列表中删除
        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

/// 清空关键帧数据库
void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

void KeyFrameDatabase::clearMap(Map* pMap)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for(std::vector<list<KeyFrame*> >::iterator vit=mvInvertedFile.begin(), vend=mvInvertedFile.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs =  *vit;

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend;)
        {
            KeyFrame* pKFi = *lit;
            if(pMap == pKFi->GetMap())
            {
                lit = lKFs.erase(lit);
                // Dont delete the KF because the class Map clean all the KF when it is destroyed
            }
            else
            {
                ++lit;
            }
        }
    }
}

/// 寻找闭环检测的 候选关键帧 列表
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    /// 寻找和当前帧 存在相同单词，但是不共视的 关键帧
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames(); // 共视帧
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        unique_lock<mutex> lock(mMutex);

        // 遍历当前帧的所有单词
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            // 存在当前单词的所有关键帧
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            // 关键帧还未与当前帧关联的话，进行关联
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->GetMap()==pKF->GetMap()) // For consider a loop candidate it a candidate it must be in the same map
                {
                    if(pKFi->mnLoopQuery!=pKF->mnId)
                    {
                        pKFi->mnLoopWords=0;
                        if(!spConnectedKeyFrames.count(pKFi)) // 不能是共视帧
                        {
                            pKFi->mnLoopQuery=pKF->mnId;
                            lKFsSharingWords.push_back(pKFi);
                        }
                    }
                    pKFi->mnLoopWords++; // 关键帧与当前帧关联的单词数
                }


            }
        }
    }

    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    // 将最多的关联单词数*0.8,作为最低阈值
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords; // 最多的单词数
    }

    int minCommonWords = maxCommonWords*0.8f;

    int nscores=0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    /// 对超过阈值的关联帧，进行相似性评分（打分越高，越相似）
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnLoopWords>minCommonWords)
        {
            nscores++;

            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

            pKFi->mLoopScore = si; // 相似性得分
            if(si>=minScore) // 相似性得分 大于 最低得分阈值
                lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    /// 将这些关联帧分组（共视），计算每个组的总分及其最高分
    list<pair<float,KeyFrame*> > lAccScoreAndMatch; // <当前组的总分，分数最高的关键帧>
    float bestAccScore = minScore; // 最高的组分

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        // 与当前关联帧，有共视的10个关键帧，看成一个组
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        // 统计当前组的（相似性总分，分数最高的关键帧）
        float bestScore = it->first; // 当前组的最高分
        float accScore = it->first;  // 当前组的总分
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            // 组内的关键帧不一定都与当前帧有关联
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
                accScore+=pKF2->mLoopScore; // 只统计有关联的，并且关联单词数足够多
                if(pKF2->mLoopScore>bestScore)
                {
                    pBestKF=pKF2;
                    bestScore = pKF2->mLoopScore; // 组内分数最高的关键帧
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    /// 将最高总分*0.75，作为阈值，筛选出总分较高的组，将组内分数最高的关键帧作为候选帧返回
    float minScoreToRetain = 0.75f*bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;    // 已经是候选帧的关键帧列表
    vector<KeyFrame*> vpLoopCandidates; // 候选帧列表
    vpLoopCandidates.reserve(lAccScoreAndMatch.size()); 

    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        // 总分超过阈值的组
        if(it->first>minScoreToRetain)
        {
            // 组内分数最高的关键帧，作为候选帧
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }


    return vpLoopCandidates;
}

void KeyFrameDatabase::DetectCandidates(KeyFrame* pKF, float minScore,vector<KeyFrame*>& vpLoopCand, vector<KeyFrame*>& vpMergeCand)
{
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    list<KeyFrame*> lKFsSharingWordsLoop,lKFsSharingWordsMerge;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        unique_lock<mutex> lock(mMutex);

        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->GetMap()==pKF->GetMap()) // For consider a loop candidate it a candidate it must be in the same map
                {
                    if(pKFi->mnLoopQuery!=pKF->mnId)
                    {
                        pKFi->mnLoopWords=0;
                        if(!spConnectedKeyFrames.count(pKFi))
                        {
                            pKFi->mnLoopQuery=pKF->mnId;
                            lKFsSharingWordsLoop.push_back(pKFi);
                        }
                    }
                    pKFi->mnLoopWords++;
                }
                else if(!pKFi->GetMap()->IsBad())
                {
                    if(pKFi->mnMergeQuery!=pKF->mnId)
                    {
                        pKFi->mnMergeWords=0;
                        if(!spConnectedKeyFrames.count(pKFi))
                        {
                            pKFi->mnMergeQuery=pKF->mnId;
                            lKFsSharingWordsMerge.push_back(pKFi);
                        }
                    }
                    pKFi->mnMergeWords++;
                }
            }
        }
    }

    if(lKFsSharingWordsLoop.empty() && lKFsSharingWordsMerge.empty())
        return;

    if(!lKFsSharingWordsLoop.empty())
    {
        list<pair<float,KeyFrame*> > lScoreAndMatch;

        // Only compare against those keyframes that share enough words
        int maxCommonWords=0;
        for(list<KeyFrame*>::iterator lit=lKFsSharingWordsLoop.begin(), lend= lKFsSharingWordsLoop.end(); lit!=lend; lit++)
        {
            if((*lit)->mnLoopWords>maxCommonWords)
                maxCommonWords=(*lit)->mnLoopWords;
        }

        int minCommonWords = maxCommonWords*0.8f;

        int nscores=0;

        // Compute similarity score. Retain the matches whose score is higher than minScore
        for(list<KeyFrame*>::iterator lit=lKFsSharingWordsLoop.begin(), lend= lKFsSharingWordsLoop.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;

            if(pKFi->mnLoopWords>minCommonWords)
            {
                nscores++;

                float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

                pKFi->mLoopScore = si;
                if(si>=minScore)
                    lScoreAndMatch.push_back(make_pair(si,pKFi));
            }
        }

        if(!lScoreAndMatch.empty())
        {
            list<pair<float,KeyFrame*> > lAccScoreAndMatch;
            float bestAccScore = minScore;

            // Lets now accumulate score by covisibility
            for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
            {
                KeyFrame* pKFi = it->second;
                vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

                float bestScore = it->first;
                float accScore = it->first;
                KeyFrame* pBestKF = pKFi;
                for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
                {
                    KeyFrame* pKF2 = *vit;
                    if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
                    {
                        accScore+=pKF2->mLoopScore;
                        if(pKF2->mLoopScore>bestScore)
                        {
                            pBestKF=pKF2;
                            bestScore = pKF2->mLoopScore;
                        }
                    }
                }

                lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
                if(accScore>bestAccScore)
                    bestAccScore=accScore;
            }

            // Return all those keyframes with a score higher than 0.75*bestScore
            float minScoreToRetain = 0.75f*bestAccScore;

            set<KeyFrame*> spAlreadyAddedKF;
            vpLoopCand.reserve(lAccScoreAndMatch.size());

            for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
            {
                if(it->first>minScoreToRetain)
                {
                    KeyFrame* pKFi = it->second;
                    if(!spAlreadyAddedKF.count(pKFi))
                    {
                        vpLoopCand.push_back(pKFi);
                        spAlreadyAddedKF.insert(pKFi);
                    }
                }
            }
        }

    }

    if(!lKFsSharingWordsMerge.empty())
    {
        //cout << "BoW candidates: " << lKFsSharingWordsMerge.size() << endl;
        list<pair<float,KeyFrame*> > lScoreAndMatch;

        // Only compare against those keyframes that share enough words
        int maxCommonWords=0;
        for(list<KeyFrame*>::iterator lit=lKFsSharingWordsMerge.begin(), lend=lKFsSharingWordsMerge.end(); lit!=lend; lit++)
        {
            if((*lit)->mnMergeWords>maxCommonWords)
                maxCommonWords=(*lit)->mnMergeWords;
        }
        //cout << "Max common words: " << maxCommonWords << endl;

        int minCommonWords = maxCommonWords*0.8f;

        int nscores=0;

        // Compute similarity score. Retain the matches whose score is higher than minScore
        for(list<KeyFrame*>::iterator lit=lKFsSharingWordsMerge.begin(), lend=lKFsSharingWordsMerge.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;

            if(pKFi->mnMergeWords>minCommonWords)
            {
                nscores++;

                float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);
                //cout << "KF score: " << si << endl;

                pKFi->mMergeScore = si;
                if(si>=minScore)
                    lScoreAndMatch.push_back(make_pair(si,pKFi));
            }
        }
        //cout << "BoW candidates2: " << lScoreAndMatch.size() << endl;

        if(!lScoreAndMatch.empty())
        {
            list<pair<float,KeyFrame*> > lAccScoreAndMatch;
            float bestAccScore = minScore;

            // Lets now accumulate score by covisibility
            for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
            {
                KeyFrame* pKFi = it->second;
                vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

                float bestScore = it->first;
                float accScore = it->first;
                KeyFrame* pBestKF = pKFi;
                for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
                {
                    KeyFrame* pKF2 = *vit;
                    if(pKF2->mnMergeQuery==pKF->mnId && pKF2->mnMergeWords>minCommonWords)
                    {
                        accScore+=pKF2->mMergeScore;
                        if(pKF2->mMergeScore>bestScore)
                        {
                            pBestKF=pKF2;
                            bestScore = pKF2->mMergeScore;
                        }
                    }
                }

                lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
                if(accScore>bestAccScore)
                    bestAccScore=accScore;
            }

            // Return all those keyframes with a score higher than 0.75*bestScore
            float minScoreToRetain = 0.75f*bestAccScore;

            //cout << "Min score to retain: " << minScoreToRetain << endl;

            set<KeyFrame*> spAlreadyAddedKF;
            vpMergeCand.reserve(lAccScoreAndMatch.size());

            for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
            {
                if(it->first>minScoreToRetain)
                {
                    KeyFrame* pKFi = it->second;
                    if(!spAlreadyAddedKF.count(pKFi))
                    {
                        vpMergeCand.push_back(pKFi);
                        spAlreadyAddedKF.insert(pKFi);
                    }
                }
            }
            //cout << "Candidates: " << vpMergeCand.size() << endl;
        }

    }

    //----
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
    {
        list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi=*lit;
            pKFi->mnLoopQuery=-1;
            pKFi->mnMergeQuery=-1;
        }
    }

}

void KeyFrameDatabase::DetectBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, vector<KeyFrame*> &vpMergeCand, int nMinWords)
{
    list<KeyFrame*> lKFsSharingWords;
    set<KeyFrame*> spConnectedKF;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);

        spConnectedKF = pKF->GetConnectedKeyFrames();

        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(spConnectedKF.find(pKFi) != spConnectedKF.end())
                {
                    continue;
                }
                if(pKFi->mnPlaceRecognitionQuery!=pKF->mnId)
                {
                    pKFi->mnPlaceRecognitionWords=0;
                    pKFi->mnPlaceRecognitionQuery=pKF->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
               pKFi->mnPlaceRecognitionWords++;

            }
        }
    }
    if(lKFsSharingWords.empty())
        return;

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnPlaceRecognitionWords>maxCommonWords)
            maxCommonWords=(*lit)->mnPlaceRecognitionWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    if(minCommonWords < nMinWords)
    {
        minCommonWords = nMinWords;
    }

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnPlaceRecognitionWords>minCommonWords)
        {
            nscores++;
            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);
            pKFi->mPlaceRecognitionScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return;

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnPlaceRecognitionQuery!=pKF->mnId)
                continue;

            accScore+=pKF2->mPlaceRecognitionScore;
            if(pKF2->mPlaceRecognitionScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mPlaceRecognitionScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vpLoopCand.reserve(lAccScoreAndMatch.size());
    vpMergeCand.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                if(pKF->GetMap() == pKFi->GetMap())
                {
                    vpLoopCand.push_back(pKFi);
                }
                else
                {
                    vpMergeCand.push_back(pKFi);
                }
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }
}

bool compFirst(const pair<float, KeyFrame*> & a, const pair<float, KeyFrame*> & b)
{
    return a.first > b.first;
}

/// 寻找 闭环检测和Merge检测 的候选关键帧
void KeyFrameDatabase::DetectNBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, vector<KeyFrame*> &vpMergeCand, int nNumCandidates)
{
    /// 寻找和当前帧 存在相同单词，但是不共视的 关键帧

    list<KeyFrame*> lKFsSharingWords;      // 具有相同的单词
    //set<KeyFrame*> spInsertedKFsSharing;
    set<KeyFrame*> spConnectedKF;          // 共视帧

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);

        spConnectedKF = pKF->GetConnectedKeyFrames(); // 共视帧

        // 遍历当前帧的所有单词
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                /*if(spConnectedKF.find(pKFi) != spConnectedKF.end())
                {
                    continue;
                }*/
                if(pKFi->mnPlaceRecognitionQuery!=pKF->mnId)
                {
                    pKFi->mnPlaceRecognitionWords=0;
                    if(!spConnectedKF.count(pKFi))
                    {

                        pKFi->mnPlaceRecognitionQuery=pKF->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnPlaceRecognitionWords++;
                /*if(spInsertedKFsSharing.find(pKFi) == spInsertedKFsSharing.end())
                {
                    lKFsSharingWords.push_back(pKFi);
                    spInsertedKFsSharing.insert(pKFi);
                }*/

            }
        }
    }
    if(lKFsSharingWords.empty())
        return;

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnPlaceRecognitionWords>maxCommonWords)
            maxCommonWords=(*lit)->mnPlaceRecognitionWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnPlaceRecognitionWords>minCommonWords)
        {
            nscores++;
            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);
            pKFi->mPlaceRecognitionScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return;

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnPlaceRecognitionQuery!=pKF->mnId)
                continue;

            accScore+=pKF2->mPlaceRecognitionScore;
            if(pKF2->mPlaceRecognitionScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mPlaceRecognitionScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    //cout << "Amount of candidates: " << lAccScoreAndMatch.size() << endl;

    lAccScoreAndMatch.sort(compFirst);

    vpLoopCand.reserve(nNumCandidates);
    vpMergeCand.reserve(nNumCandidates);
    set<KeyFrame*> spAlreadyAddedKF;
    //cout << "Candidates in score order " << endl;
    //for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    int i = 0;
    list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin();
    while(i < lAccScoreAndMatch.size() && (vpLoopCand.size() < nNumCandidates || vpMergeCand.size() < nNumCandidates))
    {
        //cout << "Accum score: " << it->first << endl;
        KeyFrame* pKFi = it->second;
        if(pKFi->isBad())
            continue;

        if(!spAlreadyAddedKF.count(pKFi))
        {
            if(pKF->GetMap() == pKFi->GetMap() && vpLoopCand.size() < nNumCandidates)
            {
                vpLoopCand.push_back(pKFi);
            }
            else if(pKF->GetMap() != pKFi->GetMap() && vpMergeCand.size() < nNumCandidates && !pKFi->GetMap()->IsBad())
            {
                vpMergeCand.push_back(pKFi);
            }
            spAlreadyAddedKF.insert(pKFi);
        }
        i++;
        it++;
    }
    //-------

    // Return all those keyframes with a score higher than 0.75*bestScore
    /*float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vpLoopCand.reserve(lAccScoreAndMatch.size());
    vpMergeCand.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                if(pKF->GetMap() == pKFi->GetMap())
                {
                    vpLoopCand.push_back(pKFi);
                }
                else
                {
                    vpMergeCand.push_back(pKFi);
                }
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }*/
}

/// 寻找重定位的候选关键帧
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F, Map* pMap)
{
    // 寻找和当前帧存在相同单词的关键帧
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);

        // 遍历当前帧的所有单词
        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
            // 存在当前单词的所有关键帧
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnRelocQuery!=F->mnId) // 关键帧还未与当前帧关联的话，进行关联
                {
                    pKFi->mnRelocWords=0;
                    pKFi->mnRelocQuery=F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++; // 关键帧与当前帧关联的单词数
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    // 将最多的关联单词数*0.8, 作为最低阈值
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    /// 对超过阈值的关联帧，进行相似性评分（打分越高，越相似）
    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnRelocWords>minCommonWords)
        {
            nscores++;
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec); // 词袋向量是经过归一化的，使用L1范数评分
            pKFi->mRelocScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    /// 将这些关联帧分组（共视），计算每个组的总分及其最高分
    list<pair<float,KeyFrame*> > lAccScoreAndMatch; // <当前组的总分，分数最高的关键帧>
    float bestAccScore = 0; // 最高的组分

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        // 与当前关联帧，有共视的10个关键帧，看成一个组
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        // 统计当前组的（相似性总分，分数最高的关键帧）
        float bestScore = it->first; // 当前组的最高分
        float accScore = bestScore;  // 当前组的总分
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            // 组内的关键帧不一定都与当前帧有关联
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnRelocQuery!=F->mnId)
                continue;

            accScore+=pKF2->mRelocScore; // 只统计有关联的
            if(pKF2->mRelocScore>bestScore) // 组内分数最高的关键帧
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore; // 总分最高的组
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    // 将最高总分*0.75，作为阈值，筛选出总分较高的组，将组内分数最高的关键帧作为候选帧返回
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;     // 已经是候选帧的关键帧列表
    vector<KeyFrame*> vpRelocCandidates; // 候选帧列表
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        // 总分超过阈值的组
        const float &si = it->first;
        if(si>minScoreToRetain)
        {
            // 组内分数最高的关键帧，作为候选帧
            KeyFrame* pKFi = it->second;
            if (pKFi->GetMap() != pMap)
                continue;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates; // 返回候选帧列表
}

void KeyFrameDatabase::PreSave()
{
    //Save the information about the inverted index of KF to node
    mvBackupInvertedFileId.resize(mvInvertedFile.size());
    for(int i = 0, numEl = mvInvertedFile.size(); i < numEl; ++i)
    {
        for(std::list<KeyFrame*>::const_iterator it = mvInvertedFile[i].begin(), end = mvInvertedFile[i].end(); it != end; ++it)
        {
            mvBackupInvertedFileId[i].push_back((*it)->mnId);
        }
    }
}

void KeyFrameDatabase::PostLoad(map<long unsigned int, KeyFrame*> mpKFid)
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
    for(unsigned int i = 0; i < mvBackupInvertedFileId.size(); ++i)
    {
        for(long unsigned int KFid : mvBackupInvertedFileId[i])
        {
            if(mpKFid.find(KFid) != mpKFid.end())
            {
                mvInvertedFile[i].push_back(mpKFid[KFid]);
            }
        }
    }

}

void KeyFrameDatabase::SetORBVocabulary(ORBVocabulary* pORBVoc)
{
    ORBVocabulary** ptr;
    ptr = (ORBVocabulary**)( &mpVoc );
    *ptr = pORBVoc;

    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

} //namespace ORB_SLAM
