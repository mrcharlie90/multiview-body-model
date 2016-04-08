//
// Created by Mauro on 07/04/16.
//
// OpenCV ver. 2.4
//

#include <iostream>
#include <vector>
#include <opencv/cv.h>
#include <opencv2/nonfree/nonfree.hpp>

#ifndef HELLOWORLD_MULTIVIEWBODYMODEL_H
#define HELLOWORLD_MULTIVIEWBODYMODEL_H

/*
 * Struct for storing a confidence for each
 * keypoint.
 */

struct ConfidenceKeypoint
{
    float confidence;
    cv::KeyPoint keypoint;
};

class MultiviewBodyModel
{
private:
    // Required Storage
    std::vector<std::vector<ConfidenceKeypoint> > vec_keypoints;
    std::vector<float> angles;

public:
    MultiviewBodyModel(std::vector<std::vector<ConfidenceKeypoint> >  vec_keypoints, std::vector<float> angles);

    void addKeypointsVector();

    double distance(); // TODO: use  initModules_nonfree(); when using SIFT



};


#endif //HELLOWORLD_MULTIVIEWBODYMODEL_H
