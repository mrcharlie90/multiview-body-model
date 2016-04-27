//
// Created by Mauro on 15/04/16.
//

#ifndef MULTIVIEWBODYMODEL_MULTIVIEWBODYMODEL_H
#define MULTIVIEWBODYMODEL_MULTIVIEWBODYMODEL_H

#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/nonfree.hpp>

namespace  multiviewbodymodel
{
    using namespace cv;
    /*
     *  Data structure containing information about skeletal tracker keypoints descriptors.
     *  A distance function is developed to comoute the distance between pairs of views
     *  of the same angle. The views must be inserted with the same order in every body model.
     */
    class MultiviewBodyModel {
    public:
        MultiviewBodyModel();
        MultiviewBodyModel(vector<int> views_id, vector<int> pose_side, vector<cv::Mat> views_descriptors,
                           vector<vector<float> > views_descriptors_confidences);

        void ConfidenceNormalization();
        vector<float> Distances(MultiviewBodyModel body_model);
        float Distance(MultiviewBodyModel body_model, int view_id);
        void ReadAndCompute(string file_path, string img_path, int view_id, string descriptor_extractor_type, float keypoint_size);

        void set_views_id(vector<int> views_id);
        vector<int> views_id();
        void set_views_descriptors(vector<cv::Mat> views_descriptors);
        vector<cv::Mat> views_descriptors();
        void set_views_descriptors_confidences(vector<vector<float> > views_descriptors_confidences);
        vector<vector<float> > views_descriptors_confidences();
    private:

        // View identifier (i.e. 0:left, 1:right, 2:center, ... )
        vector<int> views_id_;

        // Pose number (i.e. 1:front, 2:back, 3:left-side, 4:right-side )
        vector<int> pose_side_;

        // Contains descriptors relative to keypoint selected in each view
        vector<cv::Mat> views_descriptors_;

        // Confidence value for each keypoint of each view. A value between 0 and 1
        // (temporary 1: keypoint visible, 0: keypoint is occluded))
        vector<vector<float> > views_descriptors_confidences_;
    };
}
#endif //MULTIVIEWBODYMODEL_MULTIVIEWBODYMODEL_H
