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
        MultiviewBodyModel(int max_poses);
        void ConfidenceNormalization();
        vector<float> Distances(MultiviewBodyModel body_model);
//        float Distance(MultiviewBodyModel body_model, int view_id);
        bool ReadAndCompute(string file_path, string img_path, string descriptor_extractor_type, int keypoint_size);
        float match(Mat query_descritptors, int pose_side, string descriptor_matcher_type);
        bool ready();

        vector<Mat> views_descriptors();
        vector<vector<float> > views_descriptors_confidences();

    private:
        // The model will reach the ready state when the the total
        // number of poses acquired is equal to max_poses.
        int max_poses_;

        // Pose number (i.e. 1:front, 2:back, 3:left-side, 4:right-side )
        vector<int> pose_side_;

        // Contains descriptors relative to keypoint selected in each view
        vector<cv::Mat> views_descriptors_;

        // Vector containing all keypoints for each image
        vector<vector<cv::KeyPoint> > views_keypoints_;

        vector<Mat> views_images_;

        // Confidence value for each keypoint of each view. A value between 0 and 1
        // (temporary 1: keypoint visible, 0: keypoint is occluded))
        vector<vector<float> > views_descriptors_confidences_;

        string descriptor_extractor_type_;

        int keypoint_size_;
    };
}
#endif //MULTIVIEWBODYMODEL_MULTIVIEWBODYMODEL_H
