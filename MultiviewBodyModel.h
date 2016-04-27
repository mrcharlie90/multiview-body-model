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

using namespace std;

namespace  multiviewbodymodel
{

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
        void ChangeViewDescriptors(int view_id, cv::Mat descriptors, vector<float> descriptors_confidences);
        void ReadAndCompute(string file_path, string img_path, int view_id, string descriptor_extractor_type, float keypoint_size);

        void set_views_id(vector<int> views_id);
        vector<int> views_id();
        void set_views_descriptors(vector<cv::Mat> views_descriptors);
        vector<cv::Mat> views_descriptors();
        void set_views_descriptors_confidences(vector<vector<float> > views_descriptors_confidences);
        vector<vector<float> > views_descriptors_confidences();

        void set_pose_side(int view_id, int pose_side);
        int pose_side(int view_id);
        cv::Mat views_descriptors(int view_id);
        vector<float> confidences(int view_id);

    private:

        // A unique number identifying the view: left, right, center, ...
        vector<int> views_id_;

        // Pose number 1: front 2: back 3: left-side 4: right-side
        // (one for each view)
        vector<int> pose_side_;

        // Contains descriptors relative to keypoint selected in each view
        vector<cv::Mat> views_descriptors_;

        // contains the confidence of each keypoint selected by the skeletal tracker
        // for each view (for now 1: keypoint visible, 0: keypoint is occluded)
        vector<vector<float> > views_descriptors_confidences_;
    };
}
#endif //MULTIVIEWBODYMODEL_MULTIVIEWBODYMODEL_H
