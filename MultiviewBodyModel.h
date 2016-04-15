//
// Created by Mauro on 15/04/16.
//

#ifndef MULTIVIEWBODYMODEL_MULTIVIEWBODYMODEL_H
#define MULTIVIEWBODYMODEL_MULTIVIEWBODYMODEL_H

#include <iostream>
#include <vector>
#include <numeric>
#include <opencv/cv.h>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;

namespace  multiviewbodymodel
{
    class MultiviewBodyModel {
    public:
        MultiviewBodyModel();
        MultiviewBodyModel(vector<int> views_id, vector<string> views_names, vector<float> views_angles,
                           vector<cv::Mat> views_descriptors,
                           vector<vector<float> > views_descriptors_confidences);

        void ConfidenceNormalization();
        vector<float> Distance(MultiviewBodyModel body_model);
        void ChangeViewDescriptors(string name, cv::Mat descriptors, vector<float> descriptors_confidences);

        void set_views_id(vector<int> views_id);
        vector<int> views_id();
        void set_views_descriptors(vector<cv::Mat> views_descriptors);
        vector<cv::Mat> views_descriptors();
        void set_views_descriptors_confidences(vector<vector<float> > views_descriptors_confidences);
        vector<vector<float> > views_descriptors_confidences();
        void set_views_names(vector<string> names);
        vector<string> views_names();
        void set_views_angle(vector<float> views_angles);
        vector<float> views_angles();

    private:


        vector<int> views_id_; // TODO: useful?
        vector<string> views_names_;
        vector<float> views_angles_;
        vector<cv::Mat> views_descriptors_;
        vector<vector<float> > views_descriptors_confidences_;
    };
}
#endif //MULTIVIEWBODYMODEL_MULTIVIEWBODYMODEL_H
