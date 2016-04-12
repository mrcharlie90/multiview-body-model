//
// Created by Mauro on 07/04/16.
//
// OpenCV ver. 2.4
//

#include <iostream>
#include <vector>
#include <numeric>

#include <opencv/cv.h>
#include <opencv2/nonfree/nonfree.hpp>

#ifndef HELLOWORLD_MULTIVIEWBODYMODEL_H
#define HELLOWORLD_MULTIVIEWBODYMODEL_H


namespace multiviewbodymodel {

    /*
     * Struct for storing a confidence for each
     * keypoint.
     */
    struct ConfidenceDescriptor {
        int id;
        float confidence;
        cv::Mat descriptor;

    };

    struct ViewDetail
    {
        float angle;
        std::vector<ConfidenceDescriptor> keypoints_descriptors;
        float overall_confidence; // TODO: set default to 1

    };

    /*
     * Class definition
     */
    class MultiviewBodyModel
    {
    private:
        // Required member variables
        std::vector<ViewDetail> views;

    public:

        MultiviewBodyModel(std::vector<ViewDetail> view_details);

        void AddDescriptors(ViewDetail view_detail, float angle);

        void ConfidenceNormalization();

        int size();

    };

    /*
     * Nonmember functions
     */
    double Distance(MultiviewBodyModel body1, MultiviewBodyModel body2);
}
#endif //HELLOWORLD_MULTIVIEWBODYMODEL_H
