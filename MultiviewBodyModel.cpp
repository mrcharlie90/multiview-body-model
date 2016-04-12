//
// Created by Mauro on 07/04/16.
//


#include "MultiviewBodyModel.h"

using namespace multiviewbodymodel;
using namespace std;

/*
 * Comparator used for sorting compound vectors. Each time a new vector
 * is added, the array is sorted by ID (ascend order)
 */
bool confidence_comparator(ConfidenceDescriptor c1, ConfidenceDescriptor c2) { return (c1.id < c2.id); }

bool viewdetail_comparator(ViewDetail vd1, ViewDetail vd2) { return (vd1.angle < vd2.angle); }
/*
 *  Constructor: accept directly the vector with all views initialized.
 */
MultiviewBodyModel::MultiviewBodyModel(std::vector<ViewDetail> view_details)
{
    views = view_details;

    // Sorting each descriptor in each view
    for (int i = 0; i < view_details.size(); ++i)
    {
        sort(view_details[i].keypoints_descriptors.begin(), view_details[i].keypoints_descriptors.end(),
             confidence_comparator);
    }

}

void MultiviewBodyModel::AddDescriptors(ViewDetail view_detail, float angle)
{
    ConfidenceDescriptor cd;

    // Sorting
    sort(view_detail.keypoints_descriptors.begin(), view_detail.keypoints_descriptors.end(), confidence_comparator);
    views.push_back(view_detail);
}


/*
 * Normalize the confidence for each descriptor acquired:
 * this procedure should be called every time distance computation is computed.
 */
void MultiviewBodyModel::ConfidenceNormalization()
{
    if (views.size() == 0)
    {
        cerr << "No views acquired. Insert at least one view to call this procedure." << endl;
        return;
    }

    for (int i = 0; i < views.size(); ++i)
    {
        float overall_conf = 0.0f;
        int keypoints_number = views[i].keypoints_descriptors.size();
        vector<ConfidenceDescriptor> keypoint_descriptors = views[i].keypoints_descriptors;

        for (int j = 0; j < keypoints_number; ++j)
        {
            overall_conf = overall_conf + views[i].keypoints_descriptors[j].confidence;
        }

        for (int k = 0; k < keypoints_number; ++k)
        {
            float conf = keypoint_descriptors[k].confidence;
            views[i].keypoints_descriptors[k].confidence = conf / overall_conf;
        }
    }
}

/*
 * Return the number of views acquired.
 */
int MultiviewBodyModel::size()
{
    return views.size();
}

std::vector<ViewDetail> MultiviewBodyModel::getViews()
{
    return views;
}


/*
 * Returns true if the views contain the same angles
 * Note: both lists must be ordered ascend first.
 */
bool check_angles(std::vector<ViewDetail> views1, std::vector<ViewDetail> views2)
{
    // Checking lists' size
    if (views1.size() != views2.size())
    {
        cerr << "Cannot check angles: views.size() != views2.size()" << endl;
        return false;
    }

    // Comparing angles' values
    for (int i = 0; i < views1.size(); ++i)
    {
        if (views1[i].angle != views2[i].angle)
            return false;
    }

    return true;
}


/*
 *  Non-member function: given two body models computes the overall distance
 *  between two body models. It sums all the euclidean distances computed between each
 *  pair of keypoint's descriptors within each view.
 */
float multiviewbodymodel::overall_distance(MultiviewBodyModel b1, MultiviewBodyModel b2)
{
    vector<float> distances = view_distance(b1, b2);

    float overall_distance = accumulate(distances.begin(), distances.end(), 0);

    return overall_distance;
}

/*
 *  Non-member function: given two body models computes the distance
 *  between two each pair of views. It sums all the euclidean distances computed
 *  between each pair of keypoint's descriptors within each view.
 */
vector<float> multiviewbodymodel::view_distance(MultiviewBodyModel b1, MultiviewBodyModel b2)
{
    // Normalizing confidence
    b1.ConfidenceNormalization();
    b2.ConfidenceNormalization();

    // Getting views
    vector<ViewDetail> views1 = b1.getViews();
    vector<ViewDetail> views2 = b2.getViews();

    if (views1.size() != views2.size())
    {
        cerr << "Views size not equal." << endl;
        exit(0);
    }

    // Sorting by angle ascend
    sort(views1.begin(), views1.end(), viewdetail_comparator);
    sort(views2.begin(), views2.end(), viewdetail_comparator);

    // Checking angles
    if (!check_angles(views1, views2))
    {
        cerr << "The two views do not contain the same angles." << endl;
    }

    // Computing distances
    vector<float> distances;
    for (int i = 0; i < views1.size(); ++i)
    {
        // Support variables
        vector<ConfidenceDescriptor> k_descriptors1 = views1[i].keypoints_descriptors;
        vector<ConfidenceDescriptor> k_descriptors2 = views2[i].keypoints_descriptors;

        // Check size
        if (k_descriptors1.size() != k_descriptors2.size())
        {
            cerr << "The number of keypoints must be equal." << endl;
            exit(0);
        }

        // Compute the sum of the euclidean distance
        // between each pair of descriptors in the same view
        float view_distance = 0.0;
        for (int j = 1; j < k_descriptors1.size(); ++j)
        {
            // Getting the j-th keypoint
            ConfidenceDescriptor cd1 = k_descriptors1[j];
            ConfidenceDescriptor cd2 = k_descriptors2[j];

            // Computing euclidea distance weighted with the relative confidence
            view_distance += cd1.confidence * cd2.confidence * norm(cd1.descriptor, cd2.descriptor, cv::NORM_L2);

            distances.push_back(view_distance);
        }
    }

    return distances;
}









