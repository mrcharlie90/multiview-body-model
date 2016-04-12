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
bool comparator(ConfidenceDescriptor c1, ConfidenceDescriptor c2) { return (c1.id < c2.id); }

/*
 *  Constructor: accept directly the vector with all views initialized.
 */
MultiviewBodyModel::MultiviewBodyModel(std::vector<ViewDetail> view_details)
{
    views = view_details;

    // Sorting each descriptor in each view
    for (int i = 0; i < view_details.size(); ++i)
    {
        sort(view_details[i].keypoints_descriptors.begin(), view_details[i].keypoints_descriptors.end(), comparator);
    }

}

void MultiviewBodyModel::AddDescriptors(ViewDetail view_detail, float angle)
{
    ConfidenceDescriptor cd;

    // Sorting
    sort(view_detail.keypoints_descriptors.begin(), view_detail.keypoints_descriptors.end(), comparator);
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




double multiviewbodymodel::Distance(MultiviewBodyModel body1, MultiviewBodyModel body2)
{

}







