//
// Created by Mauro on 15/04/16.
//

#include "MultiviewBodyModel.h"

using namespace multiviewbodymodel;
using namespace cv;
using namespace std;

MultiviewBodyModel::MultiviewBodyModel() {

}

/*
 *  Constructor: initialize the body model with the views' descriptors
 */
MultiviewBodyModel::MultiviewBodyModel(vector<int> views_id, vector<string> views_names,
                                       vector<float> views_angles, vector<cv::Mat> views_descriptors,
                                       vector<vector<float> > views_descriptors_confidences) {
    views_id_ = views_id;
    views_descriptors_ = views_descriptors;
    views_descriptors_confidences_ = views_descriptors_confidences;
    views_names_ = views_names;
    views_angles_ = views_angles;

}

/*
 * Normalize the confidences set by the skeletal tracker
 */
void multiviewbodymodel::MultiviewBodyModel::ConfidenceNormalization() {

    // Checking views' size
    if (views_descriptors_confidences_.size() == 0) {
        cerr << "No views acquired. Insert at least one view to call this procedure." << endl;
        return;
    }

    // Confidence normalization
    for (int i = 0; i < views_descriptors_confidences_.size(); ++i) {

        vector<float> descriptors_confidences = views_descriptors_confidences_[i];

        // Summing all confidences
        float overall_conf = 0.0f;
        for (int k = 0; k < descriptors_confidences.size(); ++k) {
            overall_conf += descriptors_confidences[k];
        }

        // Confidence normalization
        for (int j = 0; j < descriptors_confidences.size(); ++j) {
            descriptors_confidences[j] = descriptors_confidences[j] / overall_conf;
        }
    }
}

/*
 * Compute the distance between views of two body model
 */
std::vector<float> MultiviewBodyModel::Distance(MultiviewBodyModel body_model) {

    // Setting up the computation and checking sizes
    this->ConfidenceNormalization();
    body_model.ConfidenceNormalization();

    vector<Mat> views_descriptors1 = views_descriptors_;
    vector<Mat> views_descriptors2 = body_model.views_descriptors();

    if (views_descriptors1.size() != views_descriptors2.size()) {
        cerr << "Views size not equal." << endl;
        exit(0);
    }

    // Computing distances
    vector<float> distances;
    for (int i = 0; i < views_descriptors1.size(); ++i) {
        Mat descriptors1 = views_descriptors1[i];
        Mat descriptors2 = views_descriptors2[i];

        assert(descriptors1.rows == descriptors2.rows);

        vector<float> confs1 = views_descriptors_confidences_[i];
        vector<float> confs2 = body_model.views_descriptors_confidences()[i];

        assert(confs1.size() == confs2.size());

        // Compute Euclidean distance weighted with the confidence of each descriptor
        float view_distance = 0.0f;
        for (int j = 0; j < descriptors1.rows; ++j) {
            view_distance += confs1[j] * confs2[j] * norm(descriptors1.row(j), descriptors2.row(j));
        }

        distances.push_back(view_distance);
    }

    return distances;
}

/*
 * Change descriptors stored in a specific view.
 */
void MultiviewBodyModel::ChangeViewDescriptors(string name, cv::Mat descriptors,
                                               vector<float> descriptors_confidences) {
    long index = find(views_names_.begin(), views_names_.end(), name) - views_names_.begin();
    cout << index << endl;

    if (index >= views_names_.size())
    {
        cerr << "The view searched does not exists." << endl;
        exit(-1);
    }

    views_descriptors_[index] = descriptors;
    views_descriptors_confidences_[index] = descriptors_confidences;
}

/*
 * Get and set methods
 */
void MultiviewBodyModel::set_views_descriptors(vector<cv::Mat> views_descriptors) {
    views_descriptors_ = views_descriptors;
}
vector<cv::Mat> MultiviewBodyModel::views_descriptors() {
    return views_descriptors_;
}

void MultiviewBodyModel::set_views_descriptors_confidences(
        vector<vector<float> > views_descriptors_confidences) {
    views_descriptors_confidences_ = views_descriptors_confidences;
}

vector<vector<float> > MultiviewBodyModel::views_descriptors_confidences() {
    return views_descriptors_confidences_;
}

void MultiviewBodyModel::set_views_names(vector<string> views_names) {
    views_names_ = views_names;
}

vector<string> MultiviewBodyModel::views_names() {
    return views_names_;
}

void MultiviewBodyModel::set_views_angle(vector<float> views_angles) {
    views_angles_ = views_angles;
}

vector<float> MultiviewBodyModel::views_angles() {
    return views_angles_;
}

void MultiviewBodyModel::set_views_id(vector<int> views_id) {
    views_id_ = views_id;
}

vector<int> MultiviewBodyModel::views_id() {
    return views_id_;
}









































