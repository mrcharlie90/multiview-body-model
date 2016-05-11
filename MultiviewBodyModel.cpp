//
// Created by Mauro on 15/04/16.
//

#include "MultiviewBodyModel.h"

using namespace multiviewbodymodel;

MultiviewBodyModel::MultiviewBodyModel() {

}

MultiviewBodyModel::MultiviewBodyModel(vector<int> views_id, vector<int> pose_side, vector<cv::Mat> views_descriptors,
                                       vector<vector<float> > views_descriptors_confidences) {
    views_id_ = views_id;
    pose_side_ = pose_side;
    views_descriptors_ = views_descriptors;
    views_descriptors_confidences_ = views_descriptors_confidences;
}

/*
 * Normalize the confidences set by the skeletal tracker.
 */
void multiviewbodymodel::MultiviewBodyModel::ConfidenceNormalization() {

    // Checking views' size
    if (views_descriptors_confidences_.size() == 0) {
        std::cerr << "No views acquired. Insert at least one view to call this procedure." << std::endl;
        return;
    }

    // Confidence normalization
    for (int i = 0; i < views_descriptors_confidences_.size(); ++i) {

        vector<float> *descriptors_confidences = &views_descriptors_confidences_[i];

        // Summing all confidences
        float overall_conf = 0.0f;
        for (int k = 0; k < descriptors_confidences->size(); ++k) {
            overall_conf += (*descriptors_confidences)[k];
        }

        // Confidence normalization
        for (int j = 0; j < descriptors_confidences->size(); ++j) {
            (*descriptors_confidences)[j] = (*descriptors_confidences)[j] / overall_conf;
        }
    }
}

/*
 * Given the view id, compute the euclidean distance between
 * two views of the body models.
 */
float MultiviewBodyModel::Distance(MultiviewBodyModel body_model, int view_id) {

    // Setting up the computation and checking sizes
    this->ConfidenceNormalization();
    body_model.ConfidenceNormalization();

    // Computing distances
    cv::Mat descriptors1 = views_descriptors_[view_id];
    cv::Mat descriptors2 = body_model.views_descriptors()[view_id];

    assert(descriptors1.rows == descriptors2.rows);

    vector<float> confs1 = views_descriptors_confidences_[view_id];
    vector<float> confs2 = body_model.views_descriptors_confidences()[view_id];

    assert(confs1.size() == confs2.size());


    // Compute Euclidean Distance weighted with the confidence
    // of each keypoint descriptor
    float view_distance = 0.0f;
    for (int i = 0; i < descriptors1.rows; ++i) {
        Mat normalized_descriptor1;
        normalize(descriptors1.row(i), normalized_descriptor1);
        Mat normalized_descriptor2;
        normalize(descriptors2.row(i), normalized_descriptor2);

        view_distance += confs1[i] * confs2[i] * norm(normalized_descriptor1, normalized_descriptor2);
    }

    return view_distance;
}

/*
 * Compute the distance between views of two body model
 */
std::vector<float> MultiviewBodyModel::Distances(MultiviewBodyModel body_model) {
    this->ConfidenceNormalization();
    body_model.ConfidenceNormalization();

    assert(views_id_.size() == body_model.views_id().size());

    // Computing the vector of distances
    vector<float> distances;
    for (int i = 0; i < views_id_.size(); ++i) {
        distances.push_back(Distance(body_model, i));
    }

    return distances;
}

/*
 * Populate the body model given a file containing the 15 keypoints' position
 * and the pose side. If one view is already saved, overwrite it.
 */
void MultiviewBodyModel::ReadAndCompute(string path, string img_path, int view_id, string descriptor_extractor_type, float keypoint_size) {
    // Output variables
    vector<cv::KeyPoint> keypoints;
    vector<float> confidences;

    int pose_side;

    // File reading
    string line;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Invalid file name." << std::endl;
        exit(-1);
    }

    int i = 0;
    while (getline(file, line) && i < 15) {
        // Current line
        std::istringstream iss(line);

        int value_type = 0; // 0:x-pos, 1:y-pos, 2:confidence
        float x = 0.0f; // x-position
        float y = 0.0f; // y-position

        string field;
        while (getline(iss, field, ',')) {
            std::stringstream ss(field);
            switch (value_type) {
                case 0:
                    // Catch the x-position
                    ss >> x;
                    ++value_type;
                    break;
                case 1:
                    // Catch the y-position
                    ss >> y;
                    ++value_type;
                    break;
                case 2:
                    // Save the keypoint...
                    cv::KeyPoint keypoint(cv::Point2f(x, y), keypoint_size);
                    keypoints.push_back(keypoint);

                    // ...and the confidence
                    float conf;
                    ss >> conf;
                    if (conf < 0)
                        confidences.push_back(0);
                    else
                        confidences.push_back(conf);

                    // Reset to 0 for the next keypoint
                    value_type %= 2;
                    break;
            }
        }
        ++i;
    }
    views_keypoints_.push_back(keypoints);

    // Last line contains the pose side
    std::stringstream ss(line);
    ss >> pose_side;

    // Read image
    cv::Mat img = cv::imread(img_path);
    if (!img.data) {
        std::cerr << "Invalid image file." << std::endl;
        exit(0);
    }

    views_images_.push_back(img);

    // Compute descriptors for this view
    cv::Mat descriptors;
    cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = cv::DescriptorExtractor::create(descriptor_extractor_type);
    descriptor_extractor->compute(img, keypoints, descriptors);

    // Populate the body model with the results and if a view already exists, replace it
    vector<int>::iterator iter = find(views_id_.begin(), views_id_.end(), view_id);
    if (iter == views_id_.end()) {
        // The view is new, so add it to this model
        views_id_.push_back(view_id);
        pose_side_.push_back(pose_side);
        views_descriptors_.push_back(descriptors);
        views_descriptors_confidences_.push_back(confidences);
    }
    else {
        // Replace the view previously acquired
        long index = iter - views_id_.begin(); // index to the element to replace
        pose_side_[index] = pose_side;
        views_descriptors_.at(index) = descriptors;
        views_descriptors_confidences_[index] = confidences;
    }
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

void MultiviewBodyModel::set_views_id(vector<int> views_id) {
    views_id_ = views_id;
}

vector<int> MultiviewBodyModel::views_id() {
    return views_id_;
}

void MultiviewBodyModel::set_view_descriptors(int view_id, Mat descriptors) {
    views_descriptors_[view_id] = descriptors;
}

cv::Mat MultiviewBodyModel::view_descriptors(int view_id) {
    return views_descriptors_[view_id];
}

int MultiviewBodyModel::pose_side(int view_id) {
    return pose_side_[view_id];
}

vector<Mat> MultiviewBodyModel::ImagesByPoseSide(int pose_side) {
    vector<Mat> imgs;
    for (int i = 0; i < views_images_.size(); ++i) {
        if (pose_side_[i] == pose_side)
            imgs.push_back(views_images_[i]);
    }

    return imgs;
}

vector<Mat> MultiviewBodyModel::DescriptorsByPoseSide(int pose_side) {
    vector<Mat> descriptors;
    for (int i = 0; i < views_descriptors_.size(); ++i) {
        if (pose_side_[i] == pose_side) {
            descriptors.push_back(views_descriptors_[i]);
        }
    }

    return descriptors;
}

vector<vector<KeyPoint> > MultiviewBodyModel::KeypointsByPoseSide(int pose_side) {
    vector<vector<KeyPoint> > keypoints;
    for (int i = 0; i < views_keypoints_.size(); ++i) {
        if (pose_side_[i] == pose_side) {
            keypoints.push_back(views_keypoints_[i]);
        }
    }

    return keypoints;
}













