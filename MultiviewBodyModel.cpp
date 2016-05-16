//
// Created by Mauro on 15/04/16.
//

#include "MultiviewBodyModel.h"

using namespace multiviewbodymodel;
using namespace std;

/*
 * Constructor
 */
MultiviewBodyModel::MultiviewBodyModel(int max_poses) {
    max_poses_ = max_poses;
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

///*
// * Given the view id, compute the euclidean distance between
// * two views of the body models.
// */
//float MultiviewBodyModel::Distance(MultiviewBodyModel body_model, int view_id) {
//
//    // Setting up the computation and checking sizes
//    this->ConfidenceNormalization();
//    body_model.ConfidenceNormalization();
//
//    // Computing distances
//    cv::Mat descriptors1 = views_descriptors_[view_id];
//    cv::Mat descriptors2 = body_model.views_descriptors()[view_id];
//
//    assert(descriptors1.rows == descriptors2.rows);
//
//    vector<float> confs1 = views_descriptors_confidences_[view_id];
//    vector<float> confs2 = body_model.views_descriptors_confidences()[view_id];
//
//    assert(confs1.size() == confs2.size());
//
//
//    // Compute Euclidean Distance weighted with the confidence
//    // of each keypoint descriptor
//    float view_distance = 0.0f;
//    for (int i = 0; i < descriptors1.rows; ++i) {
//        Mat normalized_descriptor1;
//        normalize(descriptors1.row(i), normalized_descriptor1);
//        Mat normalized_descriptor2;
//        normalize(descriptors2.row(i), normalized_descriptor2);
//
//        view_distance += confs1[i] * confs2[i] * norm(normalized_descriptor1, normalized_descriptor2);
//    }
//
//    return view_distance;
//}

///*
// * Compute the distance between views of two body model
// */
//std::vector<float> MultiviewBodyModel::Distances(MultiviewBodyModel body_model) {
//    this->ConfidenceNormalization();
//    body_model.ConfidenceNormalization();
//
//    assert(views_descriptors_.size() == body_model.views_descriptors().size());
//
//    // Computing the vector of distances
//    vector<float> distances;
//    for (int i = 0; i < views_descriptors_.size(); ++i) {
//        distances.push_back(Distance(body_model, i));
//    }
//
//    return distances;
//}

/*
 * Adds new skeleton descriptors to the body model. The *_skel.txt file should contain 15 keypoints'
 * [float] coordinates with the pose side number [int] at the end.
 * Returns true if the pose's descriptors are successfully saved and false if the pose is already acquired.
 */
bool MultiviewBodyModel::ReadAndCompute(string path, string img_path, string descriptor_extractor_type, int keypoint_size) {
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
    keypoint_size_ = keypoint_size;
    views_keypoints_.push_back(keypoints);

    // Last line contains the pose side
    std::stringstream ss(line);
    ss >> pose_side;

    // Check if the pose already exists...
    vector<int>::iterator iter = find(pose_side_.begin(), pose_side_.end(), pose_side);

    // ...if so, populate the body model with the data, otherwise discard the data
    if (iter == pose_side_.end()) {
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
        descriptor_extractor_type_ = descriptor_extractor_type;

        pose_side_.push_back(pose_side);
        views_descriptors_.push_back(descriptors);
        views_descriptors_confidences_.push_back(confidences);

        return true;
    }

    return false;
}


float MultiviewBodyModel::match(Mat query_descritptors, vector<float> confidences, int pose_side, bool occlusion_search)
{
    assert(!pose_side_.empty() && pose_side_.size() == max_poses_);
    assert(confidences.size() == query_descritptors.rows);

    // Mask creation
//    Mat  mask(static_cast<int>(confidences.size()), 1, CV_8U);
//    for (int k = 0; k < confidences.size(); ++k) {
//        if (confidences[k] > 0) {
//            mask.at<char>(k, 1) = 1;
//        }
//        else
//            mask.at<char>(k, 1) = 0;
//    }
//
//    std::cout << mask << std::endl;


    for (int i = 0; i < pose_side_.size(); ++i) {
        if (pose_side_[i] == pose_side) {
            vector<char> confidence_mask;
            create_confidence_mask(confidences, views_descriptors_confidences_[i], confidence_mask);

            for (int j = 0; j < confidence_mask.size(); ++j) {
                std::cout << confidence_mask[j] << std::endl;
            }

            float average_distance = 0.0;
            int descriptors_count = 0;
            for (int k = 0; k < views_descriptors_[i].rows; ++k) {
                char value = confidence_mask[k];

                // If the occlusion_search is true, look
                Mat descriptor_occluded;
                double dist;
                if(value == 1 && occlusion_search) {
                    if (get_descriptor_occluded(k, descriptor_occluded)) {
                        // A descriptor is found, so compute the distance
                        dist = norm(query_descritptors.row(k), descriptor_occluded);

                        std::cout << k << ":" << dist << " " << std::endl;
                        average_distance += dist;
                        descriptors_count++;
                    }
                }
                else if (value == 2) {
                    dist = norm(query_descritptors.row(k), views_descriptors_[i].row(k));
                    std::cout << k << ":" << dist << std::endl;
                    average_distance += dist;
                    descriptors_count++;
                }
            }
            cout << average_distance / descriptors_count << endl;
            return average_distance / descriptors_count;
        }
    }
    return -1;
}

/*
 * Get and set methods
 */
vector<cv::Mat> MultiviewBodyModel::views_descriptors() {
    return views_descriptors_;
}

vector<vector<float> > MultiviewBodyModel::views_descriptors_confidences() {
    return views_descriptors_confidences_;
}

bool MultiviewBodyModel::ready() {
    return (pose_side_.size() == max_poses_);
}

/*
 * The matching is performed by following the mask values:
 * if mask(i,j) = 0 -> Don't consider keypoints
 * if mask(i,j) = 1 -> Find the keypoint occluded in other views
 * if mask(i,j) = 2 -> Compute the distance between the keupoints
 */
void MultiviewBodyModel::create_confidence_mask(vector<float> &query_confidences, vector<float> &train_confidences,
                                                vector<char> &out_mask) {


    assert(query_confidences.size() == train_confidences.size());

    for (int k = 0; k < query_confidences.size(); ++k) {
        if (query_confidences[k] > 0 && train_confidences[k] == 0) {
            // Keypoint occluded in the training frame
            out_mask.push_back(1);
        }
        else if (query_confidences[k] > 0 && train_confidences[k] > 0) {
            // Both keypoints visible
            out_mask.push_back(2);
        }
        else {
            // Test keypoint occluded or both occluded: discard the keypoints
            out_mask.push_back(0);
        }
    }
}

/**
 * Finds a non occluded descriptor in another view of the model.
 * returns true if the keypoint is found, false otherwise
 */
bool MultiviewBodyModel::get_descriptor_occluded(int keypoint_index, Mat &descriptor_occluded) {


    // Find a non-occluded descriptor in one pose
    for (int i = 0; i < views_descriptors_.size(); ++i) {
        if (views_descriptors_confidences_[i][keypoint_index] > 0) {
            std::cout << "descriptor k = " << keypoint_index << " found at view = " << i << std::endl;
            descriptor_occluded = views_descriptors_[i].row(keypoint_index);
            return true;
        }
    }
    return false;
}






























