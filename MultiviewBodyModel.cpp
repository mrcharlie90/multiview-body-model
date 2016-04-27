//
// Created by Mauro on 15/04/16.
//

#include "MultiviewBodyModel.h"

using namespace multiviewbodymodel;
using namespace cv;
using namespace std;

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
 * Normalize the confidences set by the skeletal tracker
 */
void multiviewbodymodel::MultiviewBodyModel::ConfidenceNormalization() {

    // Checking views' size
    if (views_descriptors_confidences_.size() == 0) {
        cerr << "No views acquired. Insert at least one view to call this procedure." << endl;
        return;
    }

    // TODO: i valori non vengono cambiati
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
 * Compute the distance between the specified view of the body models
 */
float MultiviewBodyModel::Distance(MultiviewBodyModel body_model, int view_id) {

    // Setting up the computation and checking sizes
    this->ConfidenceNormalization();
    body_model.ConfidenceNormalization();


    vector<int>::iterator iter = find(views_id_.begin(), views_id_.end(), view_id);

    // Computing distances
    Mat descriptors1 = views_descriptors_[view_id];
    Mat descriptors2 = body_model.views_descriptors()[view_id];

    assert(descriptors1.rows == descriptors2.rows);

    vector<float> confs1 = views_descriptors_confidences_[view_id];
    vector<float> confs2 = body_model.views_descriptors_confidences()[view_id];

    assert(confs1.size() == confs2.size());

    // Compute Euclidean Distance weighted with the confidence
    // of each keypoint descriptor
    float view_distance = 0.0f;
    for (int i = 0; i < descriptors1.rows; ++i) {
        view_distance += confs1[i] * confs2[i] * norm(descriptors1.row(i), descriptors2.row(i));
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
 * Change descriptors stored in a specific view.
 */
void MultiviewBodyModel::ChangeViewDescriptors(int view_id, cv::Mat descriptors, vector<float> descriptors_confidences) {
    long index = find(views_id_.begin(), views_id_.end(), view_id) - views_id_.begin();
    cout << index << endl;

    if (index >= views_id_.size())
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

void MultiviewBodyModel::set_views_id(vector<int> views_id) {
    views_id_ = views_id;
}

vector<int> MultiviewBodyModel::views_id() {
    return views_id_;
}

void MultiviewBodyModel::set_pose_side(int view_id, int pose_side) {
    pose_side_[view_id] = pose_side;
}

int MultiviewBodyModel::pose_side(int view_id) {
    return pose_side_[view_id];
}

void MultiviewBodyModel::ReadAndCompute(string path, string img_path, int view_id, string descriptor_extractor_type, float keypoint_size) {

    ifstream file(path);

    vector<KeyPoint> keypoints;
    vector<float> confidences;

    string line;
    int pose_side;
    int i = 0;
    while (getline(file, line) && i < 15)
    {
        // Current line
        istringstream iss(line);

        // Distinguish between position and confidence
        int value_type = 0;

        float x, y;
        string field;
        while (getline(iss, field, ','))
        {
            stringstream ss(field);

            switch (value_type) {
                case 0:
                    // Catch the x-position
                    ss >> x;
                    value_type++;
                    break;
                case 1:
                    // Catch the y-position
                    ss >> y;
                    value_type++;
                    break;
                case 2:
                    // Save the keypoint...
                    KeyPoint keypoint(Point2f(x, y), keypoint_size);
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
        i++;
    }

    // Last line will contain the pose side
    stringstream ss(line);
    ss >> pose_side;

    // Printing results
//    for (int i = 0; i < keypoints.size(); ++i) {
//        printf("%d: [%.2f, %.2f] | %.2f\n",
//               i, keypoints[i].pt.x, keypoints[i].pt.y, confidences[i]);
//    }
//
//    cout << pose_side << endl;

    // Read the image
    Mat img = imread(img_path);
    if (!img.data)
    {
        cerr << "Invalid image file." << endl;
        exit(0);
    }
//
//    Mat marked;
//    drawKeypoints(img, keypoints, marked, Scalar(0, 0, 255));
//
//    namedWindow("Win");
//    imshow("Win", marked);
//    waitKey(0);

    // Compute descriptors for this view
    Mat descriptors;
    Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create(descriptor_extractor_type);
    descriptor_extractor->compute(img, keypoints, descriptors);

    // Populate the body model with the results
    vector<int>::iterator iter = find(views_id_.begin(), views_id_.end(), view_id);
    if (iter == views_id_.end())
    {
        // The view is new, so add it to this model
        views_id_.push_back(view_id);
        pose_side_.push_back(pose_side);
        views_descriptors_.push_back(descriptors);
        views_descriptors_confidences_.push_back(confidences);
    }
    else
    {
        // Replace the view previously acquired
        int index = iter - views_id_.end() + 1; // index to the element to replace
        pose_side_[index] = pose_side;
        views_descriptors_[index] = descriptors;
        views_descriptors_confidences_[index] = confidences;
    }

}

cv::Mat MultiviewBodyModel::views_descriptors(int view_id) {
    return views_descriptors_[view_id];
}

vector<float> MultiviewBodyModel::confidences(int view_id) {
    return views_descriptors_confidences_[view_id];
}






