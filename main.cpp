//
// Created by Mauro on 10/05/16.
//

#include "MultiviewBodyModel.h"

using namespace multiviewbodymodel;
using namespace std;

void read_skel(string skel_path, string img_path, string descriptor_extractor_type, int keypoint_size,
               Mat &out_image, vector<KeyPoint> &out_keypoints, vector<float> &confidences, Mat &out_descriptors, int &pose_side);

int main()
{
    // Paths
    const int n_people = 3; // number of people
    string str_paths[n_people] = {"../ds/gianluca_sync/", "../ds/marco_sync/", "../ds/matteol_sync/"};
    int n_images[n_people] = {74, 84, 68};
    const int n_views = 3;
    string str_views[n_views] = {"c", "r", "l"};



    string imgs[5] = {
            "../ds/gianluca_sync/c00000.png",
            "../ds/gianluca_sync/c00009.png",
            "../ds/gianluca_sync/c00017.png",
            "../ds/gianluca_sync/c00047.png",
            "../ds/gianluca_sync/c00063.png"
    };

    string skels[5] = {
            "../ds/gianluca_sync/c00000_skel.txt", // 2
            "../ds/gianluca_sync/c00009_skel.txt", // 1
            "../ds/gianluca_sync/c00017_skel.txt", // 4
            "../ds/gianluca_sync/c00047_skel.txt", // 4
            "../ds/gianluca_sync/c00063_skel.txt" // 3
    };

    MultiviewBodyModel mbm1(4);

    int i = 0;
    while(!mbm1.ready() && i < 5) {
        mbm1.ReadAndCompute(skels[i], imgs[i], "SIFT", 9);
        i++;
    }

    Mat out_image;
    vector<KeyPoint> out_keypoints;
    vector<float> out_confidences;
    Mat out_descriptors;
    int out_pose_side;
    read_skel("../ds/gianluca_sync/c00070_skel.txt", "../ds/gianluca_sync/c00070.png", "SIFT", 9, out_image, out_keypoints,
              out_confidences, out_descriptors, out_pose_side);

    mbm1.match(out_descriptors, out_confidences, out_pose_side, "BruteForce");

    return 0;
}

void read_skel(string skel_path, string img_path, string descriptor_extractor_type, int keypoint_size,
               Mat &out_image, vector<KeyPoint> &out_keypoints, vector<float> &confidences, Mat &out_descriptors, int &pose_side) {
    // Read the file
    string line;
    std::ifstream file(skel_path);
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
                    out_keypoints.push_back(keypoint);

                    // ...and the confidence
                    float conf;
                    ss >> conf;
                    if (conf < 0)
                        confidences.push_back(0);
                    else
                        confidences.push_back(conf);

                    // Reset to 0 and go to the next keypoint
                    value_type %= 2;
                    break;
            }
        }
        ++i;
    }

    // Last line contains the pose side
    std::stringstream ss(line);
    ss >> pose_side;

    // Read image
    out_image = cv::imread(img_path);
    if (!out_image.data) {
        std::cerr << "Invalid image file." << std::endl;
    }

    // Compute descriptors for this view
    cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = cv::DescriptorExtractor::create(descriptor_extractor_type);
    descriptor_extractor->compute(out_image, out_keypoints, out_descriptors);
}
