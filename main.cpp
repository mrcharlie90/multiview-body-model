//
// Created by Mauro on 10/05/16.
//

#include "MultiviewBodyModel.h"

using namespace multiviewbodymodel;
using namespace std;
void read_skel(string skel_path, string img_path, int keypoint_size, string descriptor_extractor_type,
               Mat &out_image, vector<KeyPoint> &out_keypoints, Mat &out_descriptors, int &pose_side);

int main()
{
    Mat query_skel_path;
    Mat query_image_path;
    Mat query_image;

    string DT = "SIFT";
    vector<KeyPoint> query_keypoints;
    Mat query_descriptors;
    int query_pose_side;
    int keypoint_size = 9;

    read_skel("../ds/gianluca_sync/c00000_skel.txt", "../ds/gianluca_sync/c00000.png",
              keypoint_size, DT, query_image, query_keypoints, query_descriptors, query_pose_side);

    if (query_image.data)
    {
        Mat img;
        drawKeypoints(query_image, query_keypoints, img);
        namedWindow("ciao");
        imshow("ciao", img);
        waitKey(0);
        cout << query_descriptors << endl;
        cout << query_pose_side << endl;
    }
    else
        cerr << "Fail reaing the image" << endl;





    return 0;
}

void read_skel(string skel_path, string img_path, int keypoint_size, string descriptor_extractor_type,
               Mat &out_image, vector<KeyPoint> &out_keypoints, Mat &out_descriptors, int &pose_side) {

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

                    // Reset to 0 for the next keypoint
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
