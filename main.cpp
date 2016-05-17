//
// Created by Mauro on 10/05/16.
//

#include "MultiviewBodyModel.h"

using namespace multiviewbodymodel;
using namespace std;

void read_skel(string skel_path, string img_path, string descriptor_extractor_type, int keypoint_size,
               Mat &out_image, vector<KeyPoint> &out_keypoints, vector<float> &confidences, Mat &out_descriptors,
               int &pose_side);

void load_train_paths(string main_path, vector<string> persons_names, vector<string> views_names,
                      vector<int> num_images,
                      vector<vector<string> > &imgs_paths, vector<vector<string> > &skel_paths);

bool load_models(vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                 vector<MultiviewBodyModel> &models);

void load_query_paths(vector<vector<string>> &train_skels_paths, vector<vector<string>> &train_imgs_paths,
                      vector<string> &query_skels_paths, vector<string> &query_imgs_paths);

int main()
{

    vector<string> persons_names;
    persons_names.push_back("gianluca_sync");
    persons_names.push_back("marco_sync");
    persons_names.push_back("matteol_sync");

    vector<string> views_names;
    views_names.push_back("c");
    views_names.push_back("l");
    views_names.push_back("r");

    vector<int> num_images;
    num_images.push_back(74);
    num_images.push_back(84);
    num_images.push_back(68);

    string main_path = "../ds/";

    vector<vector<string> > train_imgs_paths;
    vector<vector<string> > train_skels_paths;

    load_train_paths(main_path, persons_names, views_names, num_images, train_imgs_paths, train_skels_paths);

    vector<string> query_imgs_paths;
    vector<string> query_skels_paths;

    load_query_paths(train_skels_paths, train_imgs_paths, query_skels_paths, query_imgs_paths);

    assert(query_imgs_paths.size() == query_skels_paths.size());


    vector<MultiviewBodyModel> models;
    while(load_models(train_imgs_paths, train_skels_paths, models)) {
        for (int i = 0; i < query_imgs_paths.size(); ++i) {
            // Load query image
            Mat query_image;
            vector<KeyPoint> query_keypoints;
            vector<float> query_confidences;
            Mat query_descriptors;
            int query_pose_side;
            read_skel(query_skels_paths[i], query_imgs_paths[i], "BruteForce", 3,
                      query_image, query_keypoints, query_confidences, query_descriptors, query_pose_side);

            priority_queue<float> scores;

            for (int k = 0; k < models.size(); ++k) {
                float score = models[k].match(query_descriptors, query_confidences, query_pose_side);
                scores.push(score);
            }
        }
    }


    return 0;
}

void load_query_paths(vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                      vector<string> &query_skels_paths, vector<string> &query_imgs_paths) {

    assert(train_imgs_paths.size() == train_skels_paths.size());

    for (int i = 0; i < train_imgs_paths.size(); ++i) {
        for (int j = 0; j < train_imgs_paths[i].size(); ++j) {
            query_imgs_paths.push_back(train_skels_paths[i][j]);
            query_skels_paths.push_back((train_imgs_paths[i][j]));
        }
    }
}

bool load_models(vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                 vector<MultiviewBodyModel> &models) {

    // TODO:

    for (int i = 0; i < train_skels_paths.size(); ++i) {

    }
}


void load_train_paths(string main_path, vector<string> persons_names, vector<string> views_names,
                      vector<int> num_images,
                      vector<vector<string> > &imgs_paths, vector<vector<string> > &skels_paths) {

    assert(persons_names.size() == num_images.size());

    stringstream ss_imgs, ss_skels;

    for (int i = 0; i < persons_names.size(); ++i) {
        for (int j = 0; j < views_names.size(); ++j) {
            vector<string> imgs_path;
            vector<string> skels_path;
            for (int k = 0; k <= num_images[i]; ++k) {
                if (k < 10) {
                    ss_imgs << main_path << persons_names[i] << "/" << views_names[j] << "0000" << k << ".png";
                    ss_skels << main_path << persons_names[i] << "/" << views_names[j] << "0000" << k << "_skel.txt";
                }
                else {
                    ss_imgs << main_path << persons_names[i] << "/" << views_names[j] << "000" << k << ".png";
                    ss_skels << main_path << persons_names[i] << "/" << views_names[j] << "000" << k << "_skel.txt";
                }

                imgs_path.push_back(ss_imgs.str());
                skels_path.push_back(ss_skels.str());

                ss_imgs.str("");
                ss_skels.str("");
            }

            imgs_paths.push_back(imgs_path);
            skels_paths.push_back(skels_path);
        }
    }
}

void read_skel(string skel_path, string img_path, string descriptor_extractor_type, int keypoint_size,
               Mat &out_image, vector<KeyPoint> &out_keypoints, vector<float> &confidences, Mat &out_descriptors,
               int &pose_side) {
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