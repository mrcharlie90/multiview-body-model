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
                 vector<MultiviewBodyModel> &models, string descriptor_extractor_type,
                 int keypoint_size, int max_poses);

void load_query_paths(vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                      vector<string> &query_skels_paths, vector<string> &query_imgs_paths);

template<typename T>
void print_list(vector<T> vect);

template<typename T>
void print_list(priority_queue<T, vector<T>, greater<T> > queue);

int main()
{
    // Strings parameters
    string main_path = "../ds/";

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

    // Load the training set
    vector<vector<string> > train_imgs_paths;
    vector<vector<string> > train_skels_paths;
    load_train_paths(main_path, persons_names, views_names, num_images, train_skels_paths, train_imgs_paths);

    // Load the queries
    vector<string> query_imgs_paths;
    vector<string> query_skels_paths;
    load_query_paths(train_skels_paths, train_imgs_paths, query_skels_paths, query_imgs_paths);
    assert(query_imgs_paths.size() == query_skels_paths.size());

    // Parameters
    int max_poses = 4;
    string descriptor_extractor_type = "SIFT";
    int keypoint_size = 9;


    // Testing
    vector<MultiviewBodyModel> models;
    while(load_models(train_skels_paths, train_imgs_paths, models, descriptor_extractor_type, keypoint_size, max_poses)) {

        cout << "models.size = " << models.size() << endl;
        for (int i = 0; i < query_imgs_paths.size(); ++i) {
            // Load the query image
            Mat query_image;
            Mat query_descriptors;
            vector<KeyPoint> query_keypoints;
            vector<float> query_confidences;
            int query_pose_side;
            read_skel(query_skels_paths[i], query_imgs_paths[i], descriptor_extractor_type, keypoint_size,
                      query_image, query_keypoints, query_confidences, query_descriptors, query_pose_side);

            // List of scores ordered in descending order
            priority_queue<float, vector<float>, greater<float> > scores;

            // Compute the match between the query image and each model
            // then insert the result into the list
            for (int k = 0; k < models.size(); ++k) {
                float score = models[k].match(query_descriptors, query_confidences, query_pose_side);
                scores.push(score);
            }

            print_list(scores);
            cout << "--------------" << endl;
        }
    }
    return 0;
}

/**
 * Converts the training paths to lists of paths, used for the query images.
 */
/*
 * INPUT:
 *  list[0]
 *    -> people1/path/to/image1
 *    -> people1/path/to/image2
 *    ...
 *    -> people1/path/to/imageN
 *  .
 *  .
 *
 *  list[R]
 *    -> peopleR/path/to/image1
 *    -> peopleR/path/to/image2
 *    ...
 *    -> peopleR/path/to/imageM
 *
 *  OUTPUT:
 *   people1/path/to/image1
 *   people1/path/to/image2
 *   ...
 *   people1/path/to/imageN
 *   .
 *   .
 *   peopleR/path/to/image1
 *   peopleR/path/to/image2
 *   ...
 *   peopleR/path/to/imageM
 */
void load_query_paths(vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                      vector<string> &query_skels_paths, vector<string> &query_imgs_paths) {

    assert(train_imgs_paths.size() == train_skels_paths.size());

    for (int i = 0; i < train_imgs_paths.size(); ++i) {
        for (int j = 0; j < train_imgs_paths[i].size(); ++j) {
            query_skels_paths.push_back((train_skels_paths[i][j]));
            query_imgs_paths.push_back(train_imgs_paths[i][j]);
        }
    }
}

/**
 * Loads training paths ordered by person.
 */
void load_train_paths(string main_path, vector<string> persons_names, vector<string> views_names,
                      vector<int> num_images,
                      vector<vector<string> > &skels_paths, vector<vector<string> > &imgs_paths) {

    assert(persons_names.size() == num_images.size());

    stringstream ss_imgs, ss_skels;

    for (int i = 0; i < persons_names.size(); ++i) {
        vector<string> imgs_path;
        vector<string> skels_path;
        for (int j = 0; j < views_names.size(); ++j) {
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
        }
        imgs_paths.push_back(imgs_path);
        skels_paths.push_back(skels_path);
    }
}

/*
 * Initializes models poses given the training images paths and skeletons.
 */
bool load_models(vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                 vector<MultiviewBodyModel> &models, string descriptor_extractor_type,
                 int keypoint_size, int max_poses) {
    // Checking dimensions
    assert(train_imgs_paths.size() == train_skels_paths.size());


    for (int i = 0; i < train_skels_paths.size(); ++i) {
        // Checking dimensions
        assert(train_imgs_paths[i].size() == train_skels_paths[i].size());

        MultiviewBodyModel body_model(max_poses);

        int j = 0;
        while (!body_model.ready() && j < train_skels_paths[i].size()) {
            // Insert the pose if not present, and remove it from the paths
            bool condition = body_model.ReadAndCompute(train_skels_paths[i][j], train_imgs_paths[i][j],
                                                       descriptor_extractor_type, keypoint_size);
            if (condition) {
                train_imgs_paths[i].erase(train_imgs_paths[i].begin() + j);
                train_skels_paths[i].erase(train_skels_paths[i].begin() + j);
            }
            ++j;
        }

        // If the model contains all poses then add it to the vector
        // otherwise the model is not valid, then exit.
        if (body_model.ready())
            models.push_back(body_model);
        else
            return false;
    }
    return true;
}

/*
 * Reads the skeleton from a file and  computes its descritpors. This is used to compute descriptors of an image
 * on the fly.
 */
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

template<typename T>
void print_list(vector<T> vect) {
    cout << "[" << vect[0];
    for (int i = 1; i < vect.size(); ++i) {
        cout << ", " << vect[i];
    }
    cout << "]" << endl;
}

template<typename T>
void print_list(priority_queue<T, vector<T>, greater<T> > queue) {
    cout << "[" << queue.top();
    queue.pop();
    while (!queue.empty()) {
        cout << ", " << queue.top();
        queue.pop();
    }
    cout << "]" << endl;
}
