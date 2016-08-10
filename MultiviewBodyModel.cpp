// Copyright (c) [2016] [Mauro Piazza]
//
//          IASLab License
//
// This file contains all methods deifinition of the MultiviewBodyModel class
// and the multiviewbodymodel namespace.
//

#include "MultiviewBodyModel.h"

namespace multiviewbodymodel {
using std::vector;
using std::cout;
using std::endl;
using std::cerr;
using std::stringstream;
using std::istringstream;
using std::priority_queue;
using cv::string;
using cv::Mat;
using cv::Range;
using cv::KeyPoint;
using cv::norm;

// ------------------------------------------------------------------------- //
//                           Methods definitions                             //
// ------------------------------------------------------------------------- //

void MultiviewBodyModel::read_pose_compute_descriptors(string img_path, string skel_path, int keypoint_size,
                                                       string descriptor_extractor_type, Timing &timing) {
    Mat img;
    Mat descriptors;
    vector<KeyPoint> keypoints;
    vector<float> confidences;
    int ps;

    read_skel_file(skel_path, keypoint_size, keypoints, confidences, ps);

    assert(ps > 0);

    compute_descriptors(img_path, keypoints, descriptor_extractor_type, descriptors);

    // Search for a pose already inserted
    vector<int>::iterator iter = find(pose_number_.begin(), pose_number_.end(), ps);
    if (iter != pose_number_.end()) {
        // Pose present => update the model
        int pose_idx = static_cast<int>(iter - pose_number_.begin());

        pose_number_[pose_idx] = ps;
        pose_keypoints_[pose_idx] = keypoints;
        pose_descriptors_[pose_idx] = descriptors;
        pose_confidences_[pose_idx] = confidences;
    }
    else {
        // Pose not present
        pose_number_.push_back(ps);
        pose_keypoints_.push_back(keypoints);
        pose_descriptors_.push_back(descriptors);
        pose_confidences_.push_back(confidences);
    }
}

float
MultiviewBodyModel::match(const Mat &frame_desc, int frame_ps, const vector<float> &frame_conf, int norm_type,
                          bool occlusion_search, const Mat &poses_map, const Mat &kp_map, const Mat &kp_weights,
                          const Mat &ps2keypoints_map, Timing &timing) {
    assert(pose_descriptors_.size() > 0);
    assert(kp_map.cols == pose_descriptors_[0].rows && kp_weights.cols == pose_descriptors_[0].rows);
    assert(poses_map.rows >= pose_number_.size() && ps2keypoints_map.rows >= pose_number_.size() && ps2keypoints_map.rows == poses_map.rows);
    assert(kp_map.rows > 0 && kp_map.rows == kp_weights.rows);

    // Output distance
    double sum_dist = 0.0f;
    int sum_W = 0;

    // Search the pose
    vector<int>::iterator iter = find(pose_number_.begin(), pose_number_.end(), frame_ps);
    if (iter != pose_number_.end()) {
        // Pose found: get the index in the pose_number vector
        int ps_num_idx = static_cast<int>(iter - pose_number_.begin());

        // Foreach keypoint in the pose found
        for (int k = 0; k < pose_descriptors_[ps_num_idx].rows; ++k) {
            // Get the confidence of the current keypoint and check the occlusion between frame and
            // the model pose
            float *model_conf = &pose_confidences_[ps_num_idx][k];

            switch (check_occlusion(frame_conf[k], *model_conf)) {
                case VISIBLE:
                    sum_dist += norm(frame_desc.row(k), pose_descriptors_[ps_num_idx].row(k), norm_type);
                    sum_W++;
                    break;
                case MODELOCCLUDED:
                    if (occlusion_search) {
                        double weighted_dist = 0.0;
                        double overall_weights = 0.0;

                        occlusion_norm(frame_desc.row(k), frame_ps, k, poses_map, ps2keypoints_map, kp_map, kp_weights,
                                       norm_type, weighted_dist, overall_weights);

                        sum_dist += weighted_dist;
                        sum_W += overall_weights;
                    }
                    break;

                default:;
            }
        }
    }
    else {
        // Pose not found
        Mat keypoints_found;
        Mat model_descriptors;
        Mat model_weights;
        create_descriptor_from_poses(frame_ps, poses_map, ps2keypoints_map, kp_map, kp_weights, model_descriptors,
                                     model_weights, keypoints_found);

        for (int k = 0; k < model_descriptors.rows; ++k) {
            // Compute the distance only for valid keypoints and not occluded
            if (keypoints_found.row(k).at<uchar>(0) == 1 && frame_conf[k] == 1) {
                float W = model_weights.row(k).at<float>(0);
                sum_dist += W * cv::norm(frame_desc.row(k), model_descriptors.row(k), norm_type);
                sum_W += W;
            }
        }
    }

    if (sum_W != 0)
        return static_cast<float>(sum_dist / sum_W);

    return -1;
}



// Example: keypoint 10 of model_pose 2 is occluded
// alternative_pose -> look in pose_map[pose-1] for other poses (e.g. we chose [0,3,4])
// note: the pose is decremented by 1!
// kp_row_idx -> is the index in kp_map and kp_weights of the keypoints mapping
// [0    1    2    3    4   5   6   7    8    9   10   11  12  13  14] indices
// [0,  -1,   2,   3,   4, -1, -1, -1,   8,   9,  10,  11, -1, -1, -1] kp_map values
// [0.3, 0, 0.5, 0.5, 0.5,  0,  0,  0, 0.5, 0.5, 0.7, 0.7,  0,  0,  0] kp_weights values
//
// pose_idx -> contains the index of the alternative_pose in this model
// look the keypoint to match in kp_map for the keypoint 10: we have
// alternative_kp_idx = 10
// weight = 0.7
// store the descriptor with pose <alternative_pose> and <alternative_kp_idx> in model_descriptor
//
// compute the distance norm(frame_descriptor, model_descriptor) * weight
//
void MultiviewBodyModel::occlusion_norm(const cv::Mat &frame_desc, int model_ps, int model_kp_idx,
                                        const cv::Mat &model_poses_map, const cv::Mat &model_ps2keypoints_map,
                                        const cv::Mat &model_kp_map, const cv::Mat &model_kp_weights, int norm_type,
                                        double &out_weighted_distance, double &out_tot_weight) {
    // Output variables
    out_weighted_distance = 0.0;
    out_tot_weight = 0.0;

    // Necessary to be consistent with matrix indices
    model_ps--;
    // Foreach alternative pose in poses_map
    int i = 0;
    int alternative_pose = model_poses_map.at<int>(model_ps, 0);
    while(alternative_pose != -1) {
        // Get the set of keypoints mapping relative to this pose
        int kp_row_idx = model_ps2keypoints_map.at<int>(model_ps, i);

        // Search the pose inside the model
        vector<int>::iterator iter = find(pose_number_.begin(), pose_number_.end(), alternative_pose);

        if (iter != pose_number_.end()) {
            // The pose exists: get the index
            int pose_idx = static_cast<int>(iter - pose_number_.begin());

            // Get the index of the keypoint to match in this pose and the relative weight
            int alternative_kp_idx = model_kp_map.row(kp_row_idx).at<int>(model_kp_idx);
            float weight = model_kp_weights.row(kp_row_idx).at<float>(model_kp_idx);

            if (alternative_kp_idx != -1) {
                if (pose_confidences_[pose_idx][alternative_kp_idx] == 1) {
                    // An alternative and visible kp is found
                    Mat model_descriptor = pose_descriptors_[pose_idx].row(alternative_kp_idx);
                    out_weighted_distance += weight * norm(frame_desc, model_descriptor, norm_type);
                    out_tot_weight += weight;
                }
            }
        }

        // Next alternative pose
        alternative_pose = model_poses_map.at<int>(model_ps, ++i);
    }
}

void MultiviewBodyModel::create_descriptor_from_poses(int pose_not_found, const cv::Mat &model_poses_map,
                                                      const cv::Mat &model_ps2keypoints_map,
                                                      const cv::Mat &model_kp_map, const cv::Mat &model_kp_weights,
                                                      cv::Mat out_model_descriptors, cv::Mat out_descriptors_weights,
                                                      cv::Mat keypoints_mask) {

    assert(pose_descriptors_.size() > 0);

    // Coherent with matrix indices
    pose_not_found--;

    // Total number of keypoints
    int num_keypoints = pose_descriptors_[0].rows;

    // Output matrix initialization
    out_model_descriptors.create(num_keypoints, pose_descriptors_[0].cols, pose_descriptors_[0].type());
    out_descriptors_weights.create(num_keypoints, 1, CV_32F);
    keypoints_mask.create(num_keypoints, 1 , CV_8U);

    // Track the keypoint being added
    int k = 0;

    // Store information about the alternative pose being considered
    int alternative_ps_idx = 0;
    int alternative_ps = model_poses_map.row(pose_not_found).at<int>(alternative_ps_idx);

    // Build the descriptor
    while(k < num_keypoints && alternative_ps_idx < model_poses_map.cols && alternative_ps != -1) {
        // Search the alternative pose inside the model
        vector<int>::iterator iter = find(pose_number_.begin(), pose_number_.end(), alternative_ps);
        if (iter != pose_number_.end()) {
            int pose_number_idx = static_cast<int>(iter - pose_number_.begin());

            // Pose found: get the relative keypoints_mapping row
            int kp_row_idx = model_ps2keypoints_map.row(pose_not_found).at<int>(alternative_ps_idx);

            // Get the keypoint to match with
            int kp = model_kp_map.row(kp_row_idx).at<int>(k);

            // Checking if the keypoints has a valid match
            if (kp != -1) {
                if (pose_confidences_[pose_number_idx][kp] == 1) {
                    // Keypoint visible, append the relative descriptor:
                    // the k-th row of the output vector as the kp-th descriptor
                    // of pose_number_[pose_number_idx]
                    out_model_descriptors.row(k) = pose_descriptors_[pose_number_idx].row(kp);
                    float kp_weight = model_kp_weights.row(kp_row_idx).at<float>(k);
                    out_descriptors_weights.row(k).at<float>(0) = kp_weight;
                    alternative_ps = model_poses_map.row(pose_not_found).at<int>(0);
                    k++;
                    keypoints_mask.row(k).at<uchar>(0) = 1;
                }
            }
            else {
                // If there isn't a valid match keypoint, search another one in other alternative poses
                alternative_ps = model_poses_map.row(pose_not_found).at<int>(++alternative_ps_idx);
                if (alternative_ps == -1) {
                    // There are no more alternative poses to choose:
                    // not consider this keypoint for the match and go to the next keypoint
                    alternative_ps = 0;
                    keypoints_mask.row(k).at<uchar>(0) = 0;
                    k++;
                }
            }
        }
    } // end-while
}

OcclusionType MultiviewBodyModel::check_occlusion(float frame_conf, float model_conf) {
    if (frame_conf == 0 && model_conf == 0)
        return BOTHOCCLUDED;
    if (frame_conf == 1 && model_conf == 0)
        return MODELOCCLUDED;
    if (frame_conf == 0 && model_conf == 1)
        return  FRAMEOCCLUDED;
    return VISIBLE;
}



// ------------------------------------------------------------------------- //
//                           Function definitions                            //
// ------------------------------------------------------------------------- //

void read_skel_file(const string &skel_path, int keypoint_size,
                    vector<cv::KeyPoint> &out_keypoints, vector<float> &out_confidences, int &out_pose_side) {
    // File reading variables
    string line;
    std::ifstream file(skel_path);
    if (!file.is_open()) {
        std::cerr << "ReadAndCompute: " << skel_path << "Invalid file name." << std::endl;
        exit(-1);
    }

    // Read the file line by line
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
                        out_confidences.push_back(0);
                    else
                        out_confidences.push_back(conf);

                    // Reset to 0 for the next keypoint
                    value_type %= 2;
                    break;
            }
        }
        ++i;
    }
    // Last line contains the pose side
    std::stringstream ss(line);
    ss >> out_pose_side;
}


void Configuration::show() {
    cout << "---------------- CONFIGURATION --------------------------------" << endl << endl;
    cout << "MAIN PATH: " << main_path << endl;
    cout << "RESULTS PATH: " << res_file_path << endl;
    cout << "PERSONS NAMES: " << endl;
    cout << "[" << persons_names[0];
    for (int k = 1; k < persons_names.size(); ++k) {
        cout << ", " << persons_names[k];
        if (k % 2 == 0 && k < persons_names.size() - 1)
            cout << endl;
    }
    cout << "]" << endl;
    cout << "VIEWS NAMES: ";
    cout << "[" << views_names[0];
    for (int j = 1; j < views_names.size(); ++j) {
        cout << ", " << views_names[j];
    }
    cout << "]" << endl;
    cout << "NUMBER OF IMAGES: ";
    cout << "[" << (int)num_images.at<uchar>(0, 0);
    for (int i = 1; i < num_images.rows; i++) {
        cout << ", " << (int)num_images.at<uchar>(i, 0);
    }
    cout << "]" << endl << endl;
    cout << "DESCRIPTOR TYPE: " <<
            (descriptor_extractor_type.size() > 1 ? "all" : descriptor_extractor_type[0]) << endl;
    cout << "NORM_TYPE: " << (norm_type == cv::NORM_L2 ? "L2" : "Hamming") << endl;

    cout << "KEYPOINT SIZE: ";
    if (keypoint_size.size() > 1)
        cout << "predefined" << endl;
    else
        cout << keypoint_size[0] << endl;
    cout << "OCCLUSION SEARCH: " << (occlusion_search ? "T" : "F") << endl;
    cout << "MAX POSES: " << max_poses << endl;
    cout << "><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><" << endl << endl;
}

void show_help() {
    cout << "USAGE: multiviewbodymodel -c <configfile> -d <descriptortype> -k <keypointsize> -n <numberofposes>" << endl;
    cout << "EXAMPLE: $/multiviewbodymodel -c ../conf.xml -r ../res/ -d L2 -ps 3" << endl;
    cout << "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
         << endl;
    cout << "-c         path to the configuration file, it must be a valid .xml file." << endl;
    cout << "-r         path to the directory where the resutls are stored, must be already created." << endl;
    cout << "-d         descriptor to use: choose between SIFT, SURF, ORB, FREAK, BRIEF." << endl <<
            "           You must specify the keypoint size with -k flag." << endl <<
            "           If you want to compute the predefined descriptors choose:" << endl <<
            "             H for descriptor with Hamming distance (ORB, FREAK, BRIEF)" << endl <<
            "             L2 for descriptor with Euclidean distance (SIFT, SURF)" << endl <<
            "           then you don't need to specify the keypoint size." << endl;
    cout << "-k         set the keypoint size" << endl;
    cout << "-ps        set the number of pose sides" << endl;
    exit(0);
}

void parse_args(int argc, char **argv, Configuration &out_conf) {
    std::stringstream ss;

    // Default values
    out_conf.conf_file_path = "../conf.xml";
    out_conf.res_file_path = "../res/";


    for (int i = 1; i < argc; ++i) {
        if (i != argc) {
            if (strcmp(argv[i], "-c") == 0) {
                ss << argv[++i];
                out_conf.conf_file_path = ss.str();
                ss.str("");
            }
            if (strcmp(argv[i], "-r") == 0) {
                ss << argv[++i];
                out_conf.res_file_path = ss.str();
                ss.str("");
            }
            else if (strcmp(argv[i], "-d") == 0) {
                if (out_conf.keypoint_size.size() > 0)
                    out_conf.keypoint_size.clear();

                if (strcmp(argv[i+1], "L2") == 0) {
                    out_conf.descriptor_extractor_type.push_back("SURF");
                    out_conf.keypoint_size.push_back(11);
                    out_conf.descriptor_extractor_type.push_back("SIFT");
                    out_conf.keypoint_size.push_back(3);

                    out_conf.norm_type = cv::NORM_L2;
                }
                else if (strcmp(argv[i+1], "H") == 0) {
                    out_conf.descriptor_extractor_type.push_back("BRIEF");
                    out_conf.keypoint_size.push_back(11);
                    out_conf.descriptor_extractor_type.push_back("ORB");
                    out_conf.keypoint_size.push_back(9);
                    out_conf.descriptor_extractor_type.push_back("FREAK");
                    out_conf.keypoint_size.push_back(9);

                    out_conf.norm_type = cv::NORM_HAMMING;
                }
                else {
                    ss << argv[i+1];
                    out_conf.descriptor_extractor_type.push_back(ss.str());
                    out_conf.norm_type = get_norm_type(argv[++i]);
                    ss.str("");
                }
            }
            else if (strcmp(argv[i], "-k") == 0) {
                int value = atoi(argv[++i]);

                int size = out_conf.descriptor_extractor_type.size();
                if (size > 0) {
                    out_conf.keypoint_size.clear();
                    // Put the same keypoint size for all the descriptors
                    for (int j = 0; j < size; ++j) {
                        out_conf.keypoint_size.push_back(value);
                    }
                }
                else {
                    // Put only one keypoint size
                    out_conf.keypoint_size.push_back(value);
                }
            }
            else if (strcmp(argv[i], "-ps") == 0) {
                out_conf.max_poses = atoi(argv[++i]);
            }
        }
    }

    cv::FileStorage fs(out_conf.conf_file_path, cv::FileStorage::READ);
    fs["MainPath"] >> out_conf.main_path;

    cv::FileNode pn = fs["PersonNames"];
    check_sequence(pn);
    for (cv::FileNodeIterator it = pn.begin(); it != pn.end(); ++it)
        out_conf.persons_names.push_back((string)*it);

    cv::FileNode wn = fs["ViewNames"];
    check_sequence(wn);
    for (cv::FileNodeIterator it = wn.begin(); it != wn.end(); ++it)
        out_conf.views_names.push_back((string)*it);

    fs["KeypointsNumber"] >> out_conf.keypoints_number;

    fs["NumImages"] >> out_conf.num_images;

    if (out_conf.persons_names.size() != out_conf.num_images.rows) {
        cerr << "#persons != #num_images, check the configuration file!" << endl;
        exit(-1);
    }

    fs["OcclusionSearch"] >> out_conf.occlusion_search;
    fs.release();
}

int get_norm_type(const char *descriptor_name) {
    bool l2_cond = (strcmp(descriptor_name, "SIFT") == 0 || strcmp(descriptor_name, "SURF") == 0);
    bool h_cond = (strcmp(descriptor_name, "BRIEF") == 0 || strcmp(descriptor_name, "BRISK") == 0 ||
                   strcmp(descriptor_name, "ORB") == 0 || strcmp(descriptor_name, "FREAK") == 0);
    if (l2_cond)
        return cv::NORM_L2;
    else if (h_cond)
        return cv::NORM_HAMMING;

    return -1;
}

void check_sequence(cv::FileNode fn) {
    if(fn.type() != cv::FileNode::SEQ) {
        cerr << "Configuration file error: not a sequence." << endl;
        exit(-1);
    }
}

void load_person_imgs_paths(const Configuration &conf, vector<vector<string> > &out_imgs_paths, vector<vector<string> > &out_skels_paths) {

    assert(conf.persons_names.size() == conf.num_images.rows);

    std::stringstream ss_imgs, ss_skels;

    vector<string> imgs_paths;
    vector<string> skels_paths;
    for (int i = 0; i < conf.persons_names.size(); ++i) {

        for (int j = 0; j < conf.views_names.size(); ++j) {
            for (int k = 0; k <= conf.num_images.at<uchar>(i, 0); ++k) {
                if (k < 10) {
                    ss_imgs << conf.main_path << conf.persons_names[i] << "/"
                            << conf.views_names[j] << "0000" << k << ".png";
                    ss_skels << conf.main_path << conf.persons_names[i] << "/"
                             << conf.views_names[j] << "0000" << k << "_skel.txt";
                }
                else {
                    ss_imgs << conf.main_path << conf.persons_names[i] << "/"
                            << conf.views_names[j] << "000" << k << ".png";
                    ss_skels << conf.main_path << conf.persons_names[i] << "/"
                             << conf.views_names[j] << "000" << k << "_skel.txt";
                }

                imgs_paths.push_back(ss_imgs.str());
                skels_paths.push_back(ss_skels.str());

                ss_imgs.str("");
                ss_skels.str("");
            }
        }
        out_imgs_paths.push_back(imgs_paths);
        out_skels_paths.push_back(skels_paths);

        imgs_paths.clear();
        skels_paths.clear();
    }
}


template <typename T>
void load_masks(const vector<vector<T> > &skels_paths,
                vector<Mat> &masks) {
    for (int i = 0; i < skels_paths.size(); ++i) {
        Mat mask;
        mask = Mat::zeros(1, static_cast<int>(skels_paths[i].size()), CV_8S);
        masks.push_back(mask);
    }
}

template void load_masks<string>(const vector<vector<string> > &skels_paths,
                              vector<Mat> &masks);

template void load_masks<int>(const vector<vector<int> > &indices,
                                 vector<Mat> &masks);

int load_models_set(std::vector<int> &poses, std::vector<std::vector<cv::string> > img_paths,
                    std::vector<std::vector<cv::string> > skels_paths, int num_skel_keypoints, int max_size,
                    int min_keypoints_visibles, std::vector<std::vector<int> > &out_model_set) {

    assert(img_paths.size() == skels_paths.size());
    assert(poses.size() > 0);

    vector<Mat> masks;
    load_masks<string>(skels_paths, masks);

    // Computing the larger size
    int max_size_good = static_cast<int>(max_size / poses.size())
                        * static_cast<int>(poses.size());

    // Build the model set
    for (int i = 0; i < skels_paths.size(); ++i) {
        // Will contain all the  skeleton's indices for each person
        vector<int> indices;

        // Setting the initial pose
        int pose_idx = 0;
        vector<int>::iterator iter = find(poses.begin(), poses.end(), get_pose_side(skels_paths[i][0]));
        if (iter != poses.end()) {
            pose_idx = static_cast<int>(iter - poses.begin());
        }

        // Fill the indices vector
        int j = 0;
        while (indices.size() < max_size_good) {
            string *skel_path = &skels_paths[i][j];

            char *mask_elem = &masks[i].row(0).at<char>(j);

            if (get_pose_side(*skel_path) == poses[pose_idx] && *mask_elem != -1) {
                if (get_total_keyponts_visible(*skel_path, 15) > min_keypoints_visibles) {
                    indices.push_back(j);
                    pose_idx = ++pose_idx % static_cast<int>(poses.size());
                }
                else
                    *mask_elem = -1;
            }

            j++;

            // If scanning of skels is completed, start again from the beginning and
            // chose other files
            if (j == skels_paths[i].size())
                j = 0;
        }

        out_model_set.push_back(indices);
    }

    return max_size_good / static_cast<int>(poses.size());
}

int get_total_keyponts_visible(const string &skel_path, int num_keypoints) {
    // File reading variables
    string line;
    std::ifstream file(skel_path);
    if (!file.is_open()) {
        std::cerr << "get_total_keyponts_visible: " << skel_path << "Invalid file name." << std::endl;
        exit(-1);
    }

    int counter = 0;
    for (int i = 0; i < num_keypoints && getline(file, line); ++i) {
        vector<string> tokens;
        tokenize(line, ',', tokens);

        if (tokens[2] == "1")
            counter++;
    }

    return counter;
}

void tokenize(const std::string &line, char delim, vector<cv::string> &out_tokens) {
    istringstream iss(line);
    string field;
    while (std::getline(iss, field, delim)) {
        stringstream ss(field);
        out_tokens.push_back(ss.str());
    }
}

int get_pose_side(string path) {

    // Read the file
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "get_pose_side: " << path << "Invalid file name." << std::endl;
        exit(-1);
    }

    file.seekg(0, std::ios::end);
    int pos = file.tellg();
    pos-=2;
    file.seekg(pos);

    string line;
    getline(file, line);
    std::stringstream ss(line);


    int pose_side;
    ss >> pose_side;

    file.close();

    return pose_side;
}

void compute_descriptors(const std::string &img_path, const std::vector<cv::KeyPoint> &in_keypoints,
                         const cv::string &descriptor_extractor_type, cv::Mat &out_descriptors) {


    // Read the image file
    Mat img = cv::imread(img_path);
    if (!img.data) {
        cerr << "compute_descriptors(): Invalid image file." << endl;
        exit(-1);
    }

    // Required variables
    vector<cv::KeyPoint> tmp_keypoints(in_keypoints);
    cv::Mat tmp_descriptors;

    if (descriptor_extractor_type == "SIFT") {
        cv::SiftDescriptorExtractor sift_extractor(0, 3, 0.04, 15, 1.6);
        sift_extractor.compute(img, tmp_keypoints, tmp_descriptors);
    }
    else if (descriptor_extractor_type == "SURF") {
        cv::SurfDescriptorExtractor surf_extractor(0, 4, 2, true, true);
        surf_extractor.compute(img, tmp_keypoints, tmp_descriptors);
    }
    else if (descriptor_extractor_type == "BRIEF") {
        cv::BriefDescriptorExtractor brief_extractor(64);
        brief_extractor.compute(img, tmp_keypoints, tmp_descriptors);
    }
    else if (descriptor_extractor_type == "ORB") {
        cv::OrbDescriptorExtractor orb_extractor(0, 0, 0, 31, 0, 2, cv::ORB::FAST_SCORE, 31);
        orb_extractor.compute(img, tmp_keypoints, tmp_descriptors);
    }
    else if (descriptor_extractor_type == "FREAK") {
        cv::FREAK freak_extractor(true, true, 22.0f, 4, vector<int>());
        freak_extractor.compute(img, tmp_keypoints, tmp_descriptors);
    }

    // Once descriptors are computed, check if some keypoints are removed by the extractor algorithm
    Mat descriptors(static_cast<int>(in_keypoints.size()), tmp_descriptors.cols, tmp_descriptors.type());
    out_descriptors = descriptors;

    // For keypoints without a descriptor, use a row with all zeros
    Mat zero_row;
    zero_row = Mat::zeros(1, tmp_descriptors.cols, tmp_descriptors.type());

    // Check the output size
    if (tmp_keypoints.size() < in_keypoints.size()) {
        int k1 = 0;
        int k2 = 0;

        // out_descriptors
        while(k1 < tmp_keypoints.size() && k2 < in_keypoints.size()) {
            if (tmp_keypoints[k1].pt == in_keypoints[k2].pt) {
                out_descriptors.row(k2) = tmp_descriptors.row(k1);
                k1++;
                k2++;
            }
            else {
                out_descriptors.row(k2) = zero_row;
                k2++;
            }
        }
    }
    else {
        out_descriptors = tmp_descriptors;
    }
}

// Example:
// # img   |   pose
//   1     |     1
//   2     |     1
//   3     |     1
//   4     |     2
//   5     |     3
//   6     |     3
//   7     |     3
//   8     |     3
//   9     |     4
//   10    |     4
//
// produce [1 3 2 1 3 4 4 2]
void get_poses_map(vector<vector<string> > train_paths, vector<vector<int> > &out_map) {
    for (int i = 0; i < train_paths.size(); ++i) {
        vector<int> vec;
        int prev = get_pose_side(train_paths[i][0]);
        vec.push_back(prev);

        int counter = 1;
        for (int j = 1; j < train_paths[i].size(); ++j) {

            int cur = get_pose_side(train_paths[i][j]);
            if (cur == prev) {
                counter++;
            }
            else {
                // Change the current pose side and save the counter
                // the counter follow the pose side.
                vec.push_back(counter);
                vec.push_back(cur);
                prev = cur;
                counter = 1;
            }
        }
        vec.push_back(counter);
        out_map.push_back(vec);
    }
}

void multiviewbodymodel::Timing::write(string name) {
    cv::FileStorage fs(name + ".xml", cv::FileStorage::WRITE);
    if(fs.isOpened()) {
//        fs << "avgLoadingTrainingSet" << (t_tot_load_training_set / n_tot_load_training_set);
//        fs << "avgExctraction" << (t_tot_extraction / n_tot_extraction);
//        fs << "avgMatch" << (t_matching / n_matching);
//        fs << "avgRound" << (t_rounds / n_rounds);
//        fs << "avgOneModelLoading" << (t_tot_model_loading / n_tot_model_loading);
//        fs << "avgSkelLoading" << (t_tot_skel_loading / n_tot_skel_loading);
//        fs << "totExec" << t_tot_exec;
        fs.release();
    }
    else
        cerr << "Timing::write(): cannot open the file!" << endl;

}

void multiviewbodymodel::Timing::show() {
    cout << "----------------- PERFORMANCE -----------------" << endl;
//    cout << "avgLoadingTrainingSet " << (t_tot_load_training_set / n_rounds);
//    cout << "avgOneRound " << (t_rounds / n_rounds);
//    cout << "avgDescriptorsComputation " << (t_tot_extraction / n_tot_extraction);
//    cout << "avgOneModelLoading " << (t_tot_model_loading / n_tot_model_loading);
//    cout << "avgSkelLoading " << (t_tot_skel_loading / n_tot_skel_loading);
//    cout << "totMatching " << t_tot_exec;
    cout << "-----------------------------------------------" << endl;
}

template<typename T>
int get_rank_index(priority_queue<PQRank<T>, vector<PQRank<T> >, PQRank<T> > pq,
                   int test_class) {
    // Work on a copy
    priority_queue<PQRank<T>, vector<PQRank<T> >, PQRank<T> > scores(pq);

    // Searching for the element with the same class and get the rank
    for (int i = 0; i < pq.size(); i++) {
        if (scores.top().class_idx == test_class)
            return i;
        scores.pop();
    }
    return (int) (pq.size() - 1);
}

template int
get_rank_index<float>(priority_queue<PQRank<float>, vector<PQRank<float> >, PQRank<float> > pq,
                      int test_class);

int factorial(int n) {
    if (n == 0)
        return 1;
    return n * factorial(n - 1);
}

} // end namespace





