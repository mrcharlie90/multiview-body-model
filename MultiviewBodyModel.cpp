// Copyright (c) [2016] [Mauro Piazza]
//
//          IASLab License
//
// This file contains all methods deifinition of the MultiviewBodyModel class
// and the multiviewbodymodel namespace.
//

#include <opencv/cvaux.h>
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
using cv::FileNodeIterator;
using cv::getTickCount;
using cv::getTickFrequency;

// ------------------------------------------------------------------------- //
//                           Methods definitions                             //
// ------------------------------------------------------------------------- //


void MultiviewBodyModel::read_pose_compute_descriptors(string img_path, string skel_path, int keypoint_size,
                                                       string descriptor_extractor_type, Timing &timing) {
    Mat descriptors;
    vector<KeyPoint> keypoints;
    vector<float> confidences;
    int ps;

    read_skel_file(skel_path, keypoint_size, keypoints, confidences, ps);

    assert(ps > 0);

    float t0_extraction = timing.enabled ? getTickCount() : 0;

    // For each keypoint of the skeleton given, extract a descriptor
    compute_descriptors(img_path, keypoints, descriptor_extractor_type, descriptors);

    timing.enabled ? timing.extraction += (getTickCount() - t0_extraction) / getTickFrequency() : 0;
    timing.enabled ? timing.n_extraction++ : timing.n_extraction = 0;


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

OcclusionType MultiviewBodyModel::check_occlusion(float frame_conf, float model_conf) {
    if (frame_conf == 1 && model_conf == 0)
        return MODELOCCLUDED;
    if (frame_conf == 1 && model_conf == 1)
        return  VISIBLE;
    return OCCLUDED;
}

OcclusionType MultiviewBodyModel::check_occlusion(float model_conf) {
    if (model_conf == 1)
        return VISIBLE;
    return OCCLUDED;
}

float MultiviewBodyModel::find_min_match_distance(std::vector<cv::DMatch> matches) {
    float min = 1000;
    for (int j = 0; j < matches.size(); ++j) {
        float dist = matches[j].distance;
        if (dist < min)
            min = dist;
    }

    return min;
}

float MultiviewBodyModel::match(int frame_ps, const cv::Mat &frame_descriptors, const std::vector<float> &frame_conf,
                                const std::vector<std::vector<int> > &matching_poses_map,
                                const std::vector<cv::Mat> &matching_weights, int norm_type, bool occlusion_search,
                                Timing &timing) {

    assert(pose_descriptors_.size() > 0);
    assert(frame_ps > 0);

    float t0_match = timing.enabled ? getTickCount() : 0;

    float w_avg = 0.0;
    float sum_w = 0.0;
    vector<int>::iterator iter = find(pose_number_.begin(), pose_number_.end(), frame_ps);
    if (iter != pose_number_.end()) {
        // Pose found
        int pose_num_idx = static_cast<int>(iter - pose_number_.begin());
        Mat model_descriptors = pose_descriptors_[pose_num_idx].clone();

        assert(frame_descriptors.rows == model_descriptors.rows);

        int matching_idx = get_matching_index(frame_ps);

        // Foreach descriptor
        for (int i = 0; i < frame_descriptors.rows; ++i) {
            OcclusionType otype = check_occlusion(frame_conf[i], pose_confidences_[pose_num_idx][i]);

            compute_weighted_distance(frame_ps, frame_descriptors, model_descriptors, i, matching_idx,
                                      matching_poses_map, matching_weights, norm_type, otype,
                                      false, w_avg, sum_w);
        }

        w_avg /= sum_w;

        // If the pose is found compute also the brute force matching among keypoints
        cv::BFMatcher matcher(norm_type, true);
        vector<cv::DMatch> matches;
        matcher.match(frame_descriptors, model_descriptors, matches);

        float min_match_dist = 2 * find_min_match_distance(matches);

        for (int j = 0; j < matches.size(); ++j) {
            if (matches[j].queryIdx == matches[j].trainIdx && matches[j].distance < min_match_dist) {
                float w = matching_weights[matching_idx].row(matches[j].queryIdx).at<float>(0);
                w_avg -= w*10;
            }

        }
    }
    else {
        // Pose not found
        vector<int> other_poses = matching_poses_map[frame_ps - 1];
        bool computed = false;
        // Foreach other poses
        for (int i = 0; i < other_poses.size() && !computed; ++i) {
            // Find other pose
            vector<int>::iterator iter2 = find(pose_number_.begin(), pose_number_.end(), other_poses[i]);

            if (iter2 != pose_number_.end()) {
                // Other pose found
                int other_ps_num_idx = static_cast<int>(iter2 - pose_number_.begin());
                Mat model_descriptors = pose_descriptors_[other_ps_num_idx];

                assert(frame_descriptors.rows == model_descriptors.rows);

                // Foreach descriptor
                for (int j = 0; j < frame_descriptors.rows; ++j) {
                    OcclusionType otype = check_occlusion(frame_conf[j], pose_confidences_[other_ps_num_idx][j]);

                    // Get the index to the weights map
                    int matching_idx = get_matching_index(frame_ps, other_poses[i]);
                    compute_weighted_distance(frame_ps, frame_descriptors, model_descriptors, j, matching_idx,
                                              matching_poses_map, matching_weights, norm_type, otype,
                                              occlusion_search, w_avg, sum_w);
                }

                w_avg /= sum_w;
                computed = true;
            }
        }
    }

    timing.enabled ? timing.one_match += (getTickCount() - t0_match) / getTickFrequency() : 0;
    timing.enabled ? timing.n_one_match++ : timing.n_one_match = 0;

    return w_avg;
}

void
MultiviewBodyModel::compute_weighted_distance(int frame_pose, const Mat &frame_descriptors,
                                              const Mat &model_descriptors, int kp_idx, int matching_idx,
                                              const std::vector<std::vector<int> > &matching_poses_map,
                                              const std::vector<cv::Mat> &matching_weights, int norm_type,
                                              OcclusionType occlusion_type, bool occlusion_search, float &w_avg,
                                              float &sum_w) {
    // Distinguish keypoint visible and keypoint occluded
    if (occlusion_type == VISIBLE) {
        // Compute the distance
        float w = matching_weights[matching_idx].row(kp_idx).at<float>(0);
        w_avg += w * norm(frame_descriptors.row(kp_idx), model_descriptors.row(kp_idx), norm_type);
        sum_w += w;
    } else if (occlusion_type == MODELOCCLUDED && occlusion_search) {
        // Find another pose in the  mathing_pose_map vector and then compute the distance if the keypoint
        // is visible
        vector<int> other_poses = matching_poses_map[frame_pose - 1];

        bool found = false;
        for (int j = 0; j < other_poses.size() && !found; ++j) {
            vector<int>::iterator iter2 = find(pose_number_.begin(), pose_number_.end(), other_poses[j]);
            if (iter2 != pose_number_.end()) {
                // Other pose found
                int other_ps_num_idx = static_cast<int>(iter2 - pose_number_.begin());

                if (check_occlusion(pose_confidences_[other_ps_num_idx][j]) == VISIBLE) {
                    int w_idx2 = get_matching_index(frame_pose, other_poses[j]);
                    float w_found = matching_weights[w_idx2].row(kp_idx).at<float>(0);
                    w_avg += w_found * norm(frame_descriptors.row(kp_idx), pose_descriptors_[other_ps_num_idx].row(kp_idx), norm_type);
                    sum_w += w_found;
                    found = true;
                }
            }
        }
    }
}


float MultiviewBodyModel::match(Configuration &conf, const cv::Mat &frame_descriptors, int frame_ps,
                                const std::vector<float> &frame_conf, Timing &timing) {
    return match(frame_ps, frame_descriptors, frame_conf,
                 conf.matching_poses_map, conf.matching_weights, conf.norm_type,
                 conf.occlusion_search, timing);
}

int MultiviewBodyModel::get_matching_index(int ps_frame, int ps_model) {
    assert(ps_frame != 0);
    if(ps_model == 0 || (ps_frame == ps_model)) {
        // 1vs1 2vs2 3vs3 4vs4
        return ps_frame - 1;
    }
    int ret = 0;
    switch (ps_frame + ps_model) {
        case 3:
            // 1vs2 or 2vs1
            ret = 4;
            break;
        case 4:
            // 1vs3 or 3vs1
            ret = 5;
            break;
        case 5:
            if (ps_model == 1 || ps_frame == 1)
                // 1vs4 or 4vs1
                ret = 6;
            else
                // 3vs2 or 2vs3
                ret = 7;
            break;
        case 6:
            // 4vs2 or 2vs4
            ret = 8;
            break;
        default:
            // 3vs4 or 4vs3
            ret = 9;
            break;
    }
    return ret;
}

// ------------------------------------------------------------------------- //
//                           Function definitions                            //
// ------------------------------------------------------------------------- //

void multiviewbodymodel::Timing::write(string path, string name) {
    assert(enabled);

    std::ofstream file(path + name + ".dat", std::ofstream::app);
    if (file.is_open()) {
        if (file.tellp() == 0)
            file << "name   one_match   extraction    models_loading   descr_creation" << endl;

        file << name << " "
             << one_match / n_one_match << " "
             << extraction  / n_extraction << " "
             << models_loading / n_models_loading << " "
             << descr_creation / n_descr_creation << endl;
    }
    else
        cerr << endl << "Timing::write(): Cannot open the file!" << endl;
    file.close();
}

void multiviewbodymodel::Timing::show() {
    cout << endl << "----------------- PERFORMANCE -----------------" << endl;
    cout << "NAME: " << name << endl;
    cout << "ONE_MATCH: " << one_match / n_one_match << " s" << endl;
    cout << "EXTRACTION: " << extraction  / n_extraction << " s" << endl;
    cout << "MODELS_LOADING: " << models_loading / n_models_loading << " s" << endl;
    cout << "DESCRIPTOR_CONSTRUCTION: " << descr_creation / n_descr_creation << " s" << endl;
    cout << "-----------------------------------------------" << " s" << endl;
}

void Configuration::show() {
    cout << endl <<  "---------------- CONFIGURATION --------------------------------" << endl << endl;
    cout << "DS PATH: " << dataset_path << endl;
    cout << "RESULTS PATH: " << res_file_path << endl << endl;

    cout << "PERSONS NAMES: " << endl;
    cout << "[";
    for (int k = 0; k < persons_names.size(); ++k) {
        if (k != persons_names.size() - 1)
            cout << persons_names[k] << ", ";
        else
            cout << persons_names[k];
        if (k != 0 && k % 3 == 0 && k < persons_names.size() - 1)
            cout << endl;
    }
    cout << "]" << endl << endl;

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

    cout << "MODEL SET SIZE: " << model_set_size << endl;
    cout << "NUMBER OF KEYPOINTS: " << keypoints_number << endl;
    cout << "OCCLUSION SEARCH: " << (occlusion_search ? "T" : "F") << endl << endl;

    cout << "POSES NUMBERS: ";
    cout << "[" << poses[0];
    for (int k = 1; k < poses.size(); ++k) {
        cout << ", " << poses[k];
    }
    cout << "]" << endl;

    cout << "DESCRIPTOR TYPE: " << descriptor_type_str <<  endl;
    cout << "NORM_TYPE: " << (norm_type == cv::NORM_L2 ? "L2" : "Hamming") << endl;
    cout << "KEYPOINT SIZE: " << keypoint_size << endl;
    cout << "><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><" << endl << endl;
}

void show_help() {
    cout << "USAGE: multiviewbodymodel -c <configfile> -d <descriptortype> -k <keypointsize> -n <numberofposes>" << endl;
    cout << "EXAMPLE: $/multiviewbodymodel -c ../conf.xml -r ../res/ -d L2 -ps 3" << endl;
    cout << "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>" << endl;
    cout << "-c         path to the configuration file, it must be a valid .xml file." << endl;
    cout << "-r         path to the directory where the resutls are stored, must be already created." << endl;
    cout << "-d         descriptor to use: choose between SIFT, SURF, ORB, FREAK, BRIEF." << endl <<
            "           You must specify the keypoint size with -k flag." << endl <<
            "           If you want to compute the predefined descriptors choose:" << endl <<
            "             H for descriptor with Hamming distance (ORB, FREAK, BRIEF)" << endl <<
            "             L2 for descriptor with Euclidean distance (SIFT, SURF)" << endl <<
            "           then you don't need to specify the keypoint size." << endl;
    cout << "-k         set the keypoint size" << endl;
    cout << "-ms        set the model set size: number of images that will be  used to  build the " << endl
         << "           models." << endl;
    cout << "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>" << endl;
    exit(0);
}

void parse_args(int argc, char **argv, Configuration &out_conf) {

    if (argc < 6) {
        show_help();
    }

    std::stringstream ss;

    // Default values
    out_conf.conf_file_path = "../conf.xml";
    out_conf.res_file_path = "../res/";



    // Parse args
    for (int i = 1; i < argc; ++i) {
        if (i != argc) {
            if (strcmp(argv[i], "-c") == 0) {
                ss << argv[++i];
                out_conf.conf_file_path = ss.str();
                ss.str("");
            }
            else if (strcmp(argv[i], "-r") == 0) {
                ss << argv[++i];
                out_conf.res_file_path = ss.str();
                ss.str("");
            }
            else if (strcmp(argv[i], "-d") == 0) {
                ss << argv[i+1];
                out_conf.descriptor_type = char2descriptor_type(argv[i+1]);
                switch (out_conf.descriptor_type) {
                    case SIFT:
                        out_conf.keypoint_size = 3;
                        break;
                    case SURF:
                        out_conf.keypoint_size = 11;
                        break;
                    case BRIEF:
                        out_conf.keypoint_size = 11;
                        break;
                    case ORB:
                        out_conf.keypoint_size = 9;
                        break;
                    case FREAK:
                        out_conf.keypoint_size = 9;
                        break;
                    default:
                        cerr << "Invalid descriptor extractor type, "
                                "choose one from SIFT, SURF, ORB, FREAK, BRIEF." << endl;
                        exit(-1);
                }
                out_conf.norm_type = get_norm_type(out_conf.descriptor_type);
                out_conf.descriptor_type_str = ss.str();
                ss.str("");
            }
            else if (strcmp(argv[i], "-k") == 0) {
                int value = atoi(argv[++i]);
                out_conf.keypoint_size = value;
            }
            else if (strcmp(argv[i], "-ms") == 0) {
                int value = atoi(argv[++i]);
                out_conf.model_set_size = value;
            }
        }
    }
}

void read_config_file(Configuration &conf) {

    assert(!conf.conf_file_path.empty());

    cv::FileStorage fs(conf.conf_file_path, cv::FileStorage::READ);
    fs["DatasetPath"] >> conf.dataset_path;

    fs["MapFilePath"] >> conf.map_file_path;

    cv::FileNode pn = fs["PersonNames"];
    check_sequence(pn);
    for (FileNodeIterator it = pn.begin(); it != pn.end(); ++it)
        conf.persons_names.push_back((string)*it);

    cv::FileNode wn = fs["ViewNames"];
    check_sequence(wn);
    for (FileNodeIterator it = wn.begin(); it != wn.end(); ++it)
        conf.views_names.push_back((string)*it);

    cv::FileNode psn = fs["Poses"];
    for (FileNodeIterator it = psn.begin(); it != psn.end(); ++it)
        conf.poses.push_back((int)*it);
    if (conf.poses.size() == 0) {
        cerr << "Configuration error: you must insert at least one pose." << endl;
        exit(-1);
    }

    fs["KeypointsNumber"] >> conf.keypoints_number;

    fs["NumImages"] >> conf.num_images;

    if (conf.persons_names.size() != conf.num_images.rows) {
        cerr << "#persons != #num_images, check the configuration file!" << endl;
        exit(-1);
    }

    fs["OcclusionSearch"] >> conf.occlusion_search;
    fs.release();


    // Reading the matching maps
    fs.open(conf.map_file_path, cv::FileStorage::READ);

    Mat poses_map;
    fs["PosesMap"] >> poses_map;
    if (!poses_map.data) {
        int poses_map_data[12] = {2, 3, 4, // pose1
                                   1, 3, 4, // pose2
                                   1, 2, 4, // pose3
                                   1, 2, 3}; // pose4

        poses_map = Mat(4, 3, CV_16S, poses_map_data);
    }
    for (int i = 0; i < poses_map.rows; ++i) {
        vector<int> vec;
        for (int j = 0; j < poses_map.cols; ++j) {
            vec.push_back(poses_map.at<short>(i, j));
        }
        conf.matching_poses_map.push_back(vec);
    }


    stringstream ss;
    for (int i = 1; i <= 4; ++i) {
        for (int j = i; j <= 4; ++j) {
            Mat weight_mat;
            ss << "w" << i << "vs" << j;
            fs[ss.str()] >> weight_mat;
            if (!weight_mat.data) {
                cerr << "Configuration error: weight matrix not found." << endl;
                exit(-1);
            }
            conf.matching_weights.push_back(weight_mat);
            ss.str("");
        }
    }

    fs.release();
}

DescriptorType char2descriptor_type(const char *str) {
    if (strcmp(str, "SIFT") == 0)
        return SIFT;
    else if (strcmp(str, "SURF") == 0)
        return SURF;
    else if (strcmp(str, "BRIEF") == 0)
        return BRIEF;
    else if (strcmp(str, "ORB") == 0)
        return ORB;
    else if (strcmp(str, "FREAK") == 0)
        return FREAK;

    return INVALID;
}

int get_norm_type(DescriptorType descriptor_type) {

    int norm_type = -1;
    switch (descriptor_type) {
        case SIFT:
        case SURF:
            norm_type = cv::NORM_L2;
            break;
        case ORB:
        case FREAK:
        case BRIEF:
            norm_type = cv::NORM_HAMMING;
            break;
        default:;
    }

    return norm_type;
}

void check_sequence(cv::FileNode fn) {
    if(fn.type() != cv::FileNode::SEQ) {
        cerr << "Configuration file error: not a sequence." << endl;
        exit(-1);
    }
}

int load_person_imgs_paths(const Configuration &conf, vector<vector<string> > &out_imgs_paths,
                           vector<vector<string> > &out_skels_paths) {

    assert(conf.persons_names.size() == conf.num_images.rows);

    std::stringstream ss_imgs, ss_skels;

    vector<string> imgs_paths;
    vector<string> skels_paths;

    int tot_imgs = 0;
    for (int i = 0; i < conf.persons_names.size(); ++i) {

        for (int j = 0; j < conf.views_names.size(); ++j) {
            for (int k = 0; k <= conf.num_images.at<uchar>(i, 0); ++k) {
                if (k < 10) {
                    ss_imgs << conf.dataset_path << conf.persons_names[i] << "/"
                            << conf.views_names[j] << "0000" << k << ".png";
                    ss_skels << conf.dataset_path << conf.persons_names[i] << "/"
                             << conf.views_names[j] << "0000" << k << "_skel.txt";
                }
                else {
                    ss_imgs << conf.dataset_path << conf.persons_names[i] << "/"
                            << conf.views_names[j] << "000" << k << ".png";
                    ss_skels << conf.dataset_path << conf.persons_names[i] << "/"
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

        tot_imgs += imgs_paths.size();

        imgs_paths.clear();
        skels_paths.clear();
    }

    return tot_imgs;
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

int load_models_set(vector<int> &poses, const std::vector<std::vector<cv::string> > img_paths,
                    const std::vector<std::vector<cv::string> > skels_paths, int max_size,
                    int min_keypoints_visibles, std::vector<std::vector<int> > &out_model_set) {

    assert(img_paths.size() == skels_paths.size());
    assert(poses.size() > 0);

    vector<Mat> masks;
    load_masks<string>(skels_paths, masks);

    // Computing the larger size
    int max_size_good = static_cast<int>(max_size / poses.size())
                        * static_cast<int>(poses.size());

    // TODO: provare a selezionare i frame con area del convex hull maggiore
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
            string skel_path = skels_paths[i][j];

            char *mask_elem = &masks[i].row(0).at<char>(j);

            if (get_pose_side(skel_path) == poses[pose_idx] && *mask_elem != -1) {
                if (get_total_keyponts_visible(skel_path, 15) > min_keypoints_visibles) {
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
    int pos = static_cast<int>(file.tellg());
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

void read_skel_file(const string &skel_path, int keypoint_size, vector<cv::KeyPoint> &out_keypoints,
                    vector<float> &out_confidences, int &out_pose_side) {
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
                {
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

                default:;
            }
        }
        ++i;
    }
    // Last line contains the pose side
    std::stringstream ss(line);
    ss >> out_pose_side;
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
template<typename T>
int get_rank_index(priority_queue<PQRank<T>, vector<PQRank<T> >, PQRank<T> > pq,
                   int ground_truth) {
    // Work on a copy
    priority_queue<PQRank<T>, vector<PQRank<T> >, PQRank<T> > scores(pq);

    // Searching for the element with the same class and get the rank
    for (int i = 0; i < pq.size(); i++) {
        if (scores.top().class_idx == ground_truth)
            return i;
        scores.pop();
    }
    return (int) (pq.size() - 1);
}

template int get_rank_index<float>(priority_queue<PQRank<float>, vector<PQRank<float> >, PQRank<float> > pq,
                      int ground_truth);

void empty_models(int size, std::vector<MultiviewBodyModel> &models) {
    for (int i = 0; i < size; ++i) {
        MultiviewBodyModel model;
        models.push_back(model);
    }
}

void init_models(Configuration conf, int rounds, const std::vector<std::vector<cv::string> > imgs_paths,
                 const std::vector<std::vector<cv::string> > skels_paths,
                 const std::vector<std::vector<int> > &models_set,
                 std::vector<MultiviewBodyModel> &models, Timing &timing) {

    float t0_models_loading = timing.enabled ? (float)getTickCount() : 0;
    for (int i = 0; i < models_set.size(); ++i) {
        for (int pose_idx = 0; pose_idx < conf.poses.size(); ++pose_idx) {
            int frame_idx = pose_idx * rounds;
            models[i].read_pose_compute_descriptors(imgs_paths[i][models_set[i][frame_idx]],
                                                    skels_paths[i][models_set[i][frame_idx]],
                                                    conf.keypoint_size, conf.descriptor_type_str, timing);

        }
    }
    timing.enabled ? timing.models_loading += ((float)getTickCount() - t0_models_loading) / getTickFrequency() : 0;
    timing.enabled ? timing.n_models_loading++ : timing.n_models_loading = 0;
}

float area_under_curve(cv::Mat CMC) {
    assert(CMC.rows == 1);

    float nAUC = 0;
    for (int c  = 0; c < CMC.cols - 1; ++c) {
        nAUC += (CMC.at<float>(0, c) + CMC.at<float>(0, c + 1));
    }
    nAUC /= 2;
    nAUC /= CMC.cols; // normalize

    return nAUC;
}

string get_res_filename(Configuration conf) {
    stringstream ss;
    ss << "_O" << conf.occlusion_search
       << "_K" << conf.keypoint_size
       << "_MS" << conf.model_set_size
       << "_P" << conf.poses.size();

    return ss.str();
}

void write_cmc_nauc(cv::string path, string settings, cv::string desc_type, Mat CMC, float nAUC) {
    std::ofstream file(path + "CMC_" + desc_type + settings + ".dat", std::ofstream::out);
    if (file.is_open()) {
        file << "rank   recrate" << endl;

        for (int i = 0; i < CMC.cols; i++) {
            file << i+1 << " " << CMC.at<float>(0, i) << endl;
        }
    }
    else
        cerr << endl << "write_cmc_nauc(): Cannot open the file!" << endl;
    file.close();

    file.open(path + "nAU" + settings + ".dat", std::ofstream::app);
    if (file.is_open()) {
        if (file.tellp() == 0)
            file << "name   nauc" << endl;
        file << desc_type << " " << nAUC << endl;
    }
    else
        cerr << endl << "write_cmc_nauc(): Cannot open the file!" << endl;
    file.close();
}


} // end namespace





