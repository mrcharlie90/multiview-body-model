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
    using cv::string;
    using cv::Mat;

    // ------------------------------------------------------------------------- //
    //                      MultiviewBodyModel methods definitions               //
    // ------------------------------------------------------------------------- //

    MultiviewBodyModel::MultiviewBodyModel(int max_poses) { max_poses_ = max_poses; }

    int MultiviewBodyModel::ReadAndCompute(const string &skel_path, const string &img_path,
                                           const string &descriptor_extractor_type,
                                           int keypoint_size, Timing &timing) {
        // Timing
        double ti_model = (timing.enabled) ? (double)cv::getTickCount() : 0;

        // Returning value
        int ret = 0;

        // Output variables
        vector<cv::KeyPoint> keypoints;
        vector<float> confidences;
        int pose_side;

        read_skel_file(skel_path, keypoint_size, keypoints, confidences, pose_side);

        assert(pose_side > 0);

        // Check if the pose value is valid
        if (pose_side > max_poses_)
            return -1;

        // Check if the pose already exists...
        vector<int>::iterator iter = find(pose_number_.begin(), pose_number_.end(), pose_side);

        // ...if not, fill the body model with data, otherwise do not consider this skeleton
        if (iter == pose_number_.end()) {
            // Read image
            cv::Mat img = cv::imread(img_path);
            if (!img.data) {
                cerr << "ReadAndCompute:" << img_path << "Invalid image file." << endl;
                exit(0);
            }

            // Store initial time for computing descriptors
            double ti_desc = (timing.enabled) ? (double)cv::getTickCount() : 0;

            cv::Mat descriptors;
            compute_descriptors(keypoints, img, descriptor_extractor_type, descriptors);

            if (timing.enabled) {
                timing.t_tot_extraction += ((double)cv::getTickCount() - ti_desc) / cv::getTickFrequency();
                timing.n_tot_extraction++;
            }

            // Load data into the model
            pose_images_.push_back(img);
            pose_keypoints_.push_back(keypoints);
            pose_number_.push_back(pose_side);
            pose_descriptors_.push_back(descriptors);
            pose_confidences_.push_back(confidences);

            // The model has loaded the descriptors successfully
            ret = 1;
        }

        if (timing.enabled) {
            timing.t_tot_model_loading += ((double)cv::getTickCount() - ti_model) / cv::getTickFrequency();
            timing.n_tot_model_loading++;
        }

        return ret;
    }

    int MultiviewBodyModel::Replace(const string &skel_path, const Mat &img, const string &descriptor_extractor_type,
                                    int keypoint_size) {
        // Returning value
        int ret = 0;

        // Required variables
        vector<cv::KeyPoint> keypoints;
        vector<float> confidences;
        int pose_side;

        read_skel_file(skel_path, keypoint_size, keypoints, confidences, pose_side);

        // Check if the pose value is valid
        if (pose_side > max_poses_)
            return -1;

        // Compute descriptors for this frame
        cv::Mat descriptors;
        compute_descriptors(keypoints, img, descriptor_extractor_type, descriptors);

        // Check if the pose already exists...
        vector<int>::iterator iter = find(pose_number_.begin(), pose_number_.end(), pose_side);
        int index = static_cast<int>(iter - pose_number_.begin());

        // ...if so, populate the body model with the data, otherwise discard the data
        if (iter != pose_number_.end()) {
            assert(img.data);

            // Update the model
            pose_images_[index] = img;
            pose_number_[index] = pose_side;
            pose_descriptors_[index] = descriptors;
            pose_keypoints_[index] = keypoints;
            pose_confidences_[index] = confidences;

            // The model has replaced the descriptors successfully
            ret = 1;
        }
        else {
            // Add frame information to the model
            pose_images_.push_back(img);
            pose_keypoints_.push_back(keypoints);
            pose_number_.push_back(pose_side);
            pose_descriptors_.push_back(descriptors);
            pose_confidences_.push_back(confidences);

            // The model has loaded the descriptors successfully
            ret = 0;
        }

        return ret;
    }

    float MultiviewBodyModel::Match(const cv::Mat &test_descriptors, const vector<float> &test_confidences,
                                    int test_pose_side, bool occlusion_search, int norm_type, Timing &timing) {
        // Check the model is ready
        assert(this->ready());

        // Timing
        double ti_match = (timing.enabled) ? (double)cv::getTickCount() : 0;

        // Search for the corresponding pose side in this model
        for (int i = 0; i < pose_number_.size(); ++i) {
            if (pose_number_[i] == test_pose_side) {
                // Do the match, consindering the keypoint occlusion (by using confidences)
                vector<char> confidence_mask;

                // Mask used for defining the operation between keypoints occluded
                create_confidence_mask(test_confidences, pose_confidences_[i], confidence_mask);

                // Compute the average of the euclidean distances obtained among keypoints
                float average_distance = 0.0;
                int descriptors_count = 0;
                for (int k = 0; k < pose_descriptors_[i].rows; ++k) {
                    // Getting the operation to perform by the confidence mask
                    char operation = confidence_mask[k];

                    // If the occlusion_search flag is true, look for a non-occluded descriptors
                    // in the other pose sides.
                    Mat descriptor_occluded;
                    double dist;
                    if (operation == 1 && occlusion_search && get_descriptor_occluded(k, descriptor_occluded)) {
                        // A descriptor is found, so compute the distance
                        dist = norm(test_descriptors.row(k), descriptor_occluded, norm_type);
                        average_distance += dist;
                        descriptors_count++;
                    }
                    else if (operation == 2) {
                        dist = norm(test_descriptors.row(k), pose_descriptors_[i].row(k), norm_type);
                        average_distance += dist;
                        descriptors_count++;
                    }
                }

                if (timing.enabled) {
                    timing.t_matching += ((double)cv::getTickCount() - ti_match) / cv::getTickFrequency();
                    timing.n_matching++;
                }

                return average_distance / descriptors_count;
            }
        }

        return -1;
    }

    bool MultiviewBodyModel::ready() { return pose_number_.size() == max_poses_; }

    void MultiviewBodyModel::create_confidence_mask(const vector<float> &test_confidences,
                                                    const vector<float> &train_confidences,
                                                    vector<char> &out_mask) {

        assert(test_confidences.size() == train_confidences.size());

        for (int k = 0; k < test_confidences.size(); ++k) {
            if (test_confidences[k] > 0 && train_confidences[k] == 0) {
                // Keypoint occluded in the training frame
                out_mask.push_back(1);
            }
            else if (test_confidences[k] > 0 && train_confidences[k] > 0) {
                // Both keypoints visible
                out_mask.push_back(2);
            }
            else {
                // Test-frame's keypoint occluded or both occluded: discard the keypoints
                out_mask.push_back(0);
            }
        }
    }


    bool MultiviewBodyModel::get_descriptor_occluded(int keypoint_index, Mat &descriptor_occluded) {
        // Find the first non-occluded descriptor in another pose
        for (int i = 0; i < pose_descriptors_.size(); ++i) {
            if (pose_confidences_[i][keypoint_index] > 0) {
                descriptor_occluded = pose_descriptors_[i].row(keypoint_index);
                return true;
            }
        }
        return false;
    }

    int MultiviewBodyModel::size() {
        return pose_number_.size();
    }

    int MultiviewBodyModel::get_size_of() {
        int usage  = 0;
        usage += sizeof(max_poses_);
        for (int i = 0; i < pose_number_.size(); ++i) {
            usage += sizeof(pose_number_[i]);
            usage += sizeof(pose_descriptors_[i]);
            usage += sizeof(pose_keypoints_[i]);
            usage += sizeof(pose_images_[i]);
            usage += sizeof(pose_confidences_[i]);
        }
        return usage;
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

    void load_train_paths(const string &main_path, const vector<string> &persons_names,
                          const vector <string> &views_names, const Mat &num_images,
                          vector<vector<string> > &out_skels_paths, vector<vector<string> > &out_imgs_paths) {

        assert(persons_names.size() == num_images.rows);

        std::stringstream ss_imgs, ss_skels;

        vector<string> imgs_paths;
        vector<string> skels_paths;
        for (int i = 0; i < persons_names.size(); ++i) {

            for (int j = 0; j < views_names.size(); ++j) {
                for (int k = 0; k <= num_images.at<uchar>(i, 0); ++k) {
                    if (k < 10) {
                        ss_imgs << main_path << persons_names[i] << "/" << views_names[j] << "0000" << k << ".png";
                        ss_skels << main_path << persons_names[i] << "/" << views_names[j] << "0000" << k << "_skel.txt";
                    }
                    else {
                        ss_imgs << main_path << persons_names[i] << "/" << views_names[j] << "000" << k << ".png";
                        ss_skels << main_path << persons_names[i] << "/" << views_names[j] << "000" << k << "_skel.txt";
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

    void load_train_paths(const Configuration &conf, vector<vector<string> > &out_skels_paths,
                          vector<vector<string> > &out_imgs_paths) {
        load_train_paths(conf.main_path, conf.persons_names, conf.views_names, conf.num_images, out_skels_paths, out_imgs_paths);
    }

    bool load_models(const vector<vector<string> > &train_skels_paths, const vector<vector<string> > &train_imgs_paths,
                         const string &descriptor_extractor_type, int keypoint_size, int max_poses, vector<cv::Mat> &masks,
                         vector<MultiviewBodyModel> &out_models, Timing &timing) {
        // Initialize time
        double t0 = timing.enabled ? static_cast<double>(cv::getTickCount()) : 0;

        // Checking dimensions
        assert(train_imgs_paths.size() == train_skels_paths.size());

        // train_skels_paths.size() = the number of people
        for (int i = 0; i < train_skels_paths.size(); ++i) {

            // Checking dimensions
            assert(train_imgs_paths[i].size() == train_skels_paths[i].size());

            MultiviewBodyModel body_model(max_poses);

            // Store the index of the vector with the maximum number of elements
            int max_size_idx = 0;
            for (int k = 1; k < train_imgs_paths.size(); ++k) {
                if (train_imgs_paths[max_size_idx].size() < train_imgs_paths[k].size())
                    max_size_idx = k;
            }

            int current_image = 0;
            // number of images inserted
            // NOTE: if the image was already chosen for the matching then masks[i].at(0, j) > 0 , otherwise 0
            int non_zero_counter = countNonZero(masks[i].row(0));

            // Accept poses only with this value in the mask's element
            // (images could be chosen several times, then value >= 1)
            int value = 0;

            // The loop continue until the maximum size mask has all elements greater than 0
            while (!body_model.ready() && non_zero_counter < train_imgs_paths[max_size_idx].size()) {

                // Storing the current mask's element pointer for later use
                char *mask_elem = &masks[i].row(0).at<char>(current_image);

                // Insert the pose if not present, and mark it as "chosen" in the mask
                if (*mask_elem == value && *mask_elem != -1) {
                    int result = body_model.ReadAndCompute(train_skels_paths[i][current_image], train_imgs_paths[i][current_image],
                                                          descriptor_extractor_type, keypoint_size, timing);
                    // Check the value returned
                    // -1 => the pose side is not valid -> mark the relative element -1
                    //  0 => the pose side already exists -> do nothing
                    //  1 => the pose side is successfully loaded -> increment the mask's element value
                    if (result == 1) {
                        (*mask_elem)++;
                        non_zero_counter++;
                    }
                    else if (result == -1) {
                        (*mask_elem) = -1;
                        non_zero_counter++;
                    }
                }
                ++current_image;

                // If the end of the list is reached, start from the beginning
                // In this way we assure that all images in the data set will be considered (even multiple times)
                if (current_image == train_imgs_paths[i].size()) {
                    current_image = 0;
                    value++;
                }
            }

            // If the model contains all poses then add it to the vector
            // otherwise one model is invalid, then exit.
            if (body_model.ready())
                out_models.push_back(body_model);
            else
                return false;
        }

        if (timing.enabled) {
            timing.t_tot_load_training_set += ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
            timing.n_tot_load_training_set++;
        }

        return true;
    }

    void load_test_skel(const string &skel_path, const string &img_path, const string &descriptor_extractor_type,
                        int keypoint_size, cv::Mat &out_image, vector<cv::KeyPoint> &out_keypoints,
                        vector<float> &out_confidences, cv::Mat &out_descriptors, int &out_pose_side, Timing &timing) {

        double t0_load_test = (timing.enabled) ? cv::getTickCount() : 0;

        read_skel_file(skel_path, keypoint_size, out_keypoints, out_confidences, out_pose_side);

        out_image = cv::imread(img_path);
        if (!out_image.data) {
            std::cerr << "load_test_skel: " << img_path << "Invalid image file." << std::endl;
        }

        compute_descriptors(out_keypoints, out_image, descriptor_extractor_type, out_descriptors);

        assert(out_keypoints.size() == out_descriptors.rows);
        assert(out_descriptors.rows == out_confidences.size());

        if (timing.enabled) {
            timing.t_tot_skel_loading += ((double)cv::getTickCount() - t0_load_test) / cv::getTickFrequency();
            timing.n_tot_skel_loading++;
        }
    }

    void compute_descriptors(const vector<cv::KeyPoint> &in_keypoints, const cv::Mat &image,
                             const string &descriptor_extractor_type, cv::Mat &out_descriptors) {
        // Required variables
        vector<cv::KeyPoint> tmp_keypoints(in_keypoints);
        cv::Mat tmp_descriptors;

        if (descriptor_extractor_type == "SIFT") {
            cv::SiftDescriptorExtractor sift_extractor(0, 3, 0.04, 15, 1.6);
            sift_extractor.compute(image, tmp_keypoints, tmp_descriptors);
        }
        else if (descriptor_extractor_type == "SURF") {
            cv::SurfDescriptorExtractor surf_extractor(0, 4, 2, true, true);
            surf_extractor.compute(image, tmp_keypoints, tmp_descriptors);
        }
        else if (descriptor_extractor_type == "BRIEF") {
            cv::BriefDescriptorExtractor brief_extractor(64);
            brief_extractor.compute(image, tmp_keypoints, tmp_descriptors);
        }
        else if (descriptor_extractor_type == "ORB") {
            cv::OrbDescriptorExtractor orb_extractor(0, 0, 0, 31, 0, 2, cv::ORB::FAST_SCORE, 31);
            orb_extractor.compute(image, tmp_keypoints, tmp_descriptors);
        }
        else if (descriptor_extractor_type == "FREAK") {
            cv::FREAK freak_extractor(true, true, 22.0f, 4, vector<int>());
            freak_extractor.compute(image, tmp_keypoints, tmp_descriptors);
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

    template<typename T> int get_rank_index(std::priority_queue<RankElement<T>, vector<RankElement<T> >, RankElement<T> > pq,
                                            int test_class) {
        // Work on a copy
        std::priority_queue<RankElement<T>, vector<RankElement<T> >, RankElement<T> > scores(pq);

        // Searching for the element with the same class and get the rank
        for (int i = 0; i < pq.size(); i++)
        {
            if (scores.top().classIdx == test_class)
                return i;
            scores.pop();
        }
        return (int) (pq.size() - 1);
    }

    template int get_rank_index<float>(std::priority_queue<RankElement<float>, vector<RankElement<float> >, RankElement<float> > pq,
                                       int test_class);

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

    // Example:
    // A map of one person : [1 3 2 1 3 4 4 2] => tot_images = sum of odd elements = 3+1+4+2 = 10 images
    // start from pose side 1
    // cur_idx = 0
    // choose a random value from 0 to 3 => i.e. rnd_value = 2
    // store (cur_idx + rnd_value = 2)
    // point cur_idx to the next set of images with a different pose side
    // cur_idx += map[next_odd] => cur_idx = 0 + 3 = 3
    // repeat for the next person
    int get_rnd_indices(vector <vector<int> > map, int max_poses, vector <vector<int> > &out_rnd_indices) {
        // Total number of indices produced
        int tot_ind = 0;

        for (int i = 0; i < map.size(); ++i) {
            vector<int> vec;
            int current_idx = 0;

            // Build a vector containing random valid indices pointing to
            // the related images in the dataset
            for (int j = 0; j <= (map[i].size() - 1) / 2; ++j) {

                // Valid pose side check
                if (map[i][2 * j] <= max_poses) {
                    // Random number between [0, map[][])
                    cv::RNG rng((uint64) cv::getTickCount());
                    int rnd_value = (int) rng.uniform(0., (double) (map[i][2 * j + 1]));

                    vec.push_back(current_idx + rnd_value);
                }

                // Point the current index to a new pose side
                // in the training set: this can be done by adding
                // the number of images with the current pose side (odd element of the map)
                current_idx += map[i][2 * j + 1];
            }

            tot_ind += vec.size();
            out_rnd_indices.push_back(vec);
        }

        return tot_ind;
    }

    void multiviewbodymodel::Timing::write(string name) {
        cv::FileStorage fs(name + ".xml", cv::FileStorage::WRITE);
        if(fs.isOpened()) {
            fs << "avgLoadingTrainingSet" << (t_tot_load_training_set / n_tot_load_training_set);
            fs << "avgExctraction" << (t_tot_extraction / n_tot_extraction);
            fs << "avgMatch" << (t_matching / n_matching);
            fs << "avgRound" << (t_rounds / n_rounds);
            fs << "avgOneModelLoading" << (t_tot_model_loading / n_tot_model_loading);
            fs << "avgSkelLoading" << (t_tot_skel_loading / n_tot_skel_loading);
            fs << "totExec" << t_tot_exec;
            fs.release();
        }
        else
            cerr << "Timing::write(): cannot open the file!" << endl;

    }

    void multiviewbodymodel::Timing::show() {
        cout << "----------------- PERFORMANCE -----------------" << endl;
        cout << "avgLoadingTrainingSet " << (t_tot_load_training_set / n_rounds);
        cout << "avgOneRound " << (t_rounds / n_rounds);
        cout << "avgDescriptorsComputation " << (t_tot_extraction / n_tot_extraction);
        cout << "avgOneModelLoading " << (t_tot_model_loading / n_tot_model_loading);
        cout << "avgSkelLoading " << (t_tot_skel_loading / n_tot_skel_loading);
        cout << "totMatching " << t_tot_exec;
        cout << "-----------------------------------------------" << endl;
    }

    void saveCMC(string path, Mat cmc) {
        assert(cmc.rows == 1);

        cout << "Saving CMC...";
        std::ofstream file(path);
        if (file.is_open()) {
            file << "coordinates {";
            for (int j = 0; j < cmc.cols; ++j) {
                file << "(" << j+1 << "," << cmc.at<float>(0, j) * 100 << ")";
            }
            file << "};";
            file.close();
        }
        else
            cerr << endl << "saveCMC(): Cannot open the file!" << endl;

        cout << "done!" << endl;
    }

    void cmc2dat(string path, cv::Mat cmc, float nAUC) {
        assert(cmc.rows == 1);

        cout << "Saving CMC...";
        std::ofstream file(path + ".dat");
        if (file.is_open()) {
            file << "rank     recrate" << endl;
            for (int j = 0; j < cmc.cols; ++j)
                file << j + 1 << "        " << cmc.at<float>(0, j) * 100 << endl;
        }
        else
            cerr << endl << "saveCMC(): Cannot open the file!" << endl;

        file.close();
        cout << "done!" << endl;
    }

    void rank1_append_results(string path, string desc_extractor, cv::Mat cmc) {
        std::ofstream file(path + ".dat", std::ofstream::app);
        if (file.is_open()) {
            file << desc_extractor << "   " << cmc.at<float>(0, 0) * 100 << endl;
        }
        else
            cerr << endl << "rank1_append_results(): Cannot open the file!" << endl;
        file.close();
    }

    void nauc_append_result(string path, string desc_extractor, float nauc) {
        std::ofstream file(path + ".dat", std::ofstream::app);
        if (file.is_open()) {
            file << desc_extractor << "   " << nauc * 100 << endl;
        }
        else
            cerr << endl << "nauc_append_results(): Cannot open the file!" << endl;
        file.close();
    }

    void print_dataset_usage(vector<cv::Mat> masks) {
        cout << "Dataset usage [%]: ";
        for (int i = 0; i < masks.size(); ++i) {
            cout << ((float)countNonZero(masks[i].row(0)) / (float)masks[i].cols) * 100 << "% ";
        }
        cout << endl;
    }

    void save_mask(string d_name, vector<cv::Mat> masks) {
        std::ofstream file("masks");

        file << "d_name" << endl;
        for (int i = 0; i < masks.size(); ++i) {
            assert(masks[i].rows == 1);

            file << masks[i] << endl;
        }
        file.close();
    }
}





