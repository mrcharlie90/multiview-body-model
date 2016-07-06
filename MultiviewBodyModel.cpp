// Copyright (c) [2016] [Mauro Piazza]
// 
//          IASLab License
// 
// This file contains all  methods deifinition of the MultiviewBodyModel class
// and the multiviewbodymodel namespace.

#include "MultiviewBodyModel.h"

namespace multiviewbodymodel {
    using std::vector;
    using std::cout;
    using std::endl;
    using std::cerr;
    using cv::string;
    using cv::Mat;

    // -------------------------------------------------------------------------
    //                      MultiviewBodyModel methods definitions
    // -------------------------------------------------------------------------

    MultiviewBodyModel::MultiviewBodyModel(int max_poses) { max_poses_ = max_poses; }

    int MultiviewBodyModel::ReadAndCompute(string path, string img_path, string descriptor_extractor_type,
                                           int keypoint_size, Timing &timing) {
        // Timing
        double ti_model = (timing.enabled) ? (double)cv::getTickCount() : 0;

        // Returning value
        int ret = 0;

        // Output variables
        vector<cv::KeyPoint> keypoints;
        vector<float> confidences;
        int pose_side;

        // File reading variables
        string line;
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "ReadAndCompute: " << path << "Invalid file name." << std::endl;
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

        // Check the pose value is valid
        if (pose_side > max_poses_)
            return -1;

        // Check if the pose already exists...
        vector<int>::iterator iter = find(pose_side_.begin(), pose_side_.end(), pose_side);

        // ...if so, populate the body model with the data, otherwise discard the data
        if (iter == pose_side_.end()) {
            // Read image
            cv::Mat img = cv::imread(img_path);
            if (!img.data) {
                std::cerr << "ReadAndCompute:" << img_path << "Invalid image file." << std::endl;
                exit(0);
            }
            views_images_.push_back(img);

            // Compute descriptors for this view
            double ti_desc = (timing.enabled) ? (double)cv::getTickCount() : 0;

            cv::Mat descriptors;
            cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = cv::DescriptorExtractor::create(descriptor_extractor_type);
            descriptor_extractor->compute(img, keypoints, descriptors);

            if (timing.enabled) {
                timing.t_tot_descriptors += ((double)cv::getTickCount() - ti_desc) / cv::getTickFrequency();
                timing.n_tot_descriptors++;
            }

            pose_side_.push_back(pose_side);
            views_descriptors_.push_back(descriptors);
            views_descriptors_confidences_.push_back(confidences);

            // The model has successfully loaded the descriptors
            ret = 1;
        }
        if (timing.enabled) {
            timing.t_tot_model_loading += ((double)cv::getTickCount() - ti_model) / cv::getTickFrequency();
            timing.n_tot_model_loading++;
        }

        return ret;
    }

    float MultiviewBodyModel::Match(Mat query_descriptors, vector<float> query_confidences,
                                    int query_pose_side, bool occlusion_search) {
        // Checking the model is ready
        assert(this->ready());
        assert(query_confidences.size() == query_descriptors.rows);

        // Search for the corresponding pose side
        for (int i = 0; i < pose_side_.size(); ++i) {
            if (pose_side_[i] == query_pose_side) {
                // Do the match, consindering the keypoint occlusion (by using confidences)
                vector<char> confidence_mask;

                // Mask used for defining the operation between two keypoints occluded or semi-occluded.
                create_confidence_mask(query_confidences, views_descriptors_confidences_[i], confidence_mask);

                // Compute the average of the euclidean distances obtained from each keypoint
                float average_distance = 0.0;
                int descriptors_count = 0;
                for (int k = 0; k < views_descriptors_[i].rows; ++k) {
                    char operation = confidence_mask[k];

                    // If the occlusion_search flag is true, look
                    Mat descriptor_occluded;
                    double dist;
                    if (operation == 1 && occlusion_search &&
                        get_descriptor_occluded(k, descriptor_occluded)) {


                        // A descriptor is found, so compute the distance
                        dist = norm(query_descriptors.row(k), descriptor_occluded);
                        average_distance += dist;
                        descriptors_count++;
                    }
                    else if (operation == 2) {
                        dist = norm(query_descriptors.row(k), views_descriptors_[i].row(k));
                        average_distance += dist;
                        descriptors_count++;
                    }
                }
                return average_distance / descriptors_count;
            }
        }

        return -1;
    }

    bool MultiviewBodyModel::ready() {
        return (pose_side_.size() == max_poses_);
    }

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


    bool MultiviewBodyModel::get_descriptor_occluded(int keypoint_index, Mat &descriptor_occluded) {
        // Find a non-occluded descriptor in one pose
        for (int i = 0; i < views_descriptors_.size(); ++i) {
            if (views_descriptors_confidences_[i][keypoint_index] > 0) {
                descriptor_occluded = views_descriptors_[i].row(keypoint_index);
                return true;
            }
        }
        return false;
    }


    // -------------------------------------------------------------------------
    //                      Main function definitions
    // -------------------------------------------------------------------------

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
        cout << "MAX POSES: " << max_poses << endl;
        cout << "NUMBER OF IMAGES: ";
        cout << "[" << (int)num_images.at<uchar>(0, 0);
        for (int i = 1; i < num_images.rows; i++) {
            cout << ", " << (int)num_images.at<uchar>(i, 0);
        }
        cout << "]" << endl << endl;
        cout << "><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><" << endl << endl;
    }

    void parse_args(int argc, char **argv, Configuration &out_conf) {
        std::stringstream ss;

        for (int i = 1; i < argc; ++i) {
            if (i + 1 != argc) {
                if (strcmp(argv[i], "-c") == 0) {
                    ss << argv[++i];
                    out_conf.conf_file_path = ss.str();
                    ss.str("");
                }
                else if (strcmp(argv[i], "-d") == 0) {
                    if (strcmp(argv[i+1], "all") == 0) {
                        out_conf.descriptor_extractor_type.push_back("SURF");
                        out_conf.descriptor_extractor_type.push_back("SIFT");
                        out_conf.descriptor_extractor_type.push_back("BRIEF");
                        out_conf.descriptor_extractor_type.push_back("BRISK");
                        out_conf.descriptor_extractor_type.push_back("ORB");
                        out_conf.descriptor_extractor_type.push_back("FREAK");
                    }
                    else {
                        ss << argv[++i];
                        out_conf.descriptor_extractor_type.push_back(ss.str());
                        ss.str("");
                    }

                }
                else if (strcmp(argv[i], "-k") == 0) {
                    out_conf.keypoint_size = atoi(argv[++i]);
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

        fs["MaxPoses"] >> out_conf.max_poses;
        fs.release();
    }

    void check_sequence(cv::FileNode fn) {
        if(fn.type() != cv::FileNode::SEQ) {
            cerr << "Configuration file error: not a sequence." << endl;
            exit(-1);
        }
    }

    void load_train_paths(string main_path, vector<string> persons_names, vector<string> views_names,
                          Mat num_images, vector<vector<string> > &skels_paths, vector<vector<string> > &imgs_paths) {

        assert(persons_names.size() == num_images.rows);

        std::stringstream ss_imgs, ss_skels;

        for (int i = 0; i < persons_names.size(); ++i) {
            vector<string> imgs_path;
            vector<string> skels_path;
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

    void load_train_paths(Configuration conf, vector<vector<string> > &out_skels_paths,
                          vector<vector<string> > &out_imgs_paths) {
        load_train_paths(conf.main_path, conf.persons_names, conf.views_names, conf.num_images, out_skels_paths, out_imgs_paths);
    }

    bool load_models(string descriptor_extractor_type, int keypoint_size, int max_poses, vector<Mat> &masks,
                     vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                     vector<MultiviewBodyModel> &out_models, Timing &timing) {
        // Timing
        double t0 = (timing.enabled) ? (double)cv::getTickCount() : 0;

        // Checking dimensions
        assert(train_imgs_paths.size() == train_skels_paths.size());


        // train_skels_paths.size() = the number of people
        for (int i = 0; i < train_skels_paths.size(); ++i) {

            // Checking dimensions
            assert(train_imgs_paths[i].size() == train_skels_paths[i].size());

            MultiviewBodyModel body_model(max_poses);

            // current image
            int j = 0;
            // number of images inserted
            // NOTE: image already considered for the matching then masks[i].at(j, 0) = 1 , otherwise 0
            int non_zero_counter = countNonZero(masks[i].row(0));


            // Look for the list with the maximum number of elements
            int max_size_idx = 0;
            for (int k = 1; k < train_imgs_paths.size(); ++k) {
                if (train_imgs_paths[max_size_idx].size() < train_imgs_paths[k].size())
                    max_size_idx = k;
            }

            // Accept poses only with this value in the mask's element
            int value = 0;

            // The loop continue untill the maximum size mask has only non-zero elements
            while (!body_model.ready() && non_zero_counter <= train_imgs_paths[max_size_idx].size()) {

                // Insert the pose if not present, and remove it from the paths
                char *mask_elem = &masks[i].row(0).at<char>(j);
                if (*mask_elem == value && *mask_elem != -1) {

                    int result = body_model.ReadAndCompute(train_skels_paths[i][j], train_imgs_paths[i][j],
                                                          descriptor_extractor_type, keypoint_size, timing);
                    // Check the value returned
                    // -1 => the pose side is not valid -> mark the relative element -1
                    //  0 => the pose side already exists -> do nothing
                    //  1 => the pose side
                    if (result == 1) {
                        (*mask_elem)++;
                        non_zero_counter++;
                    }
                    else if (result == -1) {
                        (*mask_elem) = -1;
                        non_zero_counter++;
                    }
                }
                ++j;

                // If the end of the list is reached, start from the beginning
                // In this way we assure all images in the data set will be considered (even multiple times)
                if (j == train_imgs_paths[i].size()) {
                    j = 0;
                    value++;
                }
            }

            // If the model contains all poses then add it to the vector
            // otherwise the model is not valid, then exit.
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

    void read_skel(string descriptor_extractor_type, int keypoint_size, string skel_path, string img_path, Mat &out_image,
                   vector<cv::KeyPoint> &out_keypoints, vector<float> &out_confidences, Mat &out_descriptors,
                   int &out_pose_side, Timing &timing) {

        double t0 = (timing.enabled) ? cv::getTickCount() : 0;

        // Read the file
        string line;
        std::ifstream file(skel_path);
        if (!file.is_open()) {
            std::cerr << "read_skel: " << skel_path << "Invalid file name." << std::endl;
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
                            out_confidences.push_back(0);
                        else
                            out_confidences.push_back(conf);

                        // Reset to 0 and go to the next keypoint
                        value_type %= 2;
                        break;
                }
            }
            ++i;
        }

        // Last line contains the pose side
        std::stringstream ss(line);
        ss >> out_pose_side;

        // Read image
        out_image = cv::imread(img_path);
        if (!out_image.data) {
            std::cerr << "read_skel: " << img_path << "Invalid image file." << std::endl;
        }

        // Compute descriptors for this view
        cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = cv::DescriptorExtractor::create(descriptor_extractor_type);
        descriptor_extractor->compute(out_image, out_keypoints, out_descriptors);

        if (timing.enabled) {
            timing.t_tot_skel_loading += ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
            timing.n_tot_skel_loading++;
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
        fs << "avgLoadingTrainingSet" << (t_tot_load_training_set / n_tot_load_training_set);
        fs << "avgOneRound" << (t_tot_round / n_rounds);
        fs << "avgDescriptorsComputation" << (t_tot_descriptors / n_tot_descriptors);
        fs << "avgOneModelLoading" << (t_tot_model_loading / n_tot_model_loading);
        fs << "avgSkelLoading" << (t_tot_skel_loading / n_tot_skel_loading);
        fs << "totMatching" << t_tot_matching;
        fs << "descriptorNames" << "[";
        string data;
        for (int i = 0; i < descriptor_names.size(); ++i) {
            fs << descriptor_names[i];
        }
        fs << "]";
        fs << "descriptorTimes" << "[";
        for (int j = 0; j < t_descriptor_names.size(); ++j) {
            fs << t_descriptor_names[j];
        }
        fs << "]";
        fs.release();
    }

    void multiviewbodymodel::Timing::show() {
        cout << "----------------- PERFORMANCE -----------------" << endl;
        cout << "avgLoadingTrainingSet " << (t_tot_load_training_set / n_rounds);
        cout << "avgOneRound " << (t_tot_round / n_rounds);
        cout << "avgDescriptorsComputation " << (t_tot_descriptors / n_tot_descriptors);
        cout << "avgOneModelLoading " << (t_tot_model_loading / n_tot_model_loading);
        cout << "avgSkelLoading " << (t_tot_skel_loading / n_tot_skel_loading);
        cout << "totMatching " << t_tot_matching;
        cout << "-----------------------------------------------" << endl;
    }

    void saveCMC(string path, Mat cmc) {
        assert(cmc.rows == 1);

        cout << "Saving CMC...";
        std::ofstream file(path);
        // {(0,23.1)(1,27.5)(2,32)(3,37.8)(4,44.6)(6,61.8)(8,83.8)(10,100)};
        file << "coordinates {";
        for (int j = 0; j < cmc.cols; ++j) {
            file << "(" << j+1 << "," << cmc.at<float>(0, j) * 100 << ")";
        }
        file << "};";
        file.close();
        cout << "done!" << endl;
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





