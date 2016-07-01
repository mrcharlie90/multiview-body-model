//
// Created by Mauro on 15/04/16.
//

#include "MultiviewBodyModel.h"

namespace multiviewbodymodel {
    using namespace std;
    using namespace cv;

    /*
     * Constructor
     */
    MultiviewBodyModel::MultiviewBodyModel(int max_poses) {
        max_poses_ = max_poses;
    }

    /*
     * Adds new skeleton descriptors to the body model. The *_skel.txt file should contain 15 keypoints'
     * [float] coordinates with the pose side number [int] at the end.
     * Returns true if the pose's descriptors are successfully saved and false if the pose is already acquired.
     *
     *  Set timing NULL for no performance logging
     */
    bool MultiviewBodyModel::ReadAndCompute(string path, string img_path, string descriptor_extractor_type, int keypoint_size, Timing &timing) {

        // Timing
        double ti_model = (timing.enabled) ? (double)getTickCount() : 0;

        // Return value
        bool value = false;

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
            double ti_desc = (timing.enabled) ? (double)getTickCount() : 0;

            cv::Mat descriptors;
            cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = cv::DescriptorExtractor::create(descriptor_extractor_type);
            descriptor_extractor->compute(img, keypoints, descriptors);

            if (timing.enabled) {
                timing.t_tot_descriptors += ((double)getTickCount() - ti_desc) / getTickFrequency();
                timing.n_tot_descriptors++;
            }

            pose_side_.push_back(pose_side);
            views_descriptors_.push_back(descriptors);
            views_descriptors_confidences_.push_back(confidences);

            value = true;
        }
        if (timing.enabled) {
            timing.t_tot_model_loading += ((double)getTickCount() - ti_model) / getTickFrequency();
            timing.n_tot_model_loading++;
        }

        return value;
    }

    /**
     * Computes the match between this model poses and the query pose.
     * Confidences and the pose index side must be given.
     */
    float MultiviewBodyModel::match(Mat query_descriptors, vector<float> query_confidences,
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
                    if(operation == 1 && occlusion_search) {
                        // TODO: obsolete if
                        if (get_descriptor_occluded(k, descriptor_occluded)) {
                            // A descriptor is found, so compute the distance
                            dist = norm(query_descriptors.row(k), descriptor_occluded);
                            average_distance += dist;
                            descriptors_count++;
                        }
                    }
                    else if (operation == 2) {
                        dist = norm(query_descriptors.row(k), views_descriptors_[i].row(k));
                        average_distance += dist;
                        descriptors_count++;
                    }
                }
//            cout << average_distance / descriptors_count << endl;
                return average_distance / descriptors_count;
            }
        }
        return -1;
    }


    bool MultiviewBodyModel::ready() {
        return (pose_side_.size() == max_poses_);
    }

    /*
     * The matching is performed by following the mask values:
     * if mask(i,j) = 0 -> Don't consider keypoints
     * if mask(i,j) = 1 -> Find the keypoint occluded in the other views
     * if mask(i,j) = 2 -> Compute the distance between the keypoints
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
     * Finds the first non-occluded descriptor relative to the keypoint index
     * in another view of the model.
     * TRUE if the keypoint is found, FALSE otherwise.
     */
    bool MultiviewBodyModel::get_descriptor_occluded(int keypoint_index, Mat &descriptor_occluded) {

        // Find a non-occluded descriptor in one pose
        for (int i = 0; i < views_descriptors_.size(); ++i) {
            if (views_descriptors_confidences_[i][keypoint_index] > 0) {
//            std::cout << "descriptor k = " << keypoint_index << " found at view = " << i << std::endl;
                descriptor_occluded = views_descriptors_[i].row(keypoint_index);
                return true;
            }
        }
        return false;
    }





    //
    // <><><><><><><><><><> Main function definitions <><><><><><><><><><>
    //

    void Configuration::show() {
        cout << "<><><><><><><><> Configuration <><><><><><><><>" << endl;
        cout << "main path: " << main_path << endl;
        cout << "persons names: " << endl;
        print_list<string>(persons_names);
        cout << "views names: " << endl;
        print_list<string>(views_names);
        cout << "max poses: " << max_poses << endl;
        cout << "number of images: " << endl << num_images << endl;
        cout << "<><><><><><><><><><><><><><><><><><><><><><><><>" << endl;
    }

    /**
     * Parse input arguments and initialize the configuration object.
     */
    void parse_args(int argc, char **argv, Configuration &out_conf) {
        stringstream ss;

        for (int i = 1; i < argc; ++i) {
            if (i + 1 != argc) {
                if (strcmp(argv[i], "-c") == 0) {
                    ss << argv[++i];
                    out_conf.conf_file_path = ss.str();
                    ss.str("");
                }
                else if (strcmp(argv[i], "-d") == 0) {
                    ss << argv[++i];
                    out_conf.descriptor_extractor_type = ss.str();
                    ss.str("");
                }
                else if (strcmp(argv[i], "-k") == 0) {
                    out_conf.keypoint_size = atoi(argv[++i]);
                }
            }
        }

        FileStorage fs(out_conf.conf_file_path, FileStorage::READ);
        fs["MainPath"] >> out_conf.main_path;


        FileNode pn = fs["PersonNames"];
        check_sequence(pn);
        for (FileNodeIterator it = pn.begin(); it != pn.end(); ++it)
            out_conf.persons_names.push_back((string)*it);

        FileNode wn = fs["ViewNames"];
        check_sequence(wn);
        for (FileNodeIterator it = wn.begin(); it != wn.end(); ++it)
            out_conf.views_names.push_back((string)*it);

        fs["NumImages"] >> out_conf.num_images;

        if (out_conf.persons_names.size() != out_conf.num_images.rows) {
            cerr << "#persons != #num_images, check the configuration file!" << endl;
            exit(-1);
        }

        fs["MaxPoses"] >> out_conf.max_poses;
        fs.release();
    }

    // Checks if the file node fn is a sequence, used only in parse_args()
    void check_sequence(FileNode fn) {
        if(fn.type() != FileNode::SEQ) {
            cerr << "Configuration file error: not a sequence." << endl;
            exit(-1);
        }
    }

    /**
     * Loads training paths ordered by person.
     */
    void load_train_paths(string main_path, vector<string> persons_names, vector<string> views_names,
                          Mat num_images, vector<vector<string> > &skels_paths, vector<vector<string> > &imgs_paths) {

        assert(persons_names.size() == num_images.rows);

        stringstream ss_imgs, ss_skels;

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

    void load_train_paths(Configuration conf, vector<vector<string> > &skels_paths, vector<vector<string> > &imgs_paths) {
        load_train_paths(conf.main_path, conf.persons_names, conf.views_names, conf.num_images, skels_paths, imgs_paths);
    }

    /**
     * Converts the training paths sorted by person (vec<vec<string>>) to a straight lists of paths (vec<string>).
     */
    void load_test_paths(vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                         vector<string> &query_skels_paths, vector<string> &query_imgs_paths) {

        assert(train_imgs_paths.size() == train_skels_paths.size());

        for (int i = 0; i < train_imgs_paths.size(); ++i) {
            for (int j = 0; j < train_imgs_paths[i].size(); ++j) {
                query_skels_paths.push_back((train_skels_paths[i][j]));
                query_imgs_paths.push_back(train_imgs_paths[i][j]);
            }
        }
    }

    /*
     * Loads one model for each person with all the pose.
     * out_models: vector of body models where the descriptors are stored
     * masks: vector of masks, where mask[i].at(j, 0) is referred to the jth-pose of the ith-person.
     * Once the jth pose is loaded into the model, the relative mask element is updated (set to 1).
     *
     * Set timing = NULL for no performance logging
     *
     * returns TRUE if all the models are successfully loaded.
     */
    bool load_training_set(string descriptor_extractor_type, int keypoint_size, int max_poses, vector<Mat> &masks,
                           vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                           vector<MultiviewBodyModel> &out_models, Timing &timing) {
        // Timing
        double t0 = (timing.enabled) ? (double)getTickCount() : 0;

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
            // [NOTE: image already considered for the matching then masks[i].at(j, 0) = 1 , otherwise 0 ]
            int tot_images_marked = static_cast<int>(sum(masks[i].col(0))[0]);

            while (!body_model.ready() && tot_images_marked < train_imgs_paths[i].size()) {

                // Insert the pose if not present, and remove it from the paths
                if (masks[i].at<uchar>(j, 0) == 0) {
                    if (body_model.ReadAndCompute(train_skels_paths[i][j], train_imgs_paths[i][j],
                                                  descriptor_extractor_type, keypoint_size, timing)) {
                        masks[i].at<uchar>(j, 0) = 1;
                        tot_images_marked++;
                    }
                }
                ++j;
            }

            // If the model contains all poses then add it to the vector
            // otherwise the model is not valid, then exit.
            if (body_model.ready())
                out_models.push_back(body_model);
            else
                return false;
        }

        if (timing.enabled) {
            timing.t_tot_load_training_set += ((double)getTickCount() - t0) / getTickFrequency();
            timing.n_tot_load_training_set++;
        }

        return true;
    }

    bool load_training_set(Configuration conf, vector<Mat> &masks, vector<vector<string> > &train_skels_paths,
                           vector<vector<string> > &train_imgs_paths, vector<MultiviewBodyModel> &out_models, Timing &timing) {
        return load_training_set(conf.descriptor_extractor_type, conf.keypoint_size, conf.max_poses, masks,
                                 train_skels_paths, train_imgs_paths, out_models, timing);

    }


    /*
     * Reads the skeleton from a file and  computes its descriptors.
     * This is used to compute descriptors of an query image.
     */
    void read_skel(string descriptor_extractor_type, int keypoint_size, string skel_path, string img_path, Mat &out_image,
                   vector<KeyPoint> &out_keypoints, vector<float> &out_confidences, Mat &out_descriptors,
                   int &out_pose_side) {
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
            std::cerr << "Invalid image file." << std::endl;
        }

        // Compute descriptors for this view
        cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = cv::DescriptorExtractor::create(descriptor_extractor_type);
        descriptor_extractor->compute(out_image, out_keypoints, out_descriptors);
    }

    void read_skel(Configuration conf, string skel_path, string img_path, Mat &out_image, vector <KeyPoint> &out_keypoints,
                       vector<float> &out_confidences, Mat &out_descriptors, int &out_pose_side, Timing &timing) {
        double t0 = (timing.enabled) ? getTickCount() : 0;
        read_skel(conf.descriptor_extractor_type, conf.keypoint_size,
                  skel_path,
                  img_path, out_image, out_keypoints,
                  out_confidences, out_descriptors, out_pose_side);

        if (timing.enabled) {
            timing.t_tot_skel_loading += ((double)getTickCount() - t0) / getTickFrequency();
            timing.n_tot_skel_loading++;
        }
    }

    /**
     * Return the index of the element in the priority queue whose class is equal to query image one.
     */
    template<typename T> int get_rank_index(priority_queue<RankElement<T>, vector<RankElement<T> >, RankElement<T> > pq,
                                            int query_class) {
        // Work on a copy
        priority_queue<RankElement<T>, vector<RankElement<T> >, RankElement<T> > scores(pq);

        // Searching for the element with the same class and get the rank
        for (int i = 0; i < pq.size(); i++)
        {
            if (scores.top().classIdx == query_class)
                return i;
            scores.pop();
        }
        return (int) (pq.size() - 1);
    }

    template int get_rank_index<float>(priority_queue<RankElement<float>, vector<RankElement<float> >, RankElement<float> > pq,
                                       int query_class);


//    Mat compute_increment_matrix(vector<vector<string> > train_paths, Mat num_images, int num_persons,
//                                 int num_views, int max_poses) {
//
//        assert(num_images.rows == train_paths.size());
//
//        int size[] = {num_views, max_poses, num_persons};
//        Mat M(3, size, CV_32F);
//        Mat C;
//        C = Mat::ones(3, size, CV_32F);
//
//        for (int k = 0; k < train_paths.size(); ++k) {
//            int j = get_pose_side(train_paths[k][0]) - 1;
//            M.at<float>(0, j, k)++;
//
//            for (int i = 1; i < train_paths[k].size(); ++i) {
//                cout << train_paths[k][i] << endl;
//                int tmp = get_pose_side(train_paths[k][i]) - 1;
//                int w = i / (num_images.at<uchar>(k, 0) + 1);
//
//                assert(w < 3);
//
//                if(tmp == j) {
//                    M.at<float>(w, j, k)++;
//                }
//                else {
//                    C.at<float>(w, j, k)++;
//                    j = tmp;
//                    M.at<float>(w, j, k)++;
//                }
//            }
//        }
//
//        return (M / C);
//    }


    int get_pose_side(string path) {

        // Read the file
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Invalid file name." << std::endl;
            exit(-1);
        }

        file.seekg(0, ios::end);
        int pos = file.tellg();
        pos-=2;
        file.seekg(pos);

        string line;
        getline(file, line);
        stringstream ss(line);


        int pose_side;
        ss >> pose_side;

        file.close();

        return pose_side;
    }

    /**
     * Counts the number of successive images with the same pose side
     */
    /**
     * Example:
     *  img   |   pose
     *  1     |   1
     *  2     |   1
     *  3     |   1
     *  4     |   2
     *  5     |   3
     *  6     |   3
     *  7     |   3
     *  8     |   3
     *  9     |   4
     *  10    |   4
     *
     *  produce
     *  [1 3 2 1 3 4 4 2]
     *  for each element i
     *  if i is even => pose
     *  if i is odd => number of successive images with the same pose
     */
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

    /**
     * From a map with the number of successively pose sides, give you
     * a set of random indeces uniformly distributed.
     *
     * Returns the total number of images
     */
    int get_rnd_indeces(vector<vector<int> > map, vector<vector<int> > &rnd_indeces) {

        // Total number of images
        int tot_imgs = 0;
        for (int i = 0; i < map.size(); ++i) {
            // for i odd, map[i] has the number of successively pose sides
            // in the dataset
            vector<int> vec;
            int current_idx = 0;
            // Build a vector containing a number of random indeces referred
            // to images in the dataset
            for (int j = 0; j <= (map[i].size() - 1) / 2; ++j) {
                RNG rng((uint64) getTickCount());
                // Random number between [0, map[][])
                int n = (int)rng.uniform(0., (double)(map[i][2 * j + 1]));
                vec.push_back(current_idx + n);
                current_idx += map[i][2 * j + 1];
            }
            tot_imgs += vec.size();
            rnd_indeces.push_back(vec);
        }

        return tot_imgs;
    }

    /**
     * Results
     */
    void saveCMC(string path, Mat cmc) {
        assert(cmc.rows == 1);

        cout << "Saving CMC...";
        ofstream file(path);
        // {(0,23.1)(1,27.5)(2,32)(3,37.8)(4,44.6)(6,61.8)(8,83.8)(10,100)};
        file << "coordinates {(0,0)";
        for (int j = 0; j < cmc.cols; ++j) {
            file << "(" << j+1 << "," << cmc.at<float>(0, j) * 100 << ")";
        }
        file << "};";
        file.close();
        cout << "done!" << endl;
    }


    /*
     * Timing Methods
     */

    void multiviewbodymodel::Timing::write() {
        FileStorage fs("timing.xml", FileStorage::WRITE);
        fs << "avgLoadingTrainingSet" << (t_tot_load_training_set / n_tot_load_training_set);
        fs << "avgOneRound" << (t_tot_round / n_rounds);
        fs << "avgDescriptorsComputation" << (t_tot_descriptors / n_tot_descriptors);
        fs << "avgOneModelLoading" << (t_tot_model_loading / n_tot_model_loading);
        fs << "avgSkelLoading" << (t_tot_skel_loading / n_tot_skel_loading);
        fs << "totMatching" << t_tot_matching;
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

    /**
     * Logging
     */
    template<typename T> void print_list(vector<T> vect) {
        cout << "[" << vect[0];
        for (int i = 1; i < vect.size(); ++i) {
            cout << ", " << vect[i];
        }
        cout << "]" << endl;
    }
    template void print_list<int>(vector<int> vect);
    template void print_list<float>(vector<float> vect);
    template<typename T> void print_list(priority_queue<RankElement<T>, vector<RankElement<T> >, RankElement<T> > queue) {
        RankElement<T> re = queue.top();
        cout << "[" << re.score << "|" << re.classIdx;
        queue.pop();
        while (!queue.empty()) {
            re = queue.top();
            cout << ", " << re.score << "|" << re.classIdx;
            queue.pop();
        }
        cout << "]" << endl;
    }
    template void print_list<float>(priority_queue<RankElement<float>, vector<RankElement<float> >, RankElement<float> > queue);
}





