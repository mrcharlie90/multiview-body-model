//
// Created by Mauro on 10/05/16.
//

#include "MultiviewBodyModel.h"

using namespace multiviewbodymodel;
using namespace std;

/**
 * Element constructed in the ranking phase.
 * score: score got by the matching algorithm
 * classIdx: ground truth class of the model
 *           from which we obtained the score.
 */
template<typename T>
struct RankElement {
public:
    int classIdx;
    T score;

    // Comparator for the priority queue
    bool operator()(const RankElement<T> &re1, const RankElement<T> &re2) {
        return re1.score > re2.score;
    }
};

/**
 * Group all the necessary information for obtaining training(testing) files
 * and parameters.
 */
struct Configuration {
    // From the conf file
    string conf_file_path;
    string main_path;
    vector<string> persons_names;
    vector<string> views_names;
    Mat num_images;
    int max_poses;

    // From the command line
    string descriptor_extractor_type;
    int keypoint_size;

    void show() {
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
};


void load_train_paths(string main_path, vector<string> persons_names, vector<string> views_names,
                      Mat num_images,
                      vector<vector<string> > &imgs_paths, vector<vector<string> > &skel_paths);
void load_train_paths(Configuration conf, vector<vector<string> > &skels_paths, vector<vector<string> > &imgs_paths);

void load_test_paths(vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                     vector<string> &query_skels_paths, vector<string> &query_imgs_paths);

bool load_training_set(string descriptor_extractor_type, int keypoint_size, int max_poses, vector<Mat> &masks,
                       vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                       vector<MultiviewBodyModel> &out_models);
bool load_training_set(Configuration conf, vector<Mat> &masks, vector<vector<string> > &train_skels_paths,
                       vector<vector<string> > &train_imgs_paths, vector<MultiviewBodyModel> &out_models);

void read_skel(string descriptor_extractor_type, int keypoint_size, string skel_path, string img_path, Mat &out_image,
               vector<KeyPoint> &out_keypoints, vector<float> &out_confidences, Mat &out_descriptors,
               int &out_pose_side);

void read_skel(Configuration conf, string skel_path, string img_path, Mat &out_image, vector<KeyPoint> &out_keypoints,
               vector<float> &out_confidences, Mat &out_descriptors, int &pose_side);

void check_sequence(FileNode fn);

void parse_args(int argc, char **argv, Configuration &out_conf);

template<typename T>
int get_rank_index(priority_queue<RankElement<T>, vector<RankElement<T> >, RankElement<T> > pq, int query_class);

template<typename T>
void print_list(priority_queue<RankElement<T>, vector<RankElement<T> >, RankElement<T> > queue);

template<typename T>
void print_list(vector<T> vect);

int main(int argc, char** argv)
{

    Configuration conf;

    if (argc < 7) {
        cout << "USAGE: multiviewbodymodel -c <configfile> -d <descriptortype> -k <keypointsize>";
        exit(-1);
    }
    else {
        parse_args(argc, argv, conf);
        conf.show();
    }

    // Load the training set
    vector<vector<string> > train_imgs_paths;
    vector<vector<string> > train_skels_paths;
    load_train_paths(conf, train_skels_paths, train_imgs_paths);

    // Load queries
    vector<string> test_imgs_paths;
    vector<string> test_skels_paths;
    load_test_paths(train_skels_paths, train_imgs_paths,
                    test_skels_paths, test_imgs_paths);

    assert(test_imgs_paths.size() == test_skels_paths.size());
    
    // Used for storing a set of masks in which one of them marks (with 0 or 1) the pose
    // already chosen for the body model loading phase
    vector<Mat> masks;
    for (int j = 0; j < train_imgs_paths.size(); ++j) {
        Mat mask;
        mask = Mat::zeros(static_cast<int>(train_imgs_paths[j].size()), 1, CV_8U);
        masks.push_back(mask);
    }

    // Overall number of test images for each person
    // note: +1 because image's names start from 0
    vector<int> num_test_images;
    num_test_images.push_back( (conf.num_images.at<uchar>(0, 0) + 1) * static_cast<int>(conf.views_names.size()) );
    num_test_images.push_back( (conf.num_images.at<uchar>(1, 0) + 1) * static_cast<int>(conf.views_names.size()) );
    num_test_images.push_back( (conf.num_images.at<uchar>(2, 0) + 1) * static_cast<int>(conf.views_names.size()) );
    int tot_test_images = ((static_cast<int>(sum(conf.num_images)[0])) + 3) * static_cast<int>(conf.views_names.size());

    // Body Models used for testing
    vector<MultiviewBodyModel> models;



    // Cumulative Matching Characteristic curve:
    // contains the average person re-identification rate
    Mat CMC;
    CMC = Mat::zeros(1, static_cast<int>(conf.persons_names.size()), CV_32F);

    // Number of cycles
    int round = 1;
    while(load_training_set(conf, masks, train_skels_paths, train_imgs_paths, models)) {

        printf("----------- Models loaded, round: %d -----------\n", round);

        // Rates of the current test image
        Mat current_rates;
        current_rates = Mat::zeros(1, static_cast<int>(conf.persons_names.size()), CV_32F);

        // Variables used for computing the current query frame class
        int test_class = 0;
        int test_class_counter = 0;

        for (int j = 0; j < test_imgs_paths.size(); ++j) {
            // Check if the test frames relative to one subject are all extracted
            if (test_class_counter != 0 && test_class_counter % num_test_images[test_class] == 0) {
                // Update the class and reset the counter
                test_class++;
                test_class_counter = 0;
            }

            // Testing variables
            Mat test_image;
            Mat test_descriptors;
            vector<KeyPoint> test_keypoints;
            vector<float> test_confidences;
            int test_pose_side;

            // Loading the test frame skeleton
            read_skel(conf, test_skels_paths[j],
                      test_imgs_paths[j], test_image, test_keypoints,
                      test_confidences, test_descriptors, test_pose_side);

            // Compute the match score between the query image and each model loaded
            // the result is inserted into the priority queue "scores"
            priority_queue< RankElement<float>, vector<RankElement<float> >, RankElement<float> > scores;
            for (int k = 0; k < models.size(); ++k) {
                RankElement<float> rank_element;
                rank_element.score = models[k].match(test_descriptors, test_confidences, test_pose_side);
                rank_element.classIdx = k;
                scores.push(rank_element);
            }

            // All rates starting from rank_idx to the number of training models are updated
            // The results are used for computing the CMC curve
            int rank_idx = get_rank_index(scores, test_class);
            for (int c = rank_idx; c < current_rates.cols; ++c) {
                current_rates.at<float>(0, c)++;
            }

            test_class_counter++;
//            cout << "done! class = " << query_class << " rank_idx = " << rank_idx << endl;
        }

        for (int i = 0; i < current_rates.cols; ++i) {
            current_rates.at<float>(0, i) /= tot_test_images;
        }

        CMC += current_rates;
        cout << "CMC: " << CMC << endl;

        round++;
        models.clear();

        cout << "current_rates: " << current_rates << endl;
        printf("------------------------------------------------\n\n");
    }
    return 0;
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
 * returns TRUE if all the models are successfully loaded.
 */
bool load_training_set(string descriptor_extractor_type, int keypoint_size, int max_poses, vector<Mat> &masks,
                       vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                       vector<MultiviewBodyModel> &out_models) {
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
                                                      descriptor_extractor_type, keypoint_size)) {
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

    return true;
}

bool load_training_set(Configuration conf, vector<Mat> &masks, vector<vector<string> > &train_skels_paths,
                       vector<vector<string> > &train_imgs_paths, vector<MultiviewBodyModel> &out_models) {
    return load_training_set(conf.descriptor_extractor_type, conf.keypoint_size, conf.max_poses, masks,
                      train_skels_paths, train_imgs_paths, out_models);
}
/*
 * Reads the skeleton from a file and  computes its descritpors.
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

void read_skel(Configuration conf, string skel_path, string img_path, Mat &out_image, vector<KeyPoint> &out_keypoints,
               vector<float> &out_confidences, Mat &out_descriptors, int &out_pose_side) {
    read_skel(conf.descriptor_extractor_type, conf.keypoint_size,
              skel_path,
              img_path, out_image, out_keypoints,
              out_confidences, out_descriptors, out_pose_side);
}

/**
 * Return the index of the element in the priority queue whose class is equal to query image one.
 */
template<typename T>
int get_rank_index(priority_queue<RankElement<T>, vector<RankElement<T> >, RankElement<T> > pq, int query_class) {
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

/**
 * Methods for output on console
 */
template<typename T>
void print_list(vector<T> vect) {
    cout << "[" << vect[0];
    for (int i = 1; i < vect.size(); ++i) {
        cout << ", " << vect[i];
    }
    cout << "]" << endl;
}

template<typename T>
void print_list(priority_queue<RankElement<T>, vector<RankElement<T> >, RankElement<T> > queue) {
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