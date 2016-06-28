//
// Created by Mauro on 10/05/16.
//

#include "MultiviewBodyModel.h"

using namespace multiviewbodymodel;
using namespace std;

void read_skel(string skel_path, string img_path, string descriptor_extractor_type, int keypoint_size,
               Mat &out_image, vector<KeyPoint> &out_keypoints, vector<float> &out_confidences, Mat &out_descriptors,
               int &pose_side);

void load_train_paths(string main_path, vector<string> persons_names, vector<string> views_names,
                      Mat num_images,
                      vector<vector<string> > &imgs_paths, vector<vector<string> > &skel_paths);

bool load_training_set(vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                       int keypoint_size, int max_poses, string descriptor_extractor_type, vector<Mat> &masks,
                       vector<MultiviewBodyModel> &out_models);

void load_query_paths(vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                      vector<string> &query_skels_paths, vector<string> &query_imgs_paths);

template<typename T>
void print_list(vector<T> vect);

template<typename T>
struct RankElement {
public:
    int classIdx;
    T score;
};

template<typename T>
struct Comp {
    bool operator()(const RankElement<T> &re1, const RankElement<T> &re2) {
        return re1.score > re2.score;
    }
};

template<typename T>
void print_list(priority_queue<RankElement<T>, vector<RankElement<T> >, Comp<T> > queue);

template<typename T>
int get_rank_index(priority_queue<RankElement<T>, vector<RankElement<T> >, Comp<T> > pq, int query_class);

void parse_args();

int main(int argc, char** argv)
{

    if (argc > 1) {
        parse_args();
    }

    string filename = "conf.xml";
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "MainPath" << "../ds/";

    fs << "PersonNames" << "[";
    fs << "gianluca_sync" << "marco_sync" << "matteol_sync";
    fs << "]"

    fs << "ViewNames" << "[";
    fs << "c" << "l" << "r";
    fs << "]";

    fs << "NumImages" << "[";
    fs << 74 << 84 << 68;
    fs << "]";

    fs << "MaxPoses" << 4;
    fs.release();


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


    // Number of images for each view
    Mat num_images(3, 1, CV_8UC1);
    num_images.at<uchar>(0, 0) = 74;
    num_images.at<uchar>(1, 0) = 84;
    num_images.at<uchar>(2, 0) = 68;

    // Load the training set
    vector<vector<string> > train_imgs_paths;
    vector<vector<string> > train_skels_paths;
    load_train_paths(main_path, persons_names, views_names, num_images, train_skels_paths, train_imgs_paths);

    // Load queries
    vector<string> query_imgs_paths;
    vector<string> query_skels_paths;
    load_query_paths(train_skels_paths, train_imgs_paths, query_skels_paths, query_imgs_paths);
    assert(query_imgs_paths.size() == query_skels_paths.size());

    // Parameters
    int max_poses = 4;
    string descriptor_extractor_type = "SIFT";
    int keypoint_size = 9;

    // Mask is used for marking poses recently considered in the matching
    vector<Mat> masks;

    for (int j = 0; j < train_imgs_paths.size(); ++j) {
        Mat mask;
        mask = Mat::zeros(static_cast<int>(train_imgs_paths[j].size()), 1, CV_8U);
        masks.push_back(mask);
    }

    // Overall number of tot test images for each person
    // note: +1 because images names start from 0, look the definition of num_images
    vector<int> num_test_images;
    num_test_images.push_back((num_images.at<uchar>(0, 0) + 1) * static_cast<int>(views_names.size()) );
    num_test_images.push_back((num_images.at<uchar>(1, 0) + 1) * static_cast<int>(views_names.size()) );
    num_test_images.push_back((num_images.at<uchar>(2, 0) + 1) * static_cast<int>(views_names.size()) );

    int tot_query_images = ((static_cast<int>(sum(num_images)[0])) + 3) * static_cast<int>(views_names.size());

    // Body Models used for testing
    vector<MultiviewBodyModel> models;

    // Counting the number of times models are loaded (round of the main while cycle)
    int round = 1;

//    Mat CMC;
//    CMC = Mat::zeros(static_cast<int>(persons_names.size()), 1, CV_32F);
    Mat CMC;
    CMC = Mat::zeros(1, static_cast<int>(persons_names.size()), CV_32F);

    while(load_training_set(train_skels_paths, train_imgs_paths, keypoint_size, max_poses,
                            descriptor_extractor_type, masks, models)) {

        printf("----------- Models loaded, round: %d -----------\n", round);
        Mat current_rates;
        current_rates = Mat::zeros(1, static_cast<int>(persons_names.size()), CV_32F);

        // Variables used for computing the current query frame class
        int query_class = 0;
        int query_class_counter = 0;

        for (int j = 0; j < query_imgs_paths.size(); ++j) {

//            cout << "matching image: " << query_imgs_paths[j] << "...";

            // Check if the query frames relative to one subject are all extracted
            if (query_class_counter != 0 && query_class_counter % num_test_images[query_class] == 0) {
                // Update the class and reset the counter
                query_class++;
                query_class_counter = 0;
            }

            // Query frame variables
            Mat query_image;
            Mat query_descriptors;
            vector<KeyPoint> query_keypoints;
            vector<float> query_confidences;
            int query_pose_side;

            // Loading the query frame skeleton
            read_skel(query_skels_paths[j], query_imgs_paths[j], descriptor_extractor_type, keypoint_size,
                      query_image, query_keypoints, query_confidences, query_descriptors, query_pose_side);

            // Compute the match score between the query image and each model loaded
            // the result is inserted into the priority queue "scores"
            priority_queue< RankElement<float>, vector<RankElement<float> >, Comp<float> > scores;
            for (int k = 0; k < models.size(); ++k) {
                RankElement<float> rank_element;
                rank_element.score = models[k].match(query_descriptors, query_confidences, query_pose_side);
                rank_element.classIdx = k;
                scores.push(rank_element);
            }

            // All rates starting from rank_idx to the number of training models are updated
            // The results are used for computing the CMC curve
            int rank_idx = get_rank_index(scores, query_class);
            for (int c = rank_idx; c < current_rates.cols; ++c) {
                current_rates.at<float>(0, c)++;
            }

            query_class_counter++;
//            cout << "done! class = " << query_class << " rank_idx = " << rank_idx << endl;
        }

        for (int i = 0; i < current_rates.cols; ++i) {
            current_rates.at<float>(0, i) /= tot_query_images;
        }

        CMC += current_rates;
        cout << "CMC: " << CMC << endl;

        round++;
        models.clear();

        cout << "current_rates: " << current_rates << endl;
        printf("------------------------------------------------\n\n", round);
    }
    return 0;
}

void parse_args() {


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

/*
 * Given the training images and the skeleton images paths, it loads one model for each person with all the pose
 * int the out_models vector and returns true if all the models are successfully loaded.
 *
 * If one or more models cannot be loaded because all training images are used, then method returns false.
 *
 * Mask is a vector of masks, where mask[i].at[j, 0] is referred to the ith-person with jth-pose
 * Once the jth pose is chosen to be loaded into the model, the relative mask's element is updated.
 */
bool load_training_set(vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                       int keypoint_size, int max_poses, string descriptor_extractor_type, vector<Mat> &masks,
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

/*
 * Reads the skeleton from a file and  computes its descritpors.
 * This is used to compute descriptors of an query image.
 */
void read_skel(string skel_path, string img_path, string descriptor_extractor_type, int keypoint_size,
               Mat &out_image, vector<KeyPoint> &out_keypoints, vector<float> &out_confidences, Mat &out_descriptors,
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

/**
 * Return the index of the element in the priority queue whose class is equal to query image one.
 */
template<typename T>
int get_rank_index(priority_queue<RankElement<T>, vector<RankElement<T> >, Comp<T> > pq, int query_class) {

    // Work on a copy
    priority_queue<RankElement<T>, vector<RankElement<T> >, Comp<T> > scores(pq);

    // Searching for the element with the same class and get the rank
    for (int i = 0; i < pq.size(); i++)
    {
        if (scores.top().classIdx == query_class)
            return i;
        scores.pop();
    }
    return (int) (pq.size() - 1);
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
void print_list(priority_queue<RankElement<T>, vector<RankElement<T> >, Comp<T> > queue) {
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