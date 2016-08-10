// Copyright (c) [2016] [Mauro Piazza]
//
//          IASLab License
//
// Main used for testing the MultiviewMBodyModel class.
// First of all it loads the parameters settings.
//
// Then it test a single or a set of descriptors with the models created.
// The test set is chosen from a random set of images and the overall dataset
// is used to load models.
//
// When the models are loaded all the frames in the test set are matched,
// compunting the CMC curve and the relative nAUC.
//
// Then the results are stored in files contained in the result directory
// specified by the configuration file
//
// Methods to test the replace function and for seeing the memory usage are implemented.
//

#include "MultiviewBodyModel.h"

using namespace multiviewbodymodel;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::priority_queue;
using cv::string;
using cv::Mat;
using cv::KeyPoint;
using multiviewbodymodel::PQRank;


int main(int argc, char** argv)
{


    
//    Mat m;
//    m = Mat::zeros(1,9,CV_8S);
//    Mat a;
//    a = Mat::zeros(1,3,CV_8S);
//    a.row(0).at<char>(0) = 0;
//    a.row(0).at<char>(1) = 1;
//    a.row(0).at<char>(2) = 2;
//    Mat b;
//    b = Mat::zeros(1,3,CV_8S);
//    b.row(0).at<char>(0) = 0;
//    b.row(0).at<char>(1) = 10;
//    b.row(0).at<char>(2) = 20;
//    Mat c;
//    c = Mat::zeros(1,3,CV_8S);
//    c.row(0).at<char>(0) = 0;
//    c.row(0).at<char>(1) = 11;
//    c.row(0).at<char>(2) = 22;
//
//
//    a.copyTo(m.colRange(cv::Range(0,3)));
//    b.copyTo(m.colRange(cv::Range(3,6)));
//    c.copyTo(m.colRange(cv::Range(6,9)));
//
//    cout << m << endl;

    int kp_map_data[75] = {-1, -1, 5, -1, -1, 2, -1, -1, 8, 12, 13, -1, 9, 10, -1,
                         0, -1, 2, 3, 4, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1,
                         0, -1, -1, -1, -1, 5, 6, 7, 8, -1, -1, -1, 12, 13, 14,
                        -1, -1, 2, 3, 4, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1,
                        -1, -1, -1, -1, -1, 5, 6, 7, 8, -1, -1, -1, 12, 13, 14};

    float w_data[75] = {0, 0, 0.5, 0, 0, 0.5, 0, 0, 1, 0.5, 0.3, 0, 0.5, 0.3, 0,
                       0.3, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0.5, 0.5, 0.7, 0.7, 0, 0, 0,
                       0.3, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0.5, 0.7, 0.7,
                       0, 0, 0.5, 0.5, 0.3, 0, 0, 0, 0.5, 0.5, 0.5, 0.3, 0, 0, 0,
                       0, 0, 0, 0, 0, 0.5, 0.5, 0.3, 0.5, 0, 0, 0, 0.5, 0.5, 0.3};

    int poses2weights_data[12] = {0, 1, 2,
                                    0, 3, 4,
                                    1, 3, -1,
                                    2, 4, -1};

    int poses_map_data[12] = {2, 3, 4,
                                1, 3, 4,
                                1, 2, -1,
                                1, 2, -1};

    Mat kp_map(5, 15, CV_32S, kp_map_data);
    Mat kp_weights(5, 15, CV_32F, w_data);
    Mat poses2weights_map(4, 3, CV_32S, poses2weights_data);
    Mat poses_map(4, 3, CV_32S, poses_map_data);


    Configuration conf;

    if (argc < 7) {
        show_help();
    }
    else {
        parse_args(argc, argv, conf);
        conf.show();
    }

    // Load the training set
    vector<vector<string> > imgs_paths;
    vector<vector<string> > skels_paths;

    load_person_imgs_paths(conf, imgs_paths, skels_paths);

    // Images used for models loading
    vector<int> poses;
    poses.push_back(1);
    poses.push_back(2);
    poses.push_back(3);
    poses.push_back(4);

    const int k_model_set_size = 13;

    vector<vector<int> > models_set;
    int models_per_person = load_models_set(poses, imgs_paths, skels_paths,
                               conf.keypoints_number, k_model_set_size, 13, models_set);
    Timing timing;

    bool models_set_completed = false;

    // Init the models
    vector<MultiviewBodyModel> models;
    for (int i = 0; i < models_set.size(); ++i) {
        MultiviewBodyModel model;
        models.push_back(model);
    }

    int rounds = 1;
    while (rounds <= models_per_person) {
        for (int i = 0; i < models_set.size(); ++i) {
            for (int pose_idx = 0; pose_idx < poses.size(); ++pose_idx) {
                models[i].read_pose_compute_descriptors(imgs_paths[i][models_set[i][pose_idx * rounds]],
                                                        skels_paths[i][models_set[i][pose_idx * rounds]],
                                                        conf.keypoint_size[0], conf.descriptor_extractor_type[0], timing);
            }
        }

        // Rates of the current test image
        cv::Mat rates;
        rates = cv::Mat::zeros(1, static_cast<int>(conf.persons_names.size()), CV_32F);


        vector<Mat> weights;
        Mat W;
        W = Mat::ones(4, 15, CV_8S);

        // foreach image in the data set do
        for (int i = 0; i < skels_paths.size(); ++i) {
            for (int j = 0; j < skels_paths[i].size(); ++j) {
                Mat img;
                Mat descriptors;
                vector<KeyPoint> keypoints;
                vector<float> confidences;
                int ps;

                read_skel_file(skels_paths[i][j], conf.keypoint_size[0], keypoints, confidences, ps);
                compute_descriptors(imgs_paths[i][j], keypoints, conf.descriptor_extractor_type[0], descriptors);


                // Match the current frame with each model and compute the rank
                priority_queue<PQRank<float>, vector<PQRank<float> >, PQRank<float> > scores;
                cout << "Matching " << skels_paths[i][j] << " with" << endl;
                for (int k = 0; k < models.size(); ++k) {
                    PQRank<float> rank_elem;
                    rank_elem.score = models[k].match(descriptors, ps, confidences, conf.norm_type, true, poses_map,
                                                      kp_map, kp_weights, poses2weights_map,
                                                      timing);
                    rank_elem.class_idx = k;
                    scores.push(rank_elem);

                    cout << "   model " << k << " ";
                    cout << rank_elem.score << endl;
                }

                // Update all rates
                // The resulting rates are used to compute the CMC curve
                int rank_idx = get_rank_index<float>(scores, i);
                for (int k = 0; k < rates.cols; ++k) {
                    rates.at<float>(0, k)++;
                }
            }
        } // end foreach

        rounds++;
    }

    return 0;
}