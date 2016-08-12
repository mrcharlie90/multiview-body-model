// Copyright (c) [2016] [Mauro Piazza]
//
//          IASLab License
//
// Main used for testing the MultiviewMBodyModel class.
// - loads the parameters settings.
// - creates a model set of images to use for models building
// - for each image in the dataset
//       + match the image with the current models
//       + computes the rank
//       + computes the rates
// - stores performance results in terms of CMC and nAUC
// - stores execution times in files
//

#include "MultiviewBodyModel.h"

using namespace multiviewbodymodel;
using namespace cv;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::priority_queue;

int main(int argc, char** argv)
{

    double t0 = (double)cv::getTickCount();
    int c = 1 + 1;
    cout << ((double)cv::getTickCount() - t0) / getTickFrequency() << endl;

    // Reading command line and parameters setting
    Configuration conf;
    if (argc < 6) {
        show_help();
    }
    else {
        parse_args(argc, argv, conf);
        if (!conf.conf_file_path.empty())
            read_config_file(conf);
        else {
            cerr << "Missing configuration file." << endl;
            exit(-1);
        }
        conf.show();
    }

    // Loading paths from the configuration parameters and
    // computing the total number of images in the dataset
    vector<vector<string> > imgs_paths;
    vector<vector<string> > skels_paths;
    int tot_imgs = load_person_imgs_paths(conf, imgs_paths, skels_paths);

    // Model set loading: model_set_size defines the number of images that will
    // be used for loading one person model, models_per_person is real number of images
    // used (an multiple of the # of poses)
    vector<vector<int> > models_set;
    int models_per_person = load_models_set(conf.poses, imgs_paths, skels_paths,
                                            conf.model_set_size, 13, models_set);

    // Used for storing performance
    Timing timing;
    timing.enable(conf.descriptor_type_str);

    // Cumulative Matching Characteristic curve:
    // contains the average person re-identification rate
    Mat CMC;
    CMC = Mat::zeros(1, static_cast<int>(models_set.size()), CV_32F);

    // Create empty MultiviewBodyModel object to store information
    // contained in the model set
    vector<MultiviewBodyModel> models;
    empty_models(static_cast<int>(models_set.size()), models);

    // Models loading: the # of poses chosen from the settings are loaded
    // in each model for each round
    int rounds = 1;
    while (rounds <= models_per_person) {
        printf("------------- %s Models loaded, rounds: %d -------------\n",
               conf.descriptor_type_str.c_str(), rounds);

        // Creates the models for each person defined in the configuration settings
        init_models(conf, rounds, imgs_paths, skels_paths, models_set, models, timing);

        // Rates of the current test image
        Mat rates;
        rates = Mat::zeros(1, static_cast<int>(models_set.size()), CV_32F);

        Mat W;
        W = Mat::ones(4, conf.keypoints_number, CV_8S);

        // foreach image in the data set do
        for (int i = 0; i < skels_paths.size(); ++i) {
            for (int j = 0; j < skels_paths[i].size(); ++j) {
                Mat frame_descriptors;
                vector<KeyPoint> frame_keypoints;
                vector<float> frame_confidences;
                int frame_pose;

                read_skel_file(skels_paths[i][j], conf.keypoint_size, frame_keypoints, frame_confidences, frame_pose);
                compute_descriptors(imgs_paths[i][j], frame_keypoints, conf.descriptor_type_str, frame_descriptors);

                // Match the current frame with each model and compute the rank
                priority_queue<PQRank<float>, vector<PQRank<float> >, PQRank<float> > scores;
                for (int k = 0; k < models.size(); ++k) {
                    PQRank<float> rank_elem;
                    rank_elem.score = models[k].match(conf, frame_descriptors, frame_pose, frame_confidences,
                                                      true, timing);
                    rank_elem.class_idx = k;
                    scores.push(rank_elem);
                }

                // Update all rates
                // The resulting rates are used to compute the CMC curve
                int rank_idx = get_rank_index<float>(scores, i);
                for (int k = rank_idx; k < rates.cols; ++k) {
                    rates.at<float>(0, k)++;
                }
            }
        } // end foreach

        for (int j = 0; j < rates.cols; ++j)
            rates.col(j).at<float>(0) /= tot_imgs;

        CMC += rates;

        cout << "rates: " << rates << endl;
        cout << "><><><><><><><><><><><><><><><><><><><><><><><><><><" << endl << endl;
        rounds++;
    }

    CMC /= (rounds - 1);
    cout << "CMC: " << CMC << endl;

    float nAUC = area_under_curve(CMC);
    cout << "nAUC: " << nAUC * 100 << endl;

    timing.show();

    // Saving results
    timing.write(conf.res_file_path + "timing/", "t" + get_res_filename(conf));
    print_cmc_nauc(conf.res_file_path, get_res_filename(conf),
                   conf.descriptor_type_str, CMC, nAUC);

    return 0;
}