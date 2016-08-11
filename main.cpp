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
using namespace cv;
using multiviewbodymodel::PQRank;


int main(int argc, char** argv)
{
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

    // Load the training set
    vector<vector<string> > imgs_paths;
    vector<vector<string> > skels_paths;

    int tot_imgs = load_person_imgs_paths(conf, imgs_paths, skels_paths);

    const int k_model_set_size = 13;

    vector<vector<int> > models_set;
    int models_per_person = load_models_set(conf.poses, imgs_paths, skels_paths,
                                            k_model_set_size, 13, models_set);
    Timing timing;

    vector<MultiviewBodyModel> models;
    empty_models(models_set.size(), models);

    int rounds = 1;
    while (rounds <= models_per_person) {
        printf("------------- %s Models loaded, rounds: %d -------------\n",
               conf.descriptor_type_str.c_str(), rounds);
        init_models(conf, rounds, imgs_paths, skels_paths, models_set, models, timing);


        // Rates of the current test image
        Mat rates;
        rates = Mat::zeros(1, static_cast<int>(conf.persons_names.size()), CV_32F);

        Mat W;
        W = Mat::ones(4, 15, CV_8S);

        // foreach image in the data set do
        for (int i = 0; i < skels_paths.size(); ++i) {
            for (int j = 0; j < skels_paths[i].size(); ++j) {
                Mat descriptors;
                vector<KeyPoint> keypoints;
                vector<float> confidences;
                int ps;

                read_skel_file(skels_paths[i][j], conf.keypoint_size, keypoints, confidences, ps);
                compute_descriptors(imgs_paths[i][j], keypoints, conf.descriptor_type_str, descriptors);

                // Match the current frame with each model and compute the rank
                priority_queue<PQRank<float>, vector<PQRank<float> >, PQRank<float> > scores;
                for (int k = 0; k < models.size(); ++k) {
                    PQRank<float> rank_elem;
                    rank_elem.score = models[k].match(descriptors, ps, confidences, true, conf, timing);
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


        for (int j = 0; j < rates.cols; ++j) {
            rates.at<float>(0, j) /= tot_imgs;
        }

        cout << "rates: " << rates << endl;
        cout << "><><><><><><><><><><><><><><><><><><><><><><><><><><" << endl << endl;
        rounds++;
    }

    return 0;
}