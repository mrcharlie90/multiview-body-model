//
// Created by Mauro on 10/05/16.
//

#include "MultiviewBodyModel.h"




using namespace multiviewbodymodel;


int main(int argc, char** argv)
{
    // This will containg all information regarding file paths function parameters
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


    // Used for storing a set of masks in which one of them marks (with 0 or 1) the pose
    // already chosen for the body model loading phase
    vector<Mat> masks;
    for (int j = 0; j < train_imgs_paths.size(); ++j) {
        Mat mask;
        mask = Mat::zeros(static_cast<int>(train_imgs_paths[j].size()), 1, CV_8U);
        masks.push_back(mask);
    }



    // Body Models used for testing
    vector<MultiviewBodyModel> models;

    // Cumulative Matching Characteristic curve:
    // contains the average person re-identification rate
    Mat CMC;
    CMC = Mat::zeros(1, static_cast<int>(conf.persons_names.size()), CV_32F);

    // Used for performace logging
    Timing timing;
    timing.enable();

    // Number of cycles
    int rounds = 1;

    // Initial timing variable
    double t0 = (timing.enabled) ? (double)getTickCount() : 0;

    // Generate random indeces for testing
    vector<vector<int> > map;
    get_poses_map(train_skels_paths, map);

    // Choose a random number of images uniformly
    vector<vector<int> > indeces;
    int tot_test_imgs = get_rnd_indeces(map, indeces);


    while(load_training_set(conf, masks, train_skels_paths, train_imgs_paths, models, timing) && rounds < 3) {

        printf("----------- Models loaded, rounds: %d -----------\n", rounds);

        // Rates of the current test image
        Mat current_rates;
        current_rates = Mat::zeros(1, static_cast<int>(conf.persons_names.size()), CV_32F);

        // Match
        for (int current_class = 0; current_class < indeces.size(); ++current_class) {
            for (int i = 0; i < indeces[current_class].size(); ++i) {
                // Testing variables
                Mat test_image;
                Mat test_descriptors;
                vector<KeyPoint> test_keypoints;
                vector<float> test_confidences;
                int test_pose_side;

                // Loading the test frame skeleton
                read_skel(conf,
                          train_skels_paths[current_class][i],
                          train_imgs_paths[current_class][i], test_image, test_keypoints,
                          test_confidences, test_descriptors, test_pose_side, timing);

                // Compute the match score between the query image and each model loaded
                // the result is inserted into the priority queue "scores"
                priority_queue< RankElement<float>, vector<RankElement<float> >, RankElement<float> > scores;
                for (int j = 0; j < models.size(); ++j) {
                    RankElement<float> rank_element;
                    rank_element.score = models[j].match(test_descriptors, test_confidences, test_pose_side);
                    rank_element.classIdx = j;
                    scores.push(rank_element);
                }

                // All rates starting from rank_idx to the number of training models are updated
                // The results are used for computing the CMC curve
                int rank_idx = get_rank_index<float>(scores, current_class);
                for (int k = rank_idx; k < current_rates.cols; ++k) {
                    current_rates.at<float>(0, k)++;
                }
            }
        }

        // Divide each element with the total number of test images
        for (int i = 0; i < current_rates.cols; ++i) {
            current_rates.at<float>(0, i) /= tot_test_imgs;
        }

        // Add values to the CMC
        CMC += current_rates;

        cout << "current_rates: " << current_rates << endl;
        printf("------------------------------------------------\n\n");

        rounds++;
        models.clear();
    }

    // Log performance
    if (timing.enabled) {
        double time = ((double)getTickCount() - t0) / getTickFrequency();
        // Averaged matching time
        timing.t_tot_round += time;
        timing.n_rounds++;

        // Overall matching time
        timing.t_tot_matching += time;
    }
    timing.write();

    // Compute the average
    CMC /= (rounds - 1);
    cout << "CMC: " << CMC << endl;

    // Save the results
    stringstream ss;
    ss << "../CMC_" << conf.descriptor_extractor_type << "_RND" << endl;
    saveCMC(ss.str(), CMC);

    return 0;
}