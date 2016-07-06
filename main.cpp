// Copyright (c) [2016] [Mauro Piazza]
//
//          IASLab License
//
// Main used for testing the MultiviewMBodyModel class in particular the
// accuracy of the matching function used.

#include "MultiviewBodyModel.h"

using namespace multiviewbodymodel;
using std::vector;
using std::cout;
using std::endl;
using cv::string;



void save_mask(string d_name, vector<cv::Mat> masks) {
    std::ofstream file("masks");

    file << "d_name" << endl;
    for (int i = 0; i < masks.size(); ++i) {
        assert(masks[i].rows == 1);

        file << masks[i] << endl;
    }
    file.close();
}

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


    // Used for performace logging
    Timing timing;
    timing.enable();

    // Number of cycles
    int rounds = 1;

    // Initial timing variable
    double t0 = (timing.enabled) ? (double)cv::getTickCount() : 0;

    // Generate random indices for testing
    vector<vector<int> > map;
    get_poses_map(train_skels_paths, map);

    // Choose a random number of images uniformly
    vector<vector<int> > indices;
    int tot_test_imgs = get_rnd_indices(map, indices);

    for (int d = 0; d < conf.descriptor_extractor_type.size(); ++d) {

        // Used for storing a set of masks in which one of them marks (with 0 or 1) the pose
        // already chosen for the body model loading phase
        vector<cv::Mat> masks;
        for (int j = 0; j < train_imgs_paths.size(); ++j) {
            cv::Mat mask;
            mask = cv::Mat::zeros(1, static_cast<int>(train_imgs_paths[j].size()), CV_8S);
            masks.push_back(mask);
        }



        // Cumulative Matching Characteristic curve:
        // contains the average person re-identification rate
        cv::Mat CMC;
        CMC = cv::Mat::zeros(1, static_cast<int>(conf.persons_names.size()), CV_32F);

        // Body Models used for testing
        vector<MultiviewBodyModel> models;

        while(load_models(conf.descriptor_extractor_type[d], conf.keypoint_size, conf.max_poses,
                          masks, train_skels_paths, train_imgs_paths, models, timing)) {

            if (rounds % 10 == 0)
            {
                for (int i = 0; i < masks.size(); ++i) {
                    cout << "[" << (int)masks[i].at<uchar>(0, 0);
                    for (int j = 1; j < masks[i].cols; ++j) {
                        cout << " " << (int)masks[i].at<char>(0, j);
                    }
                    cout << "]" << endl;
                }
            }

            cout << "----------- " <<  conf.descriptor_extractor_type[d] <<
                    " Models loaded, rounds: " << rounds << " ---------" << endl;
            // Rates of the current test image
            cv::Mat rates;
            rates = cv::Mat::zeros(1, static_cast<int>(conf.persons_names.size()), CV_32F);

            // Match
            for (int current_class = 0; current_class < indices.size(); ++current_class) {
                for (int i = 0; i < indices[current_class].size(); ++i) {
                    // Testing variables
                    cv::Mat test_image;
                    cv::Mat test_descriptors;
                    vector<cv::KeyPoint> test_keypoints;
                    vector<float> test_confidences;
                    int test_pose_side;

                    // Loading the test frame skeleton
                    read_skel(conf.descriptor_extractor_type[d], conf.keypoint_size,
                              train_skels_paths[current_class][i], train_imgs_paths[current_class][i],
                              test_image, test_keypoints, test_confidences, test_descriptors, test_pose_side, timing);

                    // Compute the match score between the query image and each model loaded
                    // the result is inserted into the priority queue "scores"
                    std::priority_queue<RankElement<float>, vector<RankElement<float> >, RankElement<float> > scores;
                    for (int j = 0; j < models.size(); ++j) {
                        RankElement<float> rank_element;
                        rank_element.score = models[j].Match(test_descriptors, test_confidences, test_pose_side);
                        rank_element.classIdx = j;
                        scores.push(rank_element);
                    }

                    // All rates starting from rank_idx to the number of training models are updated
                    // The results are used for computing the CMC curve
                    int rank_idx = get_rank_index<float>(scores, current_class);
                    for (int k = rank_idx; k < rates.cols; ++k) {
                        rates.at<float>(0, k)++;
                    }
                }
            }

            // Divide each element with the total number of test images
            for (int i = 0; i < rates.cols; ++i) {
                rates.at<float>(0, i) /= tot_test_imgs;
            }

            // Add values to the CMC
            CMC += rates;

            cout << "rates: " << rates << endl;
            cout << "><><><><><><><><><><><><><><><><><><><><><><><><><><" << endl << endl;
            rounds++;
            models.clear();

            // Log performance
            if (timing.enabled) {
                double time = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
                // Averaged matching time
                timing.t_tot_round += time;
                timing.n_rounds++;

                // Overall matching time
                timing.t_tot_matching += time;


            }
        } // end-while

        if (timing.enabled) {
            timing.descriptor_names.push_back(conf.descriptor_extractor_type[d]);
            timing.t_descriptor_names.push_back(timing.t_tot_matching);
        }



        // Compute the average
        CMC = CMC / (rounds - 1);
        cout << "CMC: " << CMC << endl;

        // Save the results
        std::stringstream ss;
        ss << "../CMC_" << conf.descriptor_extractor_type[d] << "_RND" << endl;
        saveCMC(ss.str(), CMC);

        rounds = 1;

        // Saving the mask
        save_mask(conf.descriptor_extractor_type[d], masks);
        cout << "Dataset %" << endl;
        for (cv::Mat mask : masks) {
            cout << (float)countNonZero(mask.row(0)) / (float)mask.cols << " ";
        }
        cout << endl;

        for(cv::Mat mask : masks) {
            cout << mask << endl;
        }

    } // end-for

    if (timing.enabled)
        timing.write();

    return 0;
}