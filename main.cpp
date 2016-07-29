// Copyright (c) [2016] [Mauro Piazza]
//
//          IASLab License
//
// Main used for testing the MultiviewMBodyModel class in particular the
// accuracy of the matching function used.
//

#include "MultiviewBodyModel.h"

using namespace multiviewbodymodel;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using cv::string;
using multiviewbodymodel::RankElement;

void test_replace();

void test_memory_usage();

int main(int argc, char** argv)
{

    Configuration conf;

    if (argc < 7) {
        show_help();
    }
    else {
        parse_args(argc, argv, conf);
        conf.show();
    }

    // Load the training set
    vector<vector<string> > train_imgs_paths;
    vector<vector<string> > train_skels_paths;
    load_train_paths(conf, train_skels_paths, train_imgs_paths);

    // Used for performance logging
    Timing timing;
    timing.enable();

    // Initial matching time variable
    double t0_match = (timing.enabled) ? (double)cv::getTickCount() : 0;

    // Generate a map for the current dataset
    vector<vector<int> > map;
    get_poses_map(train_skels_paths, map);

    // Choose a random number of test images uniformly
    vector<vector<int> > rnd_indices;
    int tot_test_imgs = get_rnd_indices(map, conf.max_poses, rnd_indices);

    // when -L2 or -H flag is passed, a set of descriptors is computed
    // (Euclidean distance ones and Hamming distance ones respectively)
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

        // Number of cycles
        int rounds = 1;

        // Body Models used for testing
        vector<MultiviewBodyModel> models;
        while(load_models(train_skels_paths, train_imgs_paths, conf.descriptor_extractor_type[d], conf.keypoint_size[d],
                          conf.max_poses, masks, models, timing)) {

            double t0_round = (timing.enabled) ? (double)cv::getTickCount() : 0;

            cout << "----------- " <<  conf.descriptor_extractor_type[d] <<
                    " Models loaded, rounds: " << rounds << " ---------" << endl;

            // Rates of the current test image
            cv::Mat rates;
            rates = cv::Mat::zeros(1, static_cast<int>(conf.persons_names.size()), CV_32F);

            // Match: extract indices
            for (int current_class = 0; current_class < rnd_indices.size(); ++current_class) { // for each person
                for (int i = 0; i < rnd_indices[current_class].size(); ++i) { // for each frame
                    // Required variables
                    cv::Mat test_image;
                    cv::Mat test_descriptors;
                    vector<cv::KeyPoint> test_keypoints;
                    vector<float> test_confidences;
                    int test_pose_side;

                    // Load the test frame skeleton
                    int image_idx = rnd_indices[current_class][i];

                    load_test_skel(train_skels_paths[current_class][image_idx], train_imgs_paths[current_class][image_idx],
                                   conf.descriptor_extractor_type[d], conf.keypoint_size[d], test_image, test_keypoints,
                                   test_confidences, test_descriptors, test_pose_side, timing);

                    // Compute the matching score between the test image and each model
                    // the result is inserted into the priority queue named "scores"
                    std::priority_queue<RankElement<float>, vector<RankElement<float> >, RankElement<float> > scores;
                    for (int j = 0; j < models.size(); ++j) {
                        RankElement<float> rank_element;
                        rank_element.score = models[j].Match(test_descriptors, test_confidences, test_pose_side,
                                                             conf.occlusion_search, conf.norm_type, timing);
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
                double match_time = (cv::getTickCount() - t0_match) / cv::getTickFrequency();
                double round_time = (cv::getTickCount() - t0_round) / cv::getTickFrequency();
                // Averaged matching time
                timing.t_rounds += round_time;
                timing.n_rounds++;

                timing.t_tot_exec = match_time;
            }
        } // end-while

        // Compute the average
        CMC /= (rounds - 1);
        cout << "CMC: " << CMC << endl;

        // Compute nAUC
        float nAUC = 0;
        for (int c  = 0; c < CMC.cols - 1; ++c) {
            nAUC += (CMC.at<float>(0, c) + CMC.at<float>(0, c + 1));
        }
        nAUC /= 2;
        nAUC /= CMC.cols; // normalize
        cout << "nAUC: " << nAUC * 100 << endl;

        cout << "Tot time: " << timing.t_tot_exec << endl;

        // Save the results
        std::stringstream ss;
        ss << conf.res_file_path
        << "CMC_" << conf.descriptor_extractor_type[d]
        << "_N" << conf.persons_names.size()
        << "_PS" << conf.max_poses
        << "_K" << conf.keypoint_size[d]
        << "_O" << conf.occlusion_search << "_RND";
        cmc2dat(ss.str(), CMC, nAUC);
        ss.str("");

        ss << conf.res_file_path
        << "TIME_" << conf.descriptor_extractor_type[d]
        << "_N" << conf.persons_names.size()
        << "_PS" << conf.max_poses
        << "_K" << conf.keypoint_size[d]
        << "_O" << conf.occlusion_search << "_RND";

        if (timing.enabled)
            timing.write(ss.str());
        ss.str("");

        ss << conf.res_file_path << "rank1_N" << conf.persons_names.size() << "_PS" << conf.max_poses << "_O" << conf.occlusion_search;
        rank1_append_results(ss.str(), conf.descriptor_extractor_type[d], CMC);
        ss.str("");

        ss << conf.res_file_path << "nauc_N" << conf.persons_names.size() << "_PS" << conf.max_poses << "_O" << conf.occlusion_search;
        nauc_append_result(ss.str(), conf.descriptor_extractor_type[d], nAUC);
        ss.str("");

        print_dataset_usage(masks);
    }

    return 0;
}

void test_replace() {
    // Testing replace function
    MultiviewBodyModel mbm(4);
    Timing t;

    mbm.ReadAndCompute("../ds/gianluca_sync/c00000_skel.txt",
                       "../ds/gianluca_sync/c00000.png", "SURF", 9, t);
    mbm.ReadAndCompute("../ds/gianluca_sync/c00004_skel.txt",
                       "../ds/gianluca_sync/c00004.png", "SURF", 9, t);
    mbm.ReadAndCompute("../ds/gianluca_sync/c00009_skel.txt",
                       "../ds/gianluca_sync/c00009.png", "SURF", 9, t);
    mbm.ReadAndCompute("../ds/gianluca_sync/c00016_skel.txt",
                       "../ds/gianluca_sync/c00016.png", "SURF", 9, t);

    cv::Mat img = cv::imread("../ds/gianluca_sync/c00005.png");
    cout << mbm.Replace("../ds/gianluca_sync/c00005_skel.txt", img, "SURF", 9) << endl;

    exit(0);
}

void test_memory_usage() {
    // Testing replace function
    MultiviewBodyModel mbm(1);
    Timing t;
    mbm.ReadAndCompute("../ds/gianluca_sync/c00000_skel.txt",
                       "../ds/gianluca_sync/c00000.png", "SURF", 9, t);
    mbm.ReadAndCompute("../ds/gianluca_sync/c00001_skel.txt",
                       "../ds/gianluca_sync/c00001.png", "SURF", 9, t);
    mbm.ReadAndCompute("../ds/gianluca_sync/c00004_skel.txt",
                       "../ds/gianluca_sync/c00004.png", "SURF", 9, t);
    mbm.ReadAndCompute("../ds/gianluca_sync/c00009_skel.txt",
                       "../ds/gianluca_sync/c00009.png", "SURF", 9, t);
    mbm.ReadAndCompute("../ds/gianluca_sync/c00016_skel.txt",
                       "../ds/gianluca_sync/c00016.png", "SURF", 9, t);

    cout << mbm.size() << endl;
    cout << mbm.get_size_of() << endl;
    cout << sizeof(MultiviewBodyModel) << endl;

    // MultiviewBodyModel: 128 byte
    // size: 1 => 248 byte
    // size: 2 => 492 byte
    // size: 3 => 736 byte
    // size: 4 => 980 byte
    // size: 5 => 1224 byte

    exit(0);
}