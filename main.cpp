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

    // Used for performace logging
    Timing timing;
    timing.enable();

    // Number of cycles
    int rounds = 1;

    // Initial timing variable
    double t0 = (timing.enabled) ? (double)getTickCount() : 0;

    Mat m = compute_increment_matrix(train_skels_paths, conf.num_images,
                                     train_skels_paths.size(), conf.views_names.size(), conf.max_poses);

    for (int k = 0; k < train_skels_paths.size(); ++k) {
        cout << "person " << k << endl;
        for (int i = 0; i < conf.views_names.size(); ++i) {
            for (int j = 0; j < conf.max_poses; ++j) {
                cout << m.at<float>(i, j, k) << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    exit(0);
    while(load_training_set(conf, masks, train_skels_paths, train_imgs_paths, models, timing) && rounds < 3) {

        printf("----------- Models loaded, rounds: %d -----------\n", rounds);

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
            read_skel(conf,
                      test_skels_paths[j],
                      test_imgs_paths[j], test_image, test_keypoints,
                      test_confidences, test_descriptors, test_pose_side, timing);

            // Compute the match score between the query image and each model loaded
            // the result is inserted into the priority queue "scores"
            priority_queue< RankElement<float>, vector<RankElement<float> >, RankElement<float> > scores;
            for (int k = 0; k < models.size(); ++k) {
                RankElement<float> rank_element;
                rank_element.score = models[k].match(test_descriptors, test_confidences, test_pose_side);
                rank_element.classIdx = k;
                scores.push(rank_element);
            }




            test_class_counter++;
//            cout << "done! class = " << query_class << " rank_idx = " << rank_idx << endl;
        }



        rounds++;
        models.clear();


    }

    return 0;
}




//int main(int argc, char** argv)
//{
//    // This will containg all information regarding file paths function parameters
//    Configuration conf;
//
//    if (argc < 7) {
//        cout << "USAGE: multiviewbodymodel -c <configfile> -d <descriptortype> -k <keypointsize>";
//        exit(-1);
//    }
//    else {
//        parse_args(argc, argv, conf);
//        conf.show();
//    }
//
//    // Load the training set
//    vector<vector<string> > train_imgs_paths;
//    vector<vector<string> > train_skels_paths;
//    load_train_paths(conf, train_skels_paths, train_imgs_paths);
//
//    // Load queries
//    vector<string> test_imgs_paths;
//    vector<string> test_skels_paths;
//    load_test_paths(train_skels_paths, train_imgs_paths,
//                    test_skels_paths, test_imgs_paths);
//
//    assert(test_imgs_paths.size() == test_skels_paths.size());
//
//    // Used for storing a set of masks in which one of them marks (with 0 or 1) the pose
//    // already chosen for the body model loading phase
//    vector<Mat> masks;
//    for (int j = 0; j < train_imgs_paths.size(); ++j) {
//        Mat mask;
//        mask = Mat::zeros(static_cast<int>(train_imgs_paths[j].size()), 1, CV_8U);
//        masks.push_back(mask);
//    }
//
//    // Overall number of test images for each person
//    // note: +1 because image's names start from 0
//    vector<int> num_test_images;
//    num_test_images.push_back( (conf.num_images.at<uchar>(0, 0) + 1) * static_cast<int>(conf.views_names.size()) );
//    num_test_images.push_back( (conf.num_images.at<uchar>(1, 0) + 1) * static_cast<int>(conf.views_names.size()) );
//    num_test_images.push_back( (conf.num_images.at<uchar>(2, 0) + 1) * static_cast<int>(conf.views_names.size()) );
//    int tot_test_images = ((static_cast<int>(sum(conf.num_images)[0])) + 3) * static_cast<int>(conf.views_names.size());
//
//    // Body Models used for testing
//    vector<MultiviewBodyModel> models;
//
//    // Cumulative Matching Characteristic curve:
//    // contains the average person re-identification rate
//    Mat CMC;
//    CMC = Mat::zeros(1, static_cast<int>(conf.persons_names.size()), CV_32F);
//
//    // Used for performace logging
//    Timing timing;
//    timing.enable();
//
//    // Number of cycles
//    int rounds = 1;
//
//    // Initial timing variable
//    double t0 = (timing.enabled) ? (double)getTickCount() : 0;
//
//    while(load_training_set(conf, masks, train_skels_paths, train_imgs_paths, models, timing) && rounds < 3) {
//
//        printf("----------- Models loaded, rounds: %d -----------\n", rounds);
//
//        // Rates of the current test image
//        Mat current_rates;
//        current_rates = Mat::zeros(1, static_cast<int>(conf.persons_names.size()), CV_32F);
//
//        // Variables used for computing the current query frame class
//        int test_class = 0;
//        int test_class_counter = 0;
//
//        for (int j = 0; j < test_imgs_paths.size(); ++j) {
//            // Check if the test frames relative to one subject are all extracted
//            if (test_class_counter != 0 && test_class_counter % num_test_images[test_class] == 0) {
//                // Update the class and reset the counter
//                test_class++;
//                test_class_counter = 0;
//            }
//
//            // Testing variables
//            Mat test_image;
//            Mat test_descriptors;
//            vector<KeyPoint> test_keypoints;
//            vector<float> test_confidences;
//            int test_pose_side;
//
//            // Loading the test frame skeleton
//            read_skel(conf,
//                      test_skels_paths[j],
//                      test_imgs_paths[j], test_image, test_keypoints,
//                      test_confidences, test_descriptors, test_pose_side, timing);
//
//            // Compute the match score between the query image and each model loaded
//            // the result is inserted into the priority queue "scores"
//            priority_queue< RankElement<float>, vector<RankElement<float> >, RankElement<float> > scores;
//            for (int k = 0; k < models.size(); ++k) {
//                RankElement<float> rank_element;
//                rank_element.score = models[k].match(test_descriptors, test_confidences, test_pose_side);
//                rank_element.classIdx = k;
//                scores.push(rank_element);
//            }
//
//            // All rates starting from rank_idx to the number of training models are updated
//            // The results are used for computing the CMC curve
//            int rank_idx = get_rank_index<float>(scores, test_class);
//            for (int c = rank_idx; c < current_rates.cols; ++c) {
//                current_rates.at<float>(0, c)++;
//            }
//
//            test_class_counter++;
////            cout << "done! class = " << query_class << " rank_idx = " << rank_idx << endl;
//        }
//
//        for (int i = 0; i < current_rates.cols; ++i) {
//            current_rates.at<float>(0, i) /= tot_test_images;
//        }
//
//        CMC += current_rates;
//
//        cout << "current_rates: " << current_rates << endl;
//        printf("------------------------------------------------\n\n");
//
//        rounds++;
//        models.clear();
//
//
//    }
//
//    CMC /= (rounds-1);
//    cout << "CMC: " << CMC << endl;
//
//    if (timing.enabled) {
//        double time = ((double)getTickCount() - t0) / getTickFrequency();
//        // Averaged matching time
//        timing.t_tot_round += time;
//        timing.n_rounds++;
//
//        // Overall matching time
//        timing.t_tot_matching += time;
//    }
//
//
//
//    timing.write();
//    return 0;
//}
//

