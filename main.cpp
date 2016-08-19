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

#include <opencv/cvaux.h>
#include "MultiviewBodyModel.h"

using namespace multiviewbodymodel;
using namespace cv;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::priority_queue;


//void test_match();

int main(int argc, char **argv) {

    // Load settings: conf will contains all basic
    // information to run the program
    Configuration conf;
    parse_args(argc, argv, conf);
    read_config_file(conf);
    conf.show();

    // Load paths from the configuration parameters and
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

    // Models loading: the # of poses chosen from the settings are loaded
    // in each model for each round
    int rounds = 1;
    while (rounds <= models_per_person) {
        // Create empty MultiviewBodyModel object to store information
        // contained in the model set
        vector<MultiviewBodyModel> models;
        empty_models(static_cast<int>(models_set.size()), models);

        printf("------------- %s Models loaded, rounds: %d -------------\n",
               conf.descriptor_type_str.c_str(), rounds);

        // Creates the models for each person defined in the configuration settings
        init_models(conf, rounds, imgs_paths, skels_paths, models_set, models, timing);

        // Rates of the current test image
        Mat rates;
        rates = Mat::zeros(1, static_cast<int>(models_set.size()), CV_32F);

        // Foreach image in the data set do
        for (int i = 0; i < skels_paths.size(); ++i) {
            for (int j = 0; j < skels_paths[i].size(); ++j) {
                Mat frame_descriptors;
                vector<KeyPoint> frame_keypoints;
                vector<float> frame_confidences;
                int frame_pose;

                read_skel_file(skels_paths[i][j], conf.keypoint_size, frame_keypoints, frame_confidences, frame_pose);
                compute_descriptors(imgs_paths[i][j], frame_keypoints, conf.descriptor_type, frame_descriptors);

                // Match the current frame with each model and compute the rank
                priority_queue<PQRank<float>, vector<PQRank<float> >, PQRank<float> > scores;
                for (int k = 0; k < models.size(); ++k) {
                    PQRank<float> rank_elem;

                    rank_elem.score = models[k].match(conf, frame_descriptors, frame_pose, frame_confidences, timing);
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
    timing.write(conf);
    write_cmc_nauc(conf, CMC, nAUC);

    return 0;
}

//void test_match() {
//
//    vector<float> confidences1, confidences2;
//    int ps1, ps2;
//
//    const int size = 6;
//    // BACK
////    string path1 = "../ds/gianluca_sync/c00000";
////    string paths[size] = {"../ds/gianluca_sync/r00031", "../ds/marco_sync/r00051", "../ds/nicola_sync/c00000",
////                          "../ds/stefanog_sync/r00061", "../ds/stefanom_sync/r00031", "../ds/matteol_sync/r00064"};
//////    string paths[size] = {"../ds/gianluca_sync/r00031", "../ds/gianluca_sync/r00026", "../ds/gianluca_sync/r00056",
//////                          "../ds/gianluca_sync/l00047","../ds/gianluca_sync/l00052","../ds/gianluca_sync/c00052"};
////    float weights_data[15] = {1, 1, 1, 1.1, 0,
////                              1.2, 1.1, 0, 2, 0,
////                              0, 0, 0, 0, 0};
//    // FRONT
////    string path1 = "../ds/gianluca_sync/c00011";
////    string paths[size] = {"../ds/gianluca_sync/c00015", "../ds/marco_sync/c00010", "../ds/nicola_sync/c00012",
////                          "../ds/stefanog_sync/l00050", "../ds/stefanom_sync/l00017", "../ds/matteol_sync/r00055"};
//////        string paths[size] = {"../ds/gianluca_sync/c00013", "../ds/gianluca_sync/c00040", "../ds/gianluca_sync/c00044",
//////                          "../ds/gianluca_sync/r00014","../ds/gianluca_sync/c00067","../ds/gianluca_sync/l00037"};
////    float weights_data[15] = {1.5,1.5,1.2,1,0,
////                             1.2,1,0,2,1.2,
////                              1.1,0,1.2,1.1,0};
////    // LEFT
//    string path1 = "../ds/gianluca_sync/c00005";
//    string paths[size] = {"../ds/gianluca_sync/c00036", "../ds/marco_sync/c00027", "../ds/nicola_sync/r00031",
//                          "../ds/stefanog_sync/l00046", "../ds/stefanom_sync/r00020", "../ds/matteol_sync/l00045"};
////        string paths[size] = {"../ds/gianluca_sync/c00031", "../ds/gianluca_sync/l00004", "../ds/gianluca_sync/l00026",
////                          "../ds/gianluca_sync/r00038","../ds/gianluca_sync/r00012","../ds/gianluca_sync/c00032"};
//    float weights_data[15] = {1.5,1.2,1.1,1.1,1.1,
//                              0,0,0,0,1,
//                              1,1,0,1,1};
//
//    vector<cv::KeyPoint> keypoints1;
//    read_skel_file(path1 + "_skel.txt", 3, keypoints1, confidences1, ps1);
//    Mat img1 = imread(path1 + ".png");
//    Mat descriptors1;
//    cv::SiftDescriptorExtractor extractor;
//    extractor.compute(img1, keypoints1, descriptors1);
//
//
//    for (int k = 0; k < size; ++k) {
//        vector<KeyPoint> keypoints;
//        read_skel_file(paths[k] + "_skel.txt", 3, keypoints, confidences2, ps2);
//        Mat img = imread(paths[k] + ".png");
//        Mat descriptors;
//
//        extractor.compute(img, keypoints, descriptors);
//        cv::BFMatcher matcher(NORM_L2, true);
//        vector<DMatch> matches;
//        matcher.match(descriptors1, descriptors, matches);
//
//
//        float min = 1000;
//
//        for (int j = 0; j < matches.size(); ++j) {
//            float dist = matches[j].distance;
//            if (dist < min)
//                min = dist;
//        }
//
//        cout << path1 << "vs" << paths[k] << endl;
//
//        double avg_matches = 0.0;
//        int cnt_matches = 0;
//        vector<DMatch> good;
//        for (int i = 0; i < matches.size(); ++i) {
//            if (matches[i].queryIdx == matches[i].trainIdx && matches[i].distance < 3 * min) {
//                avg_matches += matches[i].distance;
//                cnt_matches++;
//
//                good.push_back(matches[i]);
//            }
//        }
//
//        avg_matches /= cnt_matches;
//
//        double score = 0.0;
//        double sum_W = 0.0;
//        for (int i = 0; i < matches.size(); ++i) {
//            if (matches[i].queryIdx == matches[i].trainIdx && matches[i].distance < 3 * min) {
//                double w = weights_data[i];
//                double diff = std::abs(matches[i].distance - avg_matches);
//                cout << i << ": " << matches[i].queryIdx << " " << matches[i].trainIdx << " " << diff << endl;
//                score += w * (1 / diff);
//                sum_W += w;
//            }
//        }
//
//        for (int j = 0; j < descriptors1.rows; ++j) {
//            double w = weights_data[j];
//            score += w * 1 / std::abs(norm(descriptors1, descriptors, NORM_L2) - avg_matches);
//            sum_W += w;
//        }
//
//        score /= sum_W;
////        score *= cnt_matches;
//
//
//
//        cout << "score = " << 1 / score << endl << endl;
//
//        namedWindow("Results", 1);
//
//        Mat img_matches;
//        drawMatches(img1, keypoints1, img, keypoints, good, img_matches);
//        imshow("Results", img_matches);
//
//
//        waitKey(0);
//
//    }
//
//
//
////    FlannBasedMatcher matcher;
//
//
//
//
//
//
//
//
//}
