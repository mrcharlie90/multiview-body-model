//
// Created by Mauro on 15/04/16.
//

#ifndef MULTIVIEWBODYMODEL_MULTIVIEWBODYMODEL_H
#define MULTIVIEWBODYMODEL_MULTIVIEWBODYMODEL_H

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <numeric>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/nonfree.hpp>

namespace  multiviewbodymodel {
    using namespace cv;
    using namespace std;

    /**
     * Used for performance logging
     */
    struct Timing {

        bool enabled;

        Timing() {
            enabled = false;
        }

        void enable() {
            enabled = true;
        }

        // Average time for loading the training set
        double t_tot_load_training_set;
        int n_tot_load_training_set;

        // Average time for one round in the main while
        double t_tot_round;
        int n_rounds;

        // Average time to compute the descriptors
        double t_tot_descriptors;
        int n_tot_descriptors;

        // Average Time to load one model
        double t_tot_model_loading;
        int n_tot_model_loading;

        // Average time to compute the skeleton descriptors
        double t_tot_skel_loading;
        int n_tot_skel_loading;

        // Overall time to compute the matching
        double t_tot_matching;

        // Overall times
        vector<double> t_descriptor_names;
        vector<string> descriptor_names;

        void write();
        void show();
    };

    /*
     *  Data structure containing information about skeletal tracker keypoints descriptors.
     *  A distance function is developed to comoute the distance between pairs of views
     *  of the same angle. The views must be inserted with the same order in every body model.
     *
     *  Set timing NULL for no performance logging
     */
    class MultiviewBodyModel {
    public:
        MultiviewBodyModel(int max_poses);
        bool ReadAndCompute(string file_path, string img_path, string descriptor_extractor_type, int keypoint_size, Timing &timing);
        float match(Mat query_descriptors, vector<float> query_confidences, int query_pose_side, bool occlusion_search=true);
        bool ready();

    private:
        // Used to create a confidences mask used for computing the matches
        void create_confidence_mask(vector<float> &query_confidences, vector<float> &train_confidences, vector<char> &out_mask);

        // Search for a descriptor of a specified keypoint occluded in one pose of the model
        bool  get_descriptor_occluded(int keypoint_index, Mat &descriptor_occluded);

        // The model will reach the ready state when the the total
        // number of poses acquired is equal to max_poses.
        int max_poses_;

        // Pose number (i.e. 1:front, 2:back, 3:left-side, 4:right-side )
        vector<int> pose_side_;

        // Contains the keypoint's descriptors of each view
        vector<cv::Mat> views_descriptors_;

        // Vector containing all keypoints for each image
        vector<vector<cv::KeyPoint> > views_keypoints_;

        vector<Mat> views_images_;

        // Confidence value for each keypoint of each view. A value between 0 and 1
        // (temporary 1: keypoint visible, 0: keypoint is occluded))
        vector<vector<float> > views_descriptors_confidences_;
    };



    /**
     * <><><><><><><><><> Main functions declarations <><><><><><><><><><><><><><>
     */

    template<typename T>
    void print_list(vector<T> vect);

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

    template<typename T>
    void print_list(priority_queue<RankElement<T>, vector<RankElement<T> >, RankElement<T> > queue);

    
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
        vector<string> descriptor_extractor_type;
        int keypoint_size;

        void show();
    };

    void load_train_paths(string main_path, vector<string> persons_names, vector<string> views_names,
                          Mat num_images,
                          vector<vector<string> > &imgs_paths, vector<vector<string> > &skel_paths);

    void load_train_paths(Configuration conf, vector<vector<string> > &skels_paths,
                          vector<vector<string> > &imgs_paths);

    bool load_training_set(string descriptor_extractor_type, int keypoint_size, int max_poses, vector<Mat> &masks,
                           vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                           vector<MultiviewBodyModel> &out_models, Timing &timing);

    bool load_training_set(Configuration conf, vector<Mat> &masks, vector<vector<string> > &train_skels_paths,
                           vector<vector<string> > &train_imgs_paths, vector<MultiviewBodyModel> &out_models,
                           Timing &timing);

    void read_skel(string descriptor_extractor_type, int keypoint_size, string skel_path, string img_path,
                   Mat &out_image, vector<KeyPoint> &out_keypoints, vector<float> &out_confidences,
                   Mat &out_descriptors, int &out_pose_side, Timing &timing);


    void check_sequence(FileNode fn);
    void parse_args(int argc, char **argv, Configuration &out_conf);

    int get_pose_side(string path);

    void get_poses_map(vector<vector<string> > train_paths, vector<vector<int> > &out_map);

    int get_rnd_indeces(vector<vector<int> > map, vector<vector<int> > &rnd_indeces);

    void saveCMC(string path, Mat cmc);

    template<typename T>
    int get_rank_index(priority_queue<RankElement<T>, vector<RankElement<T> >, RankElement<T> > pq, int query_class);
}
#endif //MULTIVIEWBODYMODEL_MULTIVIEWBODYMODEL_H
