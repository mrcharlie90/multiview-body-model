// Copyright (c) [2016] [Mauro Piazza]
//
//          IASLab License
//
// Contains the MultiviewBodyModel class definition and some utility functions
// which can be used for loading skeletons and for saving results.
// 
// Some utility structures such as Timing and Configuration are used for  logging
// performance and for loading settings.

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
    using std::vector;
    using cv::string;
    

    // Used for performance logging.
    // Each field stores a timing value named according to the
    // methods it is used in.
    //
    // Call enable() before use
    // Each methods checks the enabled flag to decide when
    // to store performance or not.
    // 
    // write() save the results in a file named timing.xml
    struct Timing {
        // All methods reads this flag to check wether
        // the stats logging are active
        bool enabled;
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
        Timing() {
            enabled = false;
        }

        void enable() {
            enabled = true;
        }
        // Writes all stats values in a file named timing.xml
        void write(string name);

        // Show the all the stats value on the console
        void show();


    };

    // Models a skeleton of one person seen in a scene captured by several cameras.
    // One person is characterized by a number of pose sides (computed from a skeletal tracker,
    // for example) which are used to re-identify a person body seen in the scene.
    class MultiviewBodyModel {
    public:
        // Sets maximum number of poses the model should contain.
        // The model cannot accept a poses' value greater than this value.
        // Example: if the skeletal tracker outputs
        // 1 for left-side, 2 for right side and 3 for front-side
        // set max_poses to 3
        MultiviewBodyModel(int max_poses);

        // Loads a model from an image and skel file.
        // It extract each keypoint descriptor for each pose and store the results in the model.
        //
        // If the pose side value of the skeleton is greater than the max number of poses
        // it discards the reading and return -1
        //
        // If the skeleton is successfully loaded it returns 1, otherwise 0
        int ReadAndCompute(string file_path, string img_path, string descriptor_extractor_type,
                           int keypoint_size, Timing &timing);

        // Search for the same pose side in the model and computes the match.
        //
        // If occlusion_search is true (by default), a keypoint descriptor which is occluded is match 
        // with a descriptor in another view (the first one which has the descriptor visible).
        float Match(cv::Mat query_descriptors, vector<float> query_confidences, int query_pose_side,
                    bool occlusion_search = true);

        // Returns true when all poses in the model are acquired successfully.
        // This means that the current pose_sides_ vector has size equal to max_poses
        bool ready();

    private:
        // Creates a confidence match which defines the operation to execute
        // when the following cases arises:
        // 1. both keypoints occluded
        // 2. one keypoint occluded and the other visible
        // 3. both keypoints visible
        //
        // out_mask is defined in this way:
        // 1. both keypoints occluded => mask(i,j) = 0 (Don't consider keypoints)
        // 2. one keypoint occluded and the other visible => mask(i,j) = 1 -> Find the keypoint
        // occluded in the other views
        // 3. both keypoints visible => mask(i,j) = 2 -> Compute the distance between the keypoints
        void create_confidence_mask(vector<float> &query_confidences, vector<float> &train_confidences,
                                    vector<char> &out_mask);

        // Used in the matching phase when one keypoint is occluded and the other
        // is visible.
        bool  get_descriptor_occluded(int keypoint_index, cv::Mat &descriptor_occluded);

        // The maximum number of poses the model should accept
        int max_poses_;

        // Pose number (i.e. 1:front, 2:back, 3:left-side, 4:right-side)
        vector<int> pose_side_;

        // Contains keypoint's descriptors for each pose
        vector<cv::Mat> views_descriptors_;

        // Keypoints for each pose
        vector<vector<cv::KeyPoint> > views_keypoints_;

        // Images of each pose loaded
        vector<cv::Mat> views_images_;

        // Confidence value in [0, 1] for each keypoint of each pose.
        // For now 1 means "keypoint visible" and 0 means "keypoint occluded"
        vector<vector<float> > views_descriptors_confidences_;
    };

    // -------------------------------------------------------------------------
    //                      Utility functions declarations
    // -------------------------------------------------------------------------

    // Element used for storing the ground truth class of the current frame with
    // the relative score obtained from the matching function.
    template<typename T>
    struct RankElement {
        int classIdx;
        T score;

        // Comparator for the priority queue
        bool operator()(const RankElement<T> &re1, const RankElement<T> &re2) {
            return re1.score > re2.score;
        }
    };

    // Group all the necessary information for obtaining training(testing) files
    // and the functions' parameters.
    struct Configuration {
        // From the conf file
        string conf_file_path;
        string main_path;
        vector<string> persons_names;
        vector<string> views_names;
        cv::Mat num_images;
        int max_poses;

        // From the command line
        vector<string> descriptor_extractor_type;
        int keypoint_size;

        // Shows the parameters loaded from the config.xml file
        void show();
    };

    // Parse input arguments and initialize the configuration object.
    void parse_args(int argc, char **argv, Configuration &out_conf);

    // Checks if the file node fn is a sequence, used only in parse_args()
    void check_sequence(cv::FileNode fn);

    // Creates two vectors containing all the paths to the skeleton and image files grouped by person
    void load_train_paths(string main_path, vector<string> persons_names, vector<string> views_names,
                          cv::Mat num_images,
                          vector<vector<string> > &out_imgs_paths, vector<vector<string> > &out_skel_paths);
    void load_train_paths(Configuration conf, vector<vector<string> > &out_skels_paths,
                          vector<vector<string> > &out_imgs_paths);

    // Loads one model from images chosen sequentially in the training set
    // A set of masks is used to check which images have already been chosen for each person
    // One mask's element masks[i].col(j) is a counter which tells how many times the relative image is considered
    //
    // out_models is a vector of body models with all poses loaded and the relative descriptors computed.
    bool load_models(string descriptor_extractor_type, int keypoint_size, int max_poses, vector<cv::Mat> &masks,
                     vector<vector<string> > &train_skels_paths, vector<vector<string> > &train_imgs_paths,
                     vector<MultiviewBodyModel> &out_models, Timing &timing);

    // Loads one image skeleton, used in the testing phase to load the test image
    // Returns the keypoints and confidences read from the skel file and the descriptors computed from the keypoints pose a
    void read_skel(string descriptor_extractor_type, int keypoint_size, string skel_path, string img_path,
                   cv::Mat &out_image, vector<cv::KeyPoint> &out_keypoints, vector<float> &out_confidences,
                   cv::Mat &out_descriptors, int &out_pose_side, Timing &timing);

    // Returns the index in the queue of the element with the class equal to test_class
    template<typename T>
    int get_rank_index(std::priority_queue<RankElement<T>, vector<RankElement<T> >, RankElement<T> > pq,
                                            int test_class);

    // Reads the pose side in the file specified by the path string
    int get_pose_side(string path);

    // Returns a set of vectors where each element v[i] store
    // - a pose side number for i is even
    // - the number of consecutive images with the previous element's pose side for i odd
    void get_poses_map(vector<vector<string> > train_paths, vector<vector<int> > &out_map);

    // Exploit the map given from get_poses_map() method to return a sequence of indices for each person
    // where the relative pose side is uniformly distribuited
    //
    // The number returned is the total number of indices produced
    int get_rnd_indices(vector <vector<int> > map, int max_poses, vector <vector<int> > &out_rnd_indices);

    // Saves the CMC curve in a file
    void saveCMC(string path, cv::Mat cmc);

    // Saves in a file mask.xml the set of mask used during models loading
    void save_mask(string d_name, vector<cv::Mat> masks);

    void print_dataset_usage(vector<cv::Mat> masks);

}
#endif // MULTIVIEWBODYMODEL_MULTIVIEWBODYMODEL_H
