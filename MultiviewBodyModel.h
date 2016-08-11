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
#include <opencv/cxcore.h>


namespace  multiviewbodymodel {
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


    // Overall time to compute the matching
    double t_tot_exec;

    void enable() {
        enabled = true;
    }

    // Writes all stats values in a file named timing.xml
    void write(cv::string name);

    // Show the all the stats value on the console
    void show();
};


enum OcclusionType {
    BOTHOCCLUDED,
    FRAMEOCCLUDED,
    MODELOCCLUDED,
    VISIBLE
};

enum DescriptorType {
    SIFT,
    SURF,
    ORB,
    FREAK,
    BRIEF,
    INVALID
};

// Group all the necessary information for obtaining training(testing) files
// and the functions' parameters.
struct Configuration {
    // From the conf file
    cv::string conf_file_path;
    cv::string res_file_path;
    cv::string main_path;
    std::vector<cv::string> persons_names;
    std::vector<cv::string> views_names;
    int keypoints_number;
    cv::Mat num_images;
    std::vector<int> poses;

    // Configuration Matrices
    cv::Mat poses_map;
    cv::Mat poses2kp_map;
    cv::Mat kp_map;
    cv::Mat kp_weights;

    // From the command line
    cv::string descriptor_type_str;
    DescriptorType descriptor_type;
    int norm_type;
    int keypoint_size;
    bool occlusion_search;

    // Shows the parameters loaded from the console and from the config.xml file
    void show();
};



// Models a skeleton of one person seen in a scene captured by several cameras.
// One person is characterized by a number of pose sides (computed from a skeletal tracker,
// for example) which are used to re-identify a person body seen in the scene.
class MultiviewBodyModel {
public:
    void read_pose_compute_descriptors(cv::string img_path, cv::string skel_path,
                                       int keypoint_size, cv::string descriptor_extractor_type,
                                       Timing &timing);

    float match(const cv::Mat &frame_descriptors, int frame_ps, const std::vector<float> &frame_conf, int norm_type,
                    bool occlusion_search, const cv::Mat &poses_map, const cv::Mat &kp_map,
                    const cv::Mat &kp_weights, const cv::Mat &ps2keypoints_map, Timing &timing);

    float match(const cv::Mat &frame_descriptors, int frame_ps, const std::vector<float> &frame_conf,
                bool occlusion_search, Configuration &conf, Timing &timing);

private:
    OcclusionType check_occlusion(float frame_conf, float model_conf);

    void occlusion_norm(const cv::Mat &frame_desc, int model_ps, int model_kp_idx,
                                            const cv::Mat &model_poses_map, const cv::Mat &model_ps2keypoints_map,
                                            const cv::Mat &model_kp_map, const cv::Mat &model_kp_weights, int norm_type,
                                            double &out_weighted_distance, double &out_tot_weight);


    void create_descriptor_from_poses(int pose_not_found, const cv::Mat &model_poses_map,
                                      const cv::Mat &model_ps2keypoints_map,
                                      const cv::Mat &model_kp_map, const cv::Mat &model_kp_weights,
                                      cv::Mat &out_model_descriptors, cv::Mat &out_descriptors_weights,
                                      cv::Mat &keypoints_mask);
    // Pose number (i.e. 1:front, 2:back, 3:left-side, 4:right-side)
    std::vector<int> pose_number_;

    // Contains descriptors for each keypoint stored
    std::vector<cv::Mat> pose_descriptors_;

    // Keypoints for each pose
    std::vector<std::vector<cv::KeyPoint> > pose_keypoints_;

    // Confidence value in [0, 1] for each keypoint of each pose.
    // For now 1 means "keypoint visible" and 0 means "keypoint occluded"
    std::vector<std::vector<float> > pose_confidences_;
};

// ------------------------------------------------------------------------- //
//                           Functions declarations                          //
// ------------------------------------------------------------------------- //

// Element used for storing the ground truth class of the current frame with
// the relative score obtained from the matching function.
template<typename T>
struct PQRank {
    int class_idx;
    T score;

    // Comparator for the priority queue
    bool operator()(const PQRank<T> &pqr1, const PQRank<T> &pqr2) {
        return pqr1.score > pqr2.score;
    }
};



// Parse input arguments and initialize the configuration object.
void parse_args(int argc, char **argv, Configuration &out_conf);

void read_config_file(Configuration &configuration);

DescriptorType char2descriptor_type(const char *str);
// Show the help for parameters settings
void show_help();

// Gets the corresponding descriptor's norm type
// Returns -1 if a invalid descriptor name is given
int get_norm_type(DescriptorType descriptor_type);

// Checks if the file node fn is a sequence, used only in parse_args()
void check_sequence(cv::FileNode fn);

void read_skel_file(const cv::string &skel_path, int keypoint_size,
                    std::vector<cv::KeyPoint> &out_keypoints,
                    std::vector<float> &out_confidences,
                    int &out_pose_side);

void compute_descriptors(const std::string &img_path, const std::vector<cv::KeyPoint> &in_keypoints,
                         const cv::string &descriptor_extractor_type, cv::Mat &out_descriptors);

int factorial(int n);

// Returns the index in the queue of the element with the class equal to test_class
template<typename T>
int get_rank_index(std::priority_queue<PQRank<T>, std::vector<PQRank<T> >, PQRank<T> > pq, int test_class);



/**
 * NEW
 */

int load_person_imgs_paths(const Configuration &conf, std::vector<std::vector<cv::string> > &out_imgs_paths,
                           std::vector<std::vector<cv::string> > &out_skels_paths);

template <typename T>
void load_masks(const std::vector<std::vector<T> > &skels_paths,
               std::vector<cv::Mat> &masks);

int load_models_set(std::vector<int> &poses, const std::vector<std::vector<cv::string> > img_paths,
                    const std::vector<std::vector<cv::string> > skels_paths, int max_size, int min_keypoints_visibles,
                    std::vector<std::vector<int> > &out_model_set);

int get_total_keyponts_visible(const std::string &skel_path, int num_keypoints);

void tokenize(const std::string &line, char delim, std::vector<cv::string> &out_tokens);

int get_pose_side(cv::string path);

void empty_models(int size, std::vector<MultiviewBodyModel> &models);

void init_models(Configuration conf, int rounds, const std::vector<std::vector<std::string> > imgs_paths,
                 const std::vector<std::vector<std::string> > skels_paths,
                 const std::vector<std::vector<int> > &models_set,
                 std::vector<MultiviewBodyModel> &models, Timing &timing);

}
#endif // MULTIVIEWBODYMODEL_MULTIVIEWBODYMODEL_H
