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

enum OcclusionType { BOTHOCCLUDED, FRAMEOCCLUDED, MODELOCCLUDED, VISIBLE, OCCLUDED};

enum DescriptorType { SIFT, SURF, ORB, FREAK, BRIEF, INVALID };

/**
 * Used for performance logging.
 * There a field for each time that will be stored.
 *
 * Call enable() before use
 * write() save the results in a file named timing.xml
 */
struct Timing {
    // All methods reads this flag to check wether
    // the stats logging are active
    bool enabled;
    std::string name;

    // Average times
    double one_match;
    int n_one_match;
    double extraction;
    int n_extraction;
    double models_loading;
    int n_models_loading;
    double descr_creation;
    int n_descr_creation;

    void enable(std::string desc_type) {
        assert(!desc_type.empty());
        name = desc_type;
        enabled = true;
    }

    // Writes all stats values in a file named timing.xml
    void write(cv::string path, cv::string name);

    // Show the all the stats value on the console
    void show();
};

/**
 * Group all the necessary information for obtaining training(testing) files
 * and the functions' parameters.
 */
struct Configuration {
    // From the conf file
    cv::string conf_file_path;
    cv::string res_file_path;
    cv::string dataset_path;
    std::vector<cv::string> persons_names;
    std::vector<cv::string> views_names;
    int model_set_size;
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

/**
 * Element used for storing the ground truth class of the current frame with
 * the relative score obtained from the matching function.
 */
template<typename T>
struct PQRank {
    int class_idx;
    T score;

    // Comparator for the priority queue
    bool operator()(const PQRank<T> &pqr1, const PQRank<T> &pqr2) {
        return pqr1.score > pqr2.score;
    }
};

/**
 * Models a skeleton of one person seen in a scene captured by several cameras.
 * One person is characterized by a number of pose sides (computed from a skeletal
 * tracker, for example) which are used to re-identify a person body seen in the scene.
 */
class MultiviewBodyModel {
public:

    /**
     * Loads the image skeleton into the model, if the poses read already exists then it is replaced
     * with the new one.
     * @param img_path image file path into the data set
     * @param skel_path skeleton file path into the data set
     * @param keypoint_size size of the keypoint
     * @param descriptor_extractor_type type of descriptor (e.g. SIFT, SURF, ORB, BRIEF, FREAK)
     * @param timing if enabled, it will store the descriptor computation execution time
     */
    void read_pose_compute_descriptors(cv::string img_path, cv::string skel_path,
                                       int keypoint_size, cv::string descriptor_extractor_type,
                                       Timing &timing);

    /**
     * Computes the match between the frame descriptors computed and the current model.
     *
     * @param frame_descriptors descriptors computed onto the keypoints location given by the skeletal tracker
     *                          with the same algorithm used in this model.
     * @param frame_ps pose of the subject in the frame being matched
     * @param frame_conf confidences of the keypoints given by the skeletal tracker
     * @param norm_type normtype of the algorithm (same used  in the openCV)
     * @param occlusion_search set this to true to enable the occlusion search algorithm
     * @param poses_map matrix containing all mapping for each pose stored in the model.
     *                  Each row contains a set of alternative pose to consider for the match.
     * @param kp_map  matrix containing
     * @param kp_weights
     * @param ps2keypoints_map
     * @param timing
     * @return
     */
    float match(const cv::Mat &frame_descriptors, int frame_ps, const std::vector<float> &frame_conf, int norm_type,
                    bool occlusion_search, const cv::Mat &poses_map, const cv::Mat &kp_map,
                    const cv::Mat &kp_weights, const cv::Mat &ps2keypoints_map, Timing &timing);

    double match(const cv::Mat &frame_descriptors, int frame_ps, const std::vector<float> &frame_conf,
                     const std::vector<std::vector<int> > &matching_poses_map, const std::vector<cv::Mat> &matching_weights,
                     int norm_type, bool occlusion_search, Timing &timing);

    float match(Configuration &conf, const cv::Mat &frame_descriptors, int frame_ps, const std::vector<float> &frame_conf,
                    bool occlusion_search, Timing &timing);

private:
    OcclusionType check_occlusion(float frame_conf, float model_conf);

    OcclusionType check_occlusion(float model_conf);

    void occlusion_norm(const cv::Mat &frame_desc, int model_ps, int model_kp_idx,
                                            const cv::Mat &model_poses_map, const cv::Mat &model_ps2keypoints_map,
                                            const cv::Mat &model_kp_map, const cv::Mat &model_kp_weights, int norm_type,
                                            double &out_weighted_distance, double &out_tot_weight);


    void create_descriptor_from_poses(int pose_not_found, const cv::Mat &model_poses_map,
                                          const cv::Mat &model_ps2keypoints_map,
                                          const cv::Mat &model_kp_map, const cv::Mat &model_kp_weights,
                                          cv::Mat &out_model_descriptors, cv::Mat &out_descriptors_weights,
                                          cv::Mat &keypoints_mask, Timing &timing);

    cv::Mat get_alternative_descriptor(int ps_idx, int kp_idx, const cv::Mat &poses_map, const cv::Mat &kp_map,
                                           const cv::Mat &ps2keypoints_map);

    double find_min_match_distance(std::vector<cv::DMatch> matches);

    int get_matching_index(int ps_frame, int ps_model = 0);

    void compute_weighted_distance(int frame_pose, const cv::Mat &frame_descriptors,
                                       const cv::Mat &model_descriptors, int kp_idx, int matching_idx,
                                       const std::vector<std::vector<int> > &matching_poses_map,
                                       const std::vector<cv::Mat> &matching_weights, int norm_type,
                                       OcclusionType occlusion_type, double &w_avg, double &sum_w);


    // Store the pose number (i.e. 1:front, 2:back, 3:left-side, 4:right-side)
    std::vector<int> pose_number_;

    // Contains the set of keypoints for each pose stored
    std::vector<std::vector<cv::KeyPoint> > pose_keypoints_;

    // Contains the set of descriptors for each pose stored
    std::vector<cv::Mat> pose_descriptors_;

    // Contains the set of confidences for each pose stored
    // 1 means "keypoint visible" and 0 means "keypoint occluded"
    std::vector<std::vector<float> > pose_confidences_;

}; // end-class



// ------------------------------------------------------------------------- //
//                           Functions declarations                          //
// ------------------------------------------------------------------------- //


/**
 * Parse input arguments and initialize the configuration object.
 */
void parse_args(int argc, char **argv, Configuration &out_conf);

/**
 * Reads the configuration file and stores all parameters settings in configuration.
 * @param configuration
 */
void read_config_file(Configuration &configuration);

DescriptorType char2descriptor_type(const char *str);

// Show the help for parameters settings
void show_help();

// Gets the corresponding descriptor's norm type
// Returns -1 if a invalid descriptor name is given
int get_norm_type(DescriptorType descriptor_type);

// Checks if the file node fn is a sequence, used only in parse_args()
void check_sequence(cv::FileNode fn);

void read_skel_file(const cv::string &skel_path, int keypoint_size, std::vector<cv::KeyPoint> &out_keypoints,
                    std::vector<float> &out_confidences, int &out_pose_side);

void compute_descriptors(const std::string &img_path, const std::vector<cv::KeyPoint> &in_keypoints,
                         const cv::string &descriptor_extractor_type, cv::Mat &out_descriptors);

int load_person_imgs_paths(const Configuration &conf, std::vector<std::vector<cv::string> > &out_imgs_paths,
                           std::vector<std::vector<cv::string> > &out_skels_paths);

template <typename T>
void load_masks(const std::vector<std::vector<T> > &skels_paths,
               std::vector<cv::Mat> &masks);

int load_models_set(std::vector<int> &poses, const std::vector<std::vector<cv::string> > img_paths,
                    const std::vector<std::vector<cv::string> > skels_paths, int max_size, int min_keypoints_visibles,
                    std::vector<std::vector<int> > &out_model_set);

void tokenize(const std::string &line, char delim, std::vector<cv::string> &out_tokens);

int get_total_keyponts_visible(const std::string &skel_path, int num_keypoints);

int get_pose_side(cv::string path);

void empty_models(int size, std::vector<MultiviewBodyModel> &models);

void init_models(Configuration conf, int rounds, const std::vector<std::vector<std::string> > imgs_paths,
                 const std::vector<std::vector<std::string> > skels_paths,
                 const std::vector<std::vector<int> > &models_set,
                 std::vector<MultiviewBodyModel> &models, Timing &timing);

// Returns the index in the queue of the element with the class equal to test_class
template<typename T>
int get_rank_index(std::priority_queue<PQRank<T>, std::vector<PQRank<T> >, PQRank<T> > pq, int test_class);

float area_under_curve(cv::Mat mat);

cv::string get_res_filename(Configuration conf);

void print_cmc_nauc(cv::string path, std::string settings, cv::string desc_type, cv::Mat CMC, float nAUC);

void vec2mat(const std::vector<cv::Mat> &vec, cv::Mat &out_mat);

}
#endif // MULTIVIEWBODYMODEL_MULTIVIEWBODYMODEL_H
