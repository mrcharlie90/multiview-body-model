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

enum OcclusionType { MODELOCCLUDED, VISIBLE, OCCLUDED};

enum DescriptorType { SIFT, SURF, ORB, FREAK, BRIEF, INVALID };

/**
 * Group all the necessary information for obtaining training(testing) files
 * and the functions' parameters.
 */
struct Configuration {
    // From the conf file
    cv::string conf_file_path;
    cv::string res_file_path;
    cv::string map_file_path;
    cv::string dataset_path;
    std::vector<cv::string> persons_names;
    std::vector<cv::string> views_names;
    int model_set_size;
    int keypoints_number;
    cv::Mat num_images;
    std::vector<int> poses;

    // Configuration Matrices
    std::vector<std::vector<int> > matching_poses_map;
    std::vector<cv::Mat> matching_weights;

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

    void enable(std::string desc_type) {
        assert(!desc_type.empty());
        name = desc_type;
        enabled = true;
    }

    // Writes all stats values in a file named timing.xml
    void write(Configuration conf);

    // Show the all the stats value on the console
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
                                       int keypoint_size, DescriptorType descriptor_extractor_type,
                                       Timing &timing);

    /**
     * Match the frame descriptors with the current model. At least one pose must be loaded.
     * Returns a similarity value between the frame and model skeletons.
     * @param conf configuration object
     * @param frame_descriptors descriptors read from the skeleton file
     * @param frame_ps pose read from the skeleton file
     * @param frame_conf confidences read from the file
     * @param timing keeps track of one matching time
     * @return similarity score between the two skeletons
     */
    float match(Configuration &conf, const cv::Mat &frame_descriptors, int frame_ps,
                    const std::vector<float> &frame_conf, Timing &timing);

    /**
     * Match the frame descriptors with the current model. At least one pose must be loaded.
     * Returns a similarity value between the frame and model skeletons.
     * @param frame_ps pose read from the skeleton file
     * @param frame_descriptors descriptors read from the skeleton file
     * @param frame_conf confidences read from the file
     * @param matching_poses_map a set of vectors of integers, one for each pose, where each element contains
     * an alternative pose to consider during the match
     * @param matching_weights  a set of cv::Mat object which contains a weight for each keypoint descriptor
     * to consider during the similarity score computation.
     * @param norm_type norm type (NORM_L2 or NORM_HAMMING)
     * @param occlusion_search true to seach another keypoint descriptor into the model to compute the score
     * @param timing track of one matching time
     * @return similarity score between the two skeletons
     */
    float match(int frame_ps, const cv::Mat &frame_descriptors, const std::vector<float> &frame_conf,
                    const std::vector<std::vector<int> > &matching_poses_map,
                    const std::vector<cv::Mat> &matching_weights, int norm_type, bool occlusion_search,
                    Timing &timing);
private:
    /**
     * Determines occlusion between fram and model confidences
     * @param frame_conf frame keypoint confidence
     * @param model_conf model keypoint confidence
     * @return the type of occlusion
     */
    OcclusionType check_occlusion(float frame_conf, float model_conf);

    /**
     * Determines the type of occlusion of one model keypoint.
     * @param model_conf model keypoint confidence
     * @return the type of occlusion
     */
    OcclusionType check_occlusion(float model_conf);


    /**
     * Finds the minimum distance in a set of matches.
     * @param matches matches returned from a DescriptorMatcher object
     * @return the minimum distance
     */
    float find_min_match_distance(std::vector<cv::DMatch> matches);

    /**
     * Returns the index relative to the weight matrix to use contained in the list of matrices given
     * in input. See the match method.
     * @param ps_frame frame pose number
     * @param ps_model model pose number
     * @return index for which matching_weighted[i] correspond to the match ps_frameVSps_model
     */
    int get_matching_index(int ps_frame, int ps_model = 0);

    /**
     * Computes the weighted distance between the kp_idx frame and model descriptors.
     * @param frame_pose frame pose number
     * @param frame_descriptors frame set of descriptors
     * @param model_descriptors model set of descriptors
     * @param kp_idx keypoint index for which compute the distance
     * @param matching_idx index of the weight matrix relative to these pose numbers
     * @param matching_poses_map alternative poses to use in case of keypoints occluded
     * @param matching_weights set of matrices containing a set of weights for each keypoint
     * for each combination of matching poses
     * @param norm_type norm type (NORM_L2 or NORM_HAMMING)
     * @param occlusion_type true to enable
     * @param w_avg weighted sum of the distance computed
     * @param sum_w sum of the weights
     */
    void compute_weighted_distance(int frame_pose, const cv::Mat &frame_descriptors,
                                   const cv::Mat &model_descriptors, int kp_idx, int matching_idx,
                                   const std::vector<std::vector<int> > &matching_poses_map,
                                   const std::vector<cv::Mat> &matching_weights, int norm_type,
                                   OcclusionType occlusion_type, bool occlusion_search, float &w_avg,
                                   float &sum_w);


    /// Store the pose number (i.e. 1:front, 2:back, 3:left-side, 4:right-side)
    std::vector<int> pose_number_;

    /// Contains the set of keypoints for each pose stored
    std::vector<std::vector<cv::KeyPoint> > pose_keypoints_;

    /// Contains the set of descriptors for each pose stored
    std::vector<cv::Mat> pose_descriptors_;

    /// Contains the set of confidences for each pose stored
    /// 1 means "keypoint visible" and 0 means "keypoint occluded"
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

/**
 * Computes the relative descriptor type value from string.
 * @param str string to convert
 * @return a descriptor type value
 */
DescriptorType char2descriptor_type(const char *str);

/**
 * Show the help for parameters settings
 */
void show_help();

/**
 * Gets the corresponding descriptor's norm type
 * @param descriptor_type
 * @return Returns -1 if an invalid descriptor type name is given
 */
int get_norm_type(DescriptorType descriptor_type);

/**
 * Checks if the file node fn is a sequence, used only in parse_args()
 * @param fn file node
 */
void check_sequence(cv::FileNode fn);

/**
 * Reads the skeleton file given from the skeletal tracker.
 * @param skel_path path to the skeleton file
 * @param keypoint_size keypoint size
 * @param out_keypoints keypoints found
 * @param out_confidences confidences found
 * @param out_pose_side pose side found
 */
void read_skel_file(const cv::string &skel_path, int keypoint_size, std::vector<cv::KeyPoint> &out_keypoints,
                    std::vector<float> &out_confidences, int &out_pose_side);

/**
 * Extract the descriptors using the algorithm specified by descriptor_extractor_type
 * @param img_path path to the image file
 * @param in_keypoints keypoints where to compute the descriptors
 * @param descriptor_extractor_type descriptor extractor type
 * @param out_descriptors descriptors computed
 */
void compute_descriptors(const std::string &img_path, const std::vector<cv::KeyPoint> &in_keypoints,
                         DescriptorType descriptor_extractor_type, cv::Mat &out_descriptors);

/**
 * Loads all persons paths to images and skeletons files.
 * @param conf configuration object
 * @param out_imgs_paths images paths returned
 * @param out_skels_paths skeletons paths returned
 * @return the total number of images in the dataset
 */
int load_person_imgs_paths(const Configuration &conf, std::vector<std::vector<cv::string> > &out_imgs_paths,
                           std::vector<std::vector<cv::string> > &out_skels_paths);

/**
 * Masks storing additional information about the image contained in skels_paths at index (i, j).
 * @param skels_paths set of paths from which creates the masks
 * @param masks a set o matrices set to zero to return
 */
template <typename T>
void load_masks(const std::vector<std::vector<T> > &skels_paths,
               std::vector<cv::Mat> &masks);

/**
 * For each set of skeleton paths, a set of file indices are returned pointed to the file that will be
 * loaded into the model associated with the person.
 * @param poses poses to load (the images with pose numbers not contained in this list won't be considered)
 * @param img_paths path to images
 * @param skels_paths skeleton paths
 * @param max_size maximum size of the model set (this value will be corrected if not a multiple
 * of poses.size())
 * @param min_keypoints_visibles the minimum number of keypoints visible in the skeleton
 * @param out_model_set set of indices to the skeleton that will be loaded into the models: in particular
 * the model_set[i] will be used for the i-th model
 * @return the number of total rounds in which the images must be loaded
 */
int load_models_set(std::vector<int> &poses, const std::vector<std::vector<cv::string> > img_paths,
                    const std::vector<std::vector<cv::string> > skels_paths, int max_size, int min_keypoints_visibles,
                    std::vector<std::vector<int> > &out_model_set);

/**
 * Tokenize a line with the delimeter given
 * @param line
 * @param delim
 * @param out_tokens tokens found in the line
 */
void tokenize(const std::string &line, char delim, std::vector<cv::string> &out_tokens);

/**
 * Finds the total number of keypoints visible in the skeleton
 * @param skel_path path to the skeleton
 * @param num_keypoints total number of keypoints in the skeleton
 * @return the number of keypoints visible
 */
int get_total_keyponts_visible(const std::string &skel_path, int num_keypoints);

/**
 * Return the pose side of the skeleton
 * @param path path to the skeleton file
 * @return the pose side
 */
int get_pose_side(cv::string path);

/**
 * Loads a set of empty models, one for each person
 * @param size the number of empty models to create
 * @param models a set of empty models
 */
void empty_models(int size, std::vector<MultiviewBodyModel> &models);

/**
 * Initialize models
 * @param conf configuration object
 * @param rounds round number
 * @param imgs_paths paths to images
 * @param skels_paths paths to skeletons
 * @param models_set the model set specifiyng the images to load
 * @param models the models to be loaded
 * @param timing store the model's loading time
 */
void init_models(Configuration conf, int rounds, const std::vector<std::vector<std::string> > imgs_paths,
                 const std::vector<std::vector<std::string> > skels_paths,
                 const std::vector<std::vector<int> > &models_set,
                 std::vector<MultiviewBodyModel> &models, Timing &timing);

/**
 * Returns the index in the queue of the element with the class equal to test_class
 * @param pq priority queue
 * @param ground_truth ground-truth class
 * @return
 */
template<typename T>
int get_rank_index(std::priority_queue<PQRank<T>, std::vector<PQRank<T> >, PQRank<T> > pq, int ground_truth);

/**
 * Computes the area under the curve.
 * @param CMC cumulative matching characteristic curve
 * @return nAUC
 */
float area_under_curve(cv::Mat CMC);

/**
 * Return the correct results file name.
 * @param conf configuration object
 * @return a string
 */
cv::string get_res_filename(Configuration conf);

/**
 * Writes the results in the right folders.
 * @param conf configuration file
 * @param CMC
 * @param nAUC
 */
void write_cmc_nauc(Configuration conf, cv::Mat CMC, float nAUC);

}
#endif // MULTIVIEWBODYMODEL_MULTIVIEWBODYMODEL_H
