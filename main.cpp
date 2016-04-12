//
// Created by Mauro Piazza 07/04/16.
//

#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>

#include "MultiviewBodyModel.h"

using namespace std;
using namespace multiviewbodymodel;
using namespace cv;

/*
 * Global variables
 */

// Images
const int g_num_images = 3;
int g_image_index = 0;
string g_paths[g_num_images] = {"../imgs/monnalisa1.jpg", "../imgs/monnalisa2.jpg", "../imgs/monnalisa3.jpg"};
vector<Mat> g_images;
Mat g_previous_image; // used for undo operation

// Window system
string g_window_name = "Window";

// Keypoints' storage
vector<KeyPoint> g_keypoints;

// Temporary variables
float g_current_confidence;


/*
 * Callbacks
 */

static void onMouse(int event, int x, int y, int, void* data)
{
    // Left button pressed
    if (event == EVENT_LBUTTONDOWN)
    {
        // Catch the keypoint passed and set the new parameters
        KeyPoint key(Point2f(static_cast<float>(x), static_cast<float>(y)), 2);
        g_keypoints.push_back(key);

        printf("Keypoint inserted at (%d, %d).\n", x, y);

        // Save the current state
        g_previous_image = g_images[g_image_index].clone();

        // Drawing a red circle in the keypoint position
        circle(g_images[g_image_index], Point(key.pt.x, key.pt.y), 2, Scalar(0, 0, 255), -1);
        imshow(g_window_name, g_images[g_image_index]);
    }
}


int main()
{
    // Keypoints storage (a vector of keypoints for each image)
    vector<vector<KeyPoint> > vec_keypoints;

    // Window parameters
    namedWindow(g_window_name);
    setMouseCallback(g_window_name, onMouse);

    // Going all over the images and finding keypoints by hand
    int c = 0;
    cout << "Keypoint selection..." << endl;
    while (c != 'q' && g_image_index < g_num_images)
    {
        if (c == 0)
        {
            // Reading and showing the image
            Mat img = imread(g_paths[g_image_index]);
            if (img.data)
            {
                g_images.push_back(img);
                imshow(g_window_name, g_images[g_image_index]);
            }
            else
            {
                cerr << "Invalid images" << endl;
                return -1;
            }
        }

        // Catching the character pressed
        c = waitKey(0);

        if (c == 13)
        {
            // Enter is pressed: store the vector of keypoints
            if (!g_keypoints.empty())
            {
                vec_keypoints.push_back(g_keypoints);
                g_keypoints.clear();
            }

            // Go to the next image
            g_image_index++;
            c = 0;
        }
        else if (c == 'z' && !g_keypoints.empty())
        {
            // Undo is pressed: undisplay the last keypoint selected
            imshow(g_window_name, g_previous_image);
            g_images[g_image_index] = g_previous_image.clone();

            // Remove the last keypoint inserted
            KeyPoint key_removed = g_keypoints.back();
            g_keypoints.pop_back();

            printf("Keypoint removed at (%f, %f).\n", key_removed.pt.x, key_removed.pt.y);
        }
    }
    cout << "done!" << endl;


    ConfidenceDescriptor cd;
    ViewDetail vd;

    cd.id = 6;
    cd.confidence = 0.8f;
    cd.descriptor.create(1, 3, CV_32FC1);

    vd.keypoints_descriptors.push_back(cd);

    cd.id = 3;
    cd.confidence = 0.9f;
    cd.descriptor.create(1, 3, CV_32FC1);

    vd.keypoints_descriptors.push_back(cd);

    cd.id = 2;
    cd.confidence = 0.3f;
    cd.descriptor.create(1, 3, CV_32FC1);

    vd.keypoints_descriptors.push_back(cd);

    cd.id = 1;
    cd.confidence = 0.5f;
    cd.descriptor.create(1, 3, CV_32FC1);

    vd.keypoints_descriptors.push_back(cd);

    vd.angle = 60.0f;

    vector<ViewDetail> views;
    views.push_back(vd);

    vd.angle = 45.0f;
    views.push_back(vd);

    MultiviewBodyModel mbm(views);

    mbm.ConfidenceNormalization();



    // Decriptors computation
//    Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create("SIFT");
//
//    vector<Mat> vec_descriptors;
//    if (!descriptor_extractor.empty())
//    {
//        descriptor_extractor->compute(g_images, vec_keypoints, vec_descriptors);
//    }
//    cout << "done!" << endl;
//
//    cout << "vec_descriptors.size = " << vec_descriptors.size() << endl;



//    // Matching
//    BruteForceMatcher<L2<float> > matcher;
//    vector<DMatch> matches;
//    matcher.match(vec_descriptors[0], vec_descriptors[1], matches);
//
//    namedWindow("matches", 1);
//    Mat img_matches;
//    drawMatches(g_images[0], vec_keypoints[0], g_images[1], vec_keypoints[1], matches, img_matches);
//    imshow("matches", img_matches);
//    waitKey(0);

    return 0;
}