//
// Created by Mauro Piazza 07/04/16.
//

#include <iostream>
#include <opencv/highgui.h>

#include "MultiviewBodyModel.h"

using namespace std;
using namespace cv;

/*
 * Global variables
 */

// Images
const int g_n_images = 3;
int g_img_index = 0;
string g_paths[g_n_images] = {"../imgs/monnalisa1.jpg", "../imgs/monnalisa2.jpg", "../imgs/monnalisa3.jpg"};
Mat g_imgs[g_n_images];
Mat g_previous_img; // used for undo operation

// Window system
string g_wnd_name = "Window";

// Keypoints' storage
vector<KeyPoint> g_keypoints;

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
        g_previous_img = g_imgs[g_img_index].clone();

        // Drawing a red circle in the keypoint position
        circle(g_imgs[g_img_index], Point(key.pt.x, key.pt.y), 2, Scalar(0, 0, 255), -1);
        imshow(g_wnd_name, g_imgs[g_img_index]);
    }
}

int main()
{
    // Showing the image
    namedWindow(g_wnd_name);
    setMouseCallback(g_wnd_name, onMouse);

    // Going all over the images and finding keypoints by hand
    int c = 0;

    // Showing the first image
    g_imgs[g_img_index] = imread(g_paths[g_img_index]);
    imshow(g_wnd_name, g_imgs[g_img_index]);

    while (g_img_index < g_n_images - 1)
    {
        if (c == 13) // c = 'enter'
        {
            g_img_index++;
            g_imgs[g_img_index] = imread(g_paths[g_img_index]);
            imshow(g_wnd_name, g_imgs[g_img_index]);

            cout << "Tot keypoints = " << g_keypoints.size() << endl;
        }
        else if (c == 'z') // c = 'z'
        {
            // undo operation
            imshow(g_wnd_name, g_previous_img);
            g_imgs[g_img_index] = g_previous_img.clone();

            KeyPoint key_removed = g_keypoints.back();
            g_keypoints.pop_back();

            printf("Keypoint removed at (%f, %f).\n", key_removed.pt.x, key_removed.pt.y);
        }

        // Go to the next image
        c = waitKey(0);
    }

    return 0;
}