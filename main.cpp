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

// Current image displayed and a backup
// taken before the selection of the last keypoint
Mat g_current_image;
Mat g_previous_image;

// Used for storing the current
// confidence chosen by the user
float g_current_confidence;

// Used for storing selected keypoints
int g_keypoints_counter;
vector<KeyPoint> g_keypoints_selected;
vector<ConfidenceDescriptor> g_confidence_descriptors;


// Window system
string g_window_name = "Window";

// Used for randomly generated colors
RNG g_rng;


/**
 * Testing function
 */
void test_distance();

/*
 * Callbacks
 */
static void onMouse(int event, int x, int y, int, void* data)
{
    // Left button pressed
    if (event == EVENT_LBUTTONDOWN)
    {
        KeyPoint key(Point2f(static_cast<float>(x), static_cast<float>(y)), 2);
        ConfidenceDescriptor cd;
        cd.id = g_keypoints_counter;
        cd.confidence = g_current_confidence;

        g_confidence_descriptors.push_back(cd);
        g_keypoints_selected.push_back(key);

        printf("Keypoint selected at (%d, %d).\n", x, y);

        // Save the current state
        g_previous_image = g_current_image.clone();

        // Defining a random color for marking the keypoint

        int icolor = (unsigned)g_rng;


        Scalar color((icolor & 255), ((icolor>>8) & 255) | 190, ((icolor>>16) & 255) | 190);

        cout << color << endl;

        // Drawing a red circle in the keypoint position
        Point key_position(key.pt.x, key.pt.y);
        circle(g_current_image, key_position, 2, color, -1);

        // Displaying the key_counter
        stringstream txt;
        txt << g_keypoints_counter;

        int fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontscale = 0.5;
        int thickness = 1;

        int baseline = 0;
        Size txtsize = getTextSize(txt.str(), fontface, fontscale, thickness, &baseline);

        baseline += thickness;

        Point txtorg((g_current_image.cols - txtsize.width) / 2,
                     (g_current_image.rows + txtsize.height) / 2);

        putText(g_current_image, txt.str(), key_position, fontface, fontscale, color, thickness, 8);
        imshow(g_window_name, g_current_image);

        g_keypoints_counter++;
        txt.str("");

    }
}

/**
 * Manage the confidence selection keyboard input.
 */
 float confidence_selection()
 {
     // Acquire a char
     char d = waitKey(0);

     switch (d)
     {
         case '1':
             g_current_confidence = 0.1f;
             break;
         case '2':
             g_current_confidence = 0.2f;
             break;
         case '3':
             g_current_confidence = 0.3f;
             break;
         case '4':
             g_current_confidence = 0.4f;
             break;
         case '5':
             g_current_confidence = 0.5f;
             break;
         case '6':
             g_current_confidence = 0.6f;
             break;
         case '7':
             g_current_confidence = 0.7f;
             break;
         case '8':
             g_current_confidence = 0.8f;
             break;
         case '9':
             g_current_confidence = 0.9f;
             break;
         case '0':
             g_current_confidence = 1.0f;
             break;
     }

     return g_current_confidence;
 }

int main(int argc, char** argv)
{

    // Loading command line arguments
    if (argc > 1)
    {
        // TODO: arguments processing
    }

    // Paths' required strings
    string main_path = "../dataset/images/";
    vector<string> persons_path;
    persons_path.push_back("gianluca_sync");
    persons_path.push_back("marco_sync");
    persons_path.push_back("matteol_sync");
    persons_path.push_back("matteom_sync");
    persons_path.push_back("nicola_sync");
    persons_path.push_back("stefanog_sync");
    persons_path.push_back("stefanom_sync");

    vector<char> view_name;
    view_name.push_back('c');
    view_name.push_back('r');
    view_name.push_back('l');

    // Parameters initialization
    int persons_number = 7;
    int views_number = 3;
    int frames_number = 74;

    // Window parameters
    namedWindow(g_window_name);
    setMouseCallback(g_window_name, onMouse);

    // Going all over the images and selecting keypoints by hand
    cout << "Keypoint selection..." << endl;

    // Required variables for making the path
    stringstream ss;
    int digit1 = 0;
    int digit2 = 0;

    // Body model variables


    for (int i = 0; i < persons_number; ++i)
    {
        for (int j = 0; j < views_number; ++j)
        {
            int c = 0; // character acquired
            int frame = 0; // frame counter

            vector<ViewDetail> views;
            ViewDetail view_detail;
            while (frame < frames_number)
            {
                if (c == 0)
                {
                    // Compute the path and show the image
                    ss << main_path << persons_path[i] << "/" << view_name[j]
                        << "000" << digit1 << digit2 << ".png";

                    g_current_image = imread(ss.str());

                    if (!g_current_image.data)
                    {
                        cerr << "Failed to load the image." << endl;
                        cerr << "current_path : " << ss.str() << endl;
                        exit(-1);
                    }

                    imshow(g_window_name, g_current_image);
                }

                // Keyboard acquisition
                c = waitKey(0);
                if (c == 13)
                {
                    // Enter key pressed: compute the descriptor
                    Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create("SIFT");

                    cout << "Computing the descriptor...";
                    Mat descriptors;
                    if (!descriptor_extractor.empty())
                    {
                        descriptor_extractor->compute(g_current_image, g_keypoints_selected, descriptors);
                    }
                    cout << "done" << endl;

                    // Storing descriptors and views
                    for (int k = 0; k < g_keypoints_counter; ++k)
                        g_confidence_descriptors[k].descriptor = descriptors.row(k);

                    view_detail.angle = 0.0;
                    view_detail.name = view_name[j];
                    view_detail.keypoints_descriptors = g_confidence_descriptors;

                    // Resetting variables
                    if (!g_keypoints_selected.empty())
                    {
                        g_keypoints_selected.clear();
                        g_keypoints_counter = 0;
                    }

                    c = 0;
                    frame++;

                    digit1 = frame / 10;
                    digit2 = frame % 10;

                    ss.str("");
                }
                else if (c == 'z') // WARNING: multiple pressures not managed
                {
                    // Undo key pressed: remove the last keypoint selected
                    imshow(g_window_name, g_previous_image);
                    g_current_image = g_previous_image.clone();

                    KeyPoint key_removed = g_keypoints_selected.back();
                    g_keypoints_selected.pop_back();
                    g_confidence_descriptors.pop_back();

                    printf("Keypoint removed at (%f, %f).\n", key_removed.pt.x, key_removed.pt.y);

                    g_keypoints_counter--;

                }
                else if (c >= 48 && c <= 58)
                {
                    // A number is pressed: setting confidence
                    // note: press ':' for setting 1.0
                    g_current_confidence = static_cast<float>(c - 48) / 10;
                    cout << "current_confidence = " << g_current_confidence << endl;
                }
                else if (c == 'q')
                {
                    exit(0);
                }
            }

            views.push_back(view_detail);
        }
    }

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

void test_distance()
{
    /**
    * Body model 1
    */
    float descriptors1_30[4][3] = {1, 0, 1,
                                   2, 0, 1,
                                   4, 3, 2,
                                   6, 2, 1};
    float confidence1_30[4] = {0.1, 0.2, 0.3, 0.4};

    float descriptors1_60[4][3] = {0, 0, 1,
                                   7, 4, 0,
                                   2, 0, 1,
                                   1, 0, 0};
    float confidence1_60[4] = {0.4, 0.2, 0.1, 0.3};

    Mat M1_30(4, 3, CV_32F, descriptors1_30);
    Mat M1_60(4, 3, CV_32F, descriptors1_60);

    ViewDetail vd1_30;
    vd1_30.angle = 30.0f;
    for (int i = 1; i <= 4; ++i)
    {
        ConfidenceDescriptor cd;
        cd.id = i;
        cd.confidence = confidence1_30[i-1];
        cd.descriptor = M1_30.row(i-1).clone();
        vd1_30.keypoints_descriptors.push_back(cd);
    }


    ViewDetail vd1_60;
    vd1_60.angle = 60.0f;
    for (int i = 1; i <= 4; ++i)
    {
        ConfidenceDescriptor cd;
        cd.id = i;
        cd.confidence = confidence1_60[i-1];
        cd.descriptor = M1_60.row(i-1).clone();
        vd1_60.keypoints_descriptors.push_back(cd);
    }

    vector<ViewDetail> views1;
    views1.push_back(vd1_30);
    views1.push_back(vd1_60);

    MultiviewBodyModel mbm1(views1);


    /**
     * Body model 2
     */

//    float descriptors2_30[4][3] = {1, 0, 1,
//                                   2, 0, 1,
//                                   4, 3, 2,
//                                   6, 2, 1};
//    float confidence2_30[4] = {0.1, 0.2, 0.3, 0.4};
//
//    float descriptors2_60[4][3] = {0, 0, 1,
//                                   7, 4, 0,
//                                   2, 0, 1,
//                                   1, 0, 0};
//    float confidence2_60[4] = {0.4, 0.2, 0.1, 0.3};

    float descriptors2_30[4][3] = {0, 0, 1,
                                   4, 0, 1,
                                   0, 2, 2,
                                   3, 0, 1};
    float confidence2_30[4] = {0.2, 0.2, 0.1, 0.5};

    float descriptors2_60[4][3] = {1, 1, 1,
                                   2, 3, 0,
                                   0, 4, 1,
                                   7, 0, 0};
    float confidence2_60[4] = {0.1, 0.1, 0.1, 0.7};

    Mat M2_30(4, 3, CV_32F, descriptors2_30);
    Mat M2_60(4, 3, CV_32F, descriptors2_60);

    ViewDetail vd2_30;
    vd2_30.angle = 30.0f;
    for (int i = 1; i <= 4; ++i)
    {
        ConfidenceDescriptor cd;
        cd.id = i;
        cd.confidence = confidence2_30[i-1];
        cd.descriptor = M2_30.row(i-1).clone();
        vd2_30.keypoints_descriptors.push_back(cd);
    }

    ViewDetail vd2_60;
    vd2_60.angle = 60.0f;

    vector<ConfidenceDescriptor> vec;

    for (int i = 1; i <= 4; ++i)
    {
        ConfidenceDescriptor cd;
        cd.id = i;
        cd.confidence = confidence2_60[i-1];
        cd.descriptor = M2_60.row(i-1).clone();

        vec.push_back(cd);

//        vd2_60.keypoints_descriptors.push_back(cd);
    }

    vd2_60.keypoints_descriptors.push_back(vec[3]);
    vd2_60.keypoints_descriptors.push_back(vec[2]);
    vd2_60.keypoints_descriptors.push_back(vec[1]);
    vd2_60.keypoints_descriptors.push_back(vec[0]);

    vector<ViewDetail> views2;
    views2.push_back(vd2_30);
    views2.push_back(vd2_60);


    MultiviewBodyModel mbm2(views2);

    vector<float> distances = view_distance(mbm1, mbm2);

    cout << "[";
    cout << distances[0];
    for (int j = 1; j < distances.size(); ++j)
    {
        cout << ", " << distances[j];

    }
    cout << "]" << endl;
}