//
// Created by Mauro on 15/04/16.
//

#include <iostream>
#include <fstream>
#include "MultiviewBodyModel.h"

using namespace std;
using namespace cv;
using namespace multiviewbodymodel;

void test_distance(); // OK

int main()
{
    MultiviewBodyModel mbm1, mbm2;

    vector<int> ids;
    vector<float> confidences;
    Mat descriptors;

    // Testing distance
    mbm1.ReadAndCompute("../imgs/matteol_sync/c00000_skel.txt",
                        "../imgs/matteol_sync/c00000.png", 0, "SIFT", 7);
    mbm1.ReadAndCompute("../imgs/matteol_sync/l00000_skel.txt",
                        "../imgs/matteol_sync/l00000.png", 1, "SIFT", 7);

    mbm2.ReadAndCompute("../imgs/gianluca_sync/c00000_skel.txt",
                        "../imgs/gianluca_sync/c00000.png", 0, "SIFT", 7);
    mbm2.ReadAndCompute("../imgs/gianluca_sync/c00000_skel.txt",
                        "../imgs/gianluca_sync/c00000.png", 1, "SIFT", 7);

    vector<float> distances = mbm1.Distances(mbm2);

    cout << "[" << distances[0];
    for (int i = 1; i < distances.size(); ++i) {
        cout << ", " << distances[i];
    }
    cout << "]" << endl;

    // Printing
    //    ids = mbm1.views_id();
//    cout << "id = " << ids[0] << endl;
//    cout << "pose = " << mbm1.pose_side(0) << endl;
//
//    descriptors = mbm1.views_descriptors(0);
//    for (int j = 0; j < descriptors.rows; ++j) {
//        cout << j << ": " << descriptors.row(j) << endl;
//    }
//
//    confidences = mbm1.confidences(0);
//    cout << "[" << confidences[0];
//    for (int i = 1; i < confidences.size(); ++i) {
//        cout << ", " << confidences[i];
//    }
//    cout << "]" << endl;

//    test_distance(); // OK
    

    return 0;
}

void test_distance()
{
    // Body model 1
    float data1[4][3] = {1,0,1,2,0,1,4,3,2,6,2,1};
    float data2[4][3] = {0,0,1,7,4,0,2,0,1,1,0,0};

    Mat D1(4, 3, CV_32F, data1);
    Mat D2(4, 3, CV_32F, data2);

    vector<Mat> views_descriptors1;
    views_descriptors1.push_back(D1);
    views_descriptors1.push_back(D2);

    vector<float> c1;
    c1.push_back(0.1);
    c1.push_back(0.2);
    c1.push_back(0.3);
    c1.push_back(0.4);

    vector<float> c2;
    c2.push_back(0.4);
    c2.push_back(0.2);
    c2.push_back(0.1);
    c2.push_back(0.3);

    vector<vector<float> > views_descriptors_conf1;
    views_descriptors_conf1.push_back(c1);
    views_descriptors_conf1.push_back(c2);

    // Body model 2
    float data3[4][3] = {0,0,1,4,0,1,0,2,2,3,0,1};
    float data4[4][3] = {1,1,1,2,3,0,0,4,1,7,0,0};

    Mat D3(4, 3, CV_32F, data3);
    Mat D4(4, 3, CV_32F, data4);

    vector<Mat> views_descriptors2;
    views_descriptors2.push_back(D3);
    views_descriptors2.push_back(D4);

    vector<float> c3;
    c3.push_back(0.2);
    c3.push_back(0.2);
    c3.push_back(0.1);
    c3.push_back(0.5);

    vector<float> c4;
    c4.push_back(0.1);
    c4.push_back(0.1);
    c4.push_back(0.1);
    c4.push_back(0.7);

    vector<vector<float> > views_descriptors_conf2;
    views_descriptors_conf2.push_back(c3);
    views_descriptors_conf2.push_back(c4);

    vector<int> ids;
    ids.push_back(1);
    ids.push_back(2);

    MultiviewBodyModel mbm1;
    mbm1.set_views_id(ids);
    mbm1.set_views_descriptors(views_descriptors1);
    mbm1.set_views_descriptors_confidences(views_descriptors_conf1);

    MultiviewBodyModel mbm2;
    mbm2.set_views_id(ids);
    mbm2.set_views_descriptors(views_descriptors2);
    mbm2.set_views_descriptors_confidences(views_descriptors_conf2);

    // Testing change view: OK
//    mbm2.ChangeViewDescriptors(1, D1, c1);
//    mbm2.ChangeViewDescriptors(2, D2, c2);

    vector<float> distances = mbm1.Distances(mbm2);

    cout << "[" << distances[0];
    for (int i = 1; i < distances.size(); ++i) {
        cout << ", " << distances[i];
    }
    cout << "]";

    // output: [0.944803, 1.46327]
}
