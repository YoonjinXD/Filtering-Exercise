//
//  main.cpp
//  Filtering
//
//  Created by Yoonjin Chung on 30/11/2018.
//  Copyright © 2018 YJ. All rights reserved.
//

#include "main.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;
Mat image;

static Mat GradientMag(Mat src){
    Mat sx, sy, magn;
    Sobel( src, sx, CV_32F, 1, 0, 3, BORDER_DEFAULT);
    Sobel( src, sy, CV_32F, 0, 1, 3, BORDER_DEFAULT);
    cv::magnitude(sx, sy, magn);
    return magn;
}

static int CalMedian(Mat input){
    int sum = 0;
    for(int i=0;i<input.rows;i++){
        for(int j=0;j<input.cols;j++){
            sum += input.at<uchar>(i, j);
        }
    }
    return sum/(input.rows * input.cols);
}

int main()
{
    // [PART 0] Image setting and Declaration variables
    image = imread("/Users/yoonjinchung/Desktop/lenna.jpg", 1);
    if(image.empty())
    {
        printf("Cannot read image file\n");
        return -1;
    }
    
    // [PART 1] Gaussian Filtering
    Mat gaus_1, gaus_2, gaus_4, gaus_8;
    GaussianBlur( image, gaus_1, Size( 5, 5 ), 1);
    imshow("Gaussian Filtering: sigma = 1", gaus_1);
    GaussianBlur( image, gaus_2, Size( 5, 5 ), 2);
    imshow("Gaussian Filtering: sigma = 2", gaus_2);
    GaussianBlur( image, gaus_4, Size( 5, 5 ), 4);
    imshow("Gaussian Filtering: sigma = 4", gaus_4);
    GaussianBlur( image, gaus_8, Size( 5, 5 ), 8);
    imshow("Gaussian Filtering: sigma = 8", gaus_8);
    
    // [PART 2] Gradient Magnitude Computation
    // You can see the function 'GradientMag' above the main()
    imshow("Gradient Magnitude: sigma = 1", GradientMag(gaus_1));
    imshow("Gradient Magnitude: sigma = 2", GradientMag(gaus_2));
    imshow("Gradient Magnitude: sigma = 4", GradientMag(gaus_4));
    imshow("Gradient Magnitude: sigma = 8", GradientMag(gaus_8));
    
    // [PART 3] Laplacian-Gaussian Filtering
    Mat temp; // The temporary value for the results.
    Laplacian( gaus_1, temp, CV_16S, 5, 1, 0, BORDER_DEFAULT);
    imshow("Laplacian-Gaussian: sigma = 1", temp);
    Laplacian( gaus_2, temp, CV_16S, 5, 1, 0, BORDER_DEFAULT);
    imshow("Laplacian-Gaussian: sigma = 2", temp);
    Laplacian( gaus_4, temp, CV_16S, 5, 1, 0, BORDER_DEFAULT);
    imshow("Laplacian-Gaussian: sigma = 4", temp);
    Laplacian( gaus_8, temp, CV_16S, 5, 1, 0, BORDER_DEFAULT);
    imshow("Laplacian-Gaussian: sigma = 8", temp);
    
    // [PART 4] Canny-Edge Detection
    // 1. Seting the environment
    int edgeThresh = 50;
    Mat canny_gray, detected_edges, cedge;
    cedge.create(image.size(), image.type());
    cvtColor(image, canny_gray, COLOR_BGR2GRAY);
    namedWindow("My Canny-Eddge", 1);
    
    // 2. Finding the optimal thresholds
    int median = CalMedian(canny_gray);
    float sigma = 0.35;
    int lower = fmax(0, (1.0 - sigma) * median);
    int upper = fmin(255, (1.0 + sigma) * median);
    
    // 3. Run the edge detector
    blur( canny_gray, detected_edges, Size(3,3) );
    Canny( detected_edges, detected_edges, lower, upper, 3);
    cedge = Scalar::all(0);
    image.copyTo( cedge, detected_edges );
    imshow("My Canny-Edge", cedge);
    
    waitKey(0);
    return 0;
}
