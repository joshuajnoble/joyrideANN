#pragma once

#include <vector>
#include <set>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ofxCv.h"


//using boost::asio::ip::tcp;
//
//struct Context
//{
//    cv::Mat vocabulary;
//    cv::FlannBasedMatcher flann;
//    std::map<int, std::string> classes;
//    cv::Ptr<cv::ml::ANN_MLP> mlp;
//};


#include "ofMain.h"

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();
    
    void dragEvent(ofDragInfo dragInfo);

    cv::Mat getDescriptors(const cv::Mat& img);
    cv::Mat getBOWFeatures(cv::FlannBasedMatcher& flann, const cv::Mat& img, int vocabularySize);
    int getPredictedClass(const cv::Mat& predictions);
    int getClass(const cv::Mat& bowFeatures, cv::Ptr<cv::ml::ANN_MLP> mlp);
    int processImage( string file );
		
    
    cv::Ptr<cv::ml::ANN_MLP> mlp;
    cv::Mat vocabulary;
    std::map<int, std::string> classes;
    cv::FlannBasedMatcher flann;
    
    std::string neuralNetworkInputFilename;
    std::string vocabularyInputFilename;
    std::string classesInputFilename;
    
    string result;
    ofImage img;
    
};
