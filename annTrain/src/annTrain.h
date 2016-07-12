#pragma once

#include <boost/filesystem.hpp>

#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <fstream>

#include "ofxOpenCv.h"
#include "ofxCv.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

#include <boost/filesystem.hpp>
#include "ofMain.h"

//namespace fs = boost::filesystem;
typedef std::vector<std::string>::const_iterator vec_iter;

struct ImageData
{
    std::string classname;
    cv::Mat bowFeatures;
};

class annTrain : public ofBaseApp{

	public:
    
    // of functions
    void setup();
    void update();
    void draw();

    
    // our cv functions
    void processClassAndDesc(const std::string& classname, const cv::Mat& descriptors);
    std::vector<std::string> getFilesInDirectory(const std::string& directory);
    inline std::string getClassName(const std::string& filename);
    
    cv::Mat getDescriptors(const cv::Mat& img);
    void readImages(vec_iter begin, vec_iter end);
    void readImagesToTest(vec_iter begin, vec_iter end);
    
    int getClassId(const std::set<std::string>& classes, const std::string& classname);
    cv::Mat getClassCode(const std::set<std::string>& classes, const std::string& classname);
    cv::Mat getBOWFeatures(const cv::Mat& descriptors,int vocabularySize);
    
    cv::Ptr<cv::ml::ANN_MLP> getTrainedNeuralNetwork(const cv::Mat& trainSamples, const cv::Mat& trainResponses);
    int getPredictedClass(const cv::Mat& predictions);
    
    std::vector<std::vector<int> > getConfusionMatrix();
    void printConfusionMatrix(const std::vector<std::vector<int> >& confusionMatrix, const std::set<std::string>& classes);
    float getAccuracy(const std::vector<std::vector<int> >& confusionMatrix);
    void saveModels(const cv::Mat& vocabulary, const std::set<std::string>& classes);
    int train(std::string imagesDir, int networkInputSize, float trainSplitRatio);
    
    void processClassAndDescForTest(const std::string& classname, const cv::Mat& descriptors);
    
    // class variables
    cv::Mat descriptorsSet;
    std::vector<ImageData*> descriptorsMetadata;
    std::set<std::string> classes;
    cv::Ptr<cv::FlannBasedMatcher> flann;
    cv::Mat testSamples;
    std::vector<int> testOutputExpected;
    //
    cv::Mat trainSamples;
    cv::Mat trainResponses;
    
    cv::Ptr<cv::ml::ANN_MLP> mlp;

};
