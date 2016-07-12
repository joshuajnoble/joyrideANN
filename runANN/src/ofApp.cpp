#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
    
    neuralNetworkInputFilename = ofToDataPath("5_18/mlp.yaml", true);
    vocabularyInputFilename = ofToDataPath("5_18/vocabulary.yaml", true);
    classesInputFilename = ofToDataPath("5_18/classes.txt", true);
    //int portNumber = atoi(argv[4]);
    
    std::cout << "Loading models..." << std::endl;
    uint64 start = ofGetElapsedMillis();
    // Reading neural network
    mlp = cv::ml::ANN_MLP::load<cv::ml::ANN_MLP>(neuralNetworkInputFilename);
    // Read vocabulary
    cv::FileStorage fs(vocabularyInputFilename, cv::FileStorage::READ);
    fs["vocabulary"] >> vocabulary;
    fs.release();
    // Reading existing classes
    
    std::ifstream classesInput(classesInputFilename.c_str());
    std::string line;
    while (std::getline(classesInput, line))
    {
        std::stringstream ss;
        ss << line;
        int index;
        std::string classname;
        ss >> index;
        ss >> classname;
        classes[index] = classname;
    }
    std::cout << "Time elapsed in seconds: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() << std::endl;
    
    // Train FLANN
    std::cout << "Training FLANN..." << std::endl;
    start = ofGetElapsedMillis();
    
    flann.add(vocabulary);
    flann.train();
    std::cout << "Time elapsed in seconds: " << (ofGetElapsedMillis() - start)/1000 << std::endl;
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){
    stringstream ss;
    ss << "This is a " << result;
    ofDrawBitmapString(ss.str(), 100, 100);
    
    img.draw(100, 150);
}


void ofApp::dragEvent(ofDragInfo dragInfo) {

    if(dragInfo.files.size() > 0)
    {
        img.load(dragInfo.files[0]);
        processImage(dragInfo.files[0]);
    }
    

}
 
 /**
 * Extract local features for an image
 */
cv::Mat ofApp::getDescriptors(const cv::Mat& img)
{
    cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    kaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
    return descriptors;
}

/**
 * Get a histogram of visual words for an image
 */
cv::Mat ofApp::getBOWFeatures(cv::FlannBasedMatcher& flann, const cv::Mat& img, int vocabularySize)
{
    cv::Mat descriptors = getDescriptors(img);
    cv::Mat outputArray = cv::Mat::zeros(cv::Size(vocabularySize, 1), CV_32F);
    std::vector<cv::DMatch> matches;
    flann.match(descriptors, matches);
    for (size_t j = 0; j < matches.size(); j++)
    {
        int visualWord = matches[j].trainIdx;
        outputArray.at<float>(visualWord)++;
    }
    return outputArray;
}

/**
 * Receives a column matrix contained the probabilities associated to
 * each class and returns the id of column which contains the highest
 * probability
 */
int ofApp::getPredictedClass(const cv::Mat& predictions)
{
    float maxPrediction = predictions.at<float>(0);
    float maxPredictionIndex = 0;
    const float* ptrPredictions = predictions.ptr<float>(0);
    for (int i = 0; i < predictions.cols; i++)
    {
        float prediction = *ptrPredictions++;
        if (prediction > maxPrediction)
        {
            maxPrediction = prediction;
            maxPredictionIndex = i;
        }
    }
    return maxPredictionIndex;
}

/**
 * Get the predicted class for a sample
 */
int ofApp::getClass(const cv::Mat& bowFeatures, cv::Ptr<cv::ml::ANN_MLP> mlp)
{
    cv::Mat output;
    mlp->predict(bowFeatures, output);
    return getPredictedClass(output);
}

int ofApp::processImage( string file )
{
    
    cout << file << endl;
    
    ofPixels p;
    ofLoadImage(p, file);
    
    long millis = ofGetElapsedTimeMillis();
    
    //std::string filename(data_, std::find(data_, data_ + bytes_transferred, '\n') - 1);
    cv::Mat input = ofxCv::toCv(p);
    cv::Mat greyInput;
    
    if( input.channels() > 2 )
    {
        cv::cvtColor(input, greyInput, CV_BGR2GRAY);
    }
    else
    {
        greyInput = input;
    }
    
    if (!input.empty())
    {
        // Processing image
        cv::Mat bowFeatures = getBOWFeatures(flann, input, vocabulary.rows);
        cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
        int predictedClass = getClass(bowFeatures, mlp);
        switch(predictedClass)
        {
            case 0:
                result = "Left One Finger";
                break;
            case 1:
                result = "Right One Finger";
                break;
            case 2:
                result = "Left Two Finger";
                break;
            case 3:
                result = "Right Two Finger";
                break;
            case 4:
                result = "No Fingers";
                break;
        }
    }
    
    cout << " took " << ofGetElapsedTimeMillis() - millis << " millseconds " << endl;
    
}
