
#include "annTrain.h"


const int networkInputSize = 512;

void annTrain::setup()
{
    //train(ofToDataPath("fingers/all", true), 512, 0.7);

}

void annTrain::update()
{
}

void annTrain::draw()
{
    ofSetColor(255, 255, 255);
    kazeImage.draw(0, 0);
    ofNoFill();
    ofSetColor(0, 255, 255);
    for ( int i = 0; i < kazeKeypoints.size(); i++ )
    {
        ofDrawCircle(kazeKeypoints.at(i).pt.x, kazeKeypoints.at(i).pt.y, 5);
    }
}

void annTrain::dragEvent(ofDragInfo dragInfo )
{
    
    cv::Mat mat;
    processImage( dragInfo.files[0] );
}


void annTrain::processClassAndDescForTest(const std::string& classname, const cv::Mat& descriptors)
{
    // Get histogram of visual words using bag of words technique
    cv::Mat bowFeatures = getBOWFeatures(descriptors, networkInputSize);
    cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
    testSamples.push_back(bowFeatures);
    testOutputExpected.push_back(getClassId(classes, classname));
};


void annTrain::processClassAndDesc(const std::string& classname, const cv::Mat& descriptors)
{
    // Append to the set of classes
    classes.insert(classname);
    // Append to the list of descriptors
    descriptorsSet.push_back(descriptors);
    // Append metadata to each extracted feature
    ImageData* data = new ImageData;
    data->classname = classname;
    data->bowFeatures = cv::Mat::zeros(cv::Size(networkInputSize, 1), CV_32F);
    for (int j = 0; j < descriptors.rows; j++)
    {
        descriptorsMetadata.push_back(data);
    }
}



//Get all files in directory (not recursive)
std::vector<std::string> annTrain::getFilesInDirectory(const std::string& directory)
{

    std::vector<std::string> files;
    filesystem::path root(directory);
    filesystem::directory_iterator it_end;
    for (filesystem::directory_iterator it(root); it != it_end; ++it)
    {
        if (filesystem::is_regular_file(it->path()))
        {
            files.push_back(it->path().string());
        }
    }
    return files;

}

//Extract the class name from a file name
std::string annTrain::getClassName(const std::string& filename)
{
//    cout << filename << endl;
    
    if( filename.find("1l") != std::string::npos )
    {
        return "1l";
    }
    if( filename.find("2l") != std::string::npos )
    {
        return "2l";
    }
    if( filename.find("1r") != std::string::npos )
    {
        return "1r";
    }
    if( filename.find("2r") != std::string::npos )
    {
        return "2r";
    }
    if( filename.find("nf") != std::string::npos )
    {
        return "nf";
    }
}

// get local features for an image
cv::Mat annTrain::getDescriptors(const cv::Mat& img)
{
    cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    //kaze->detectAndCompute(dst, cv::noArray(), keypoints, descriptors);
    kaze->detect(img, keypoints);
    kaze->compute(img, keypoints, descriptors);
    return descriptors;
}

//Read images from a list of file names and gets class name and its local descriptors
void annTrain::readImages(vec_iter begin, vec_iter end)
{
    for (auto it = begin; it != end; ++it)
    {
        std::string filename = *it;
        std::cout << "Reading image " << filename << "..." << std::endl;
        cv::Mat img = cv::imread(filename, 0);
        if (img.empty())
        {
            std::cerr << " Could not read image " << filename << std::endl;
            continue;
        }
        std::string classname = getClassName(filename);
        cv::Mat descriptors = getDescriptors(img);
        processClassAndDesc(classname, descriptors);
    }
}

//Read images from a list of file names and returns, for each read image for testing
void annTrain::readImagesToTest(vec_iter begin, vec_iter end)
{
    for (auto it = begin; it != end; ++it)
    {
        std::string filename = *it;
        std::cout << "Reading image " << filename << "..." << std::endl;
        cv::Mat img = cv::imread(filename, 0);
        if (img.empty())
        {
            std::cerr << "WARNING: Could not read image." << std::endl;
            continue;
        }
        std::string classname = getClassName(filename);
        cv::Mat descriptors = getDescriptors(img);
        processClassAndDescForTest(classname, descriptors);
    }
}

//Transform a class name into an id
int annTrain::getClassId(const std::set<std::string>& classes, const std::string& classname)
{
    int index = 0;
    for (auto it = classes.begin(); it != classes.end(); ++it)
    {
        if (*it == classname) break;
        ++index;
    }
    return index;
}

//Get a binary code associated to a class
cv::Mat annTrain::getClassCode(const std::set<std::string>& classes, const std::string& classname)
{
    cv::Mat code = cv::Mat::zeros(cv::Size((int)classes.size(), 1), CV_32F);
    int index = getClassId(classes, classname);
    code.at<float>(index) = 1;
    return code;
}

//Turn local features into a single bag of words histogram of
cv::Mat annTrain::getBOWFeatures(const cv::Mat& descriptors, int vocabularySize)
{
    cv::Mat outputArray = cv::Mat::zeros(cv::Size(vocabularySize, 1), CV_32F);
    std::vector<cv::DMatch> matches;
    flann->match(descriptors, matches);
    for (size_t j = 0; j < matches.size(); j++)
    {
        int visualWord = matches[j].trainIdx;
        outputArray.at<float>(visualWord)++;
    }
    return outputArray;
}

//create the trained neural network
cv::Ptr<cv::ml::ANN_MLP> annTrain::getTrainedNeuralNetwork(const cv::Mat& trainSamples,
                                                 const cv::Mat& trainResponses)
{
    int networkInputSize = trainSamples.cols;
    int networkOutputSize = trainResponses.cols;
    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
    std::vector<int> layerSizes = { networkInputSize, networkInputSize / 2,
        networkOutputSize };
    mlp->setLayerSizes(layerSizes);
    mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
    mlp->train(trainSamples, cv::ml::ROW_SAMPLE, trainResponses);
    return mlp;
}


int annTrain::getPredictedClass(const cv::Mat& predictions)
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

std::vector<std::vector<int> > annTrain::getConfusionMatrix()
{
    cv::Mat testOutput;
    mlp->predict(testSamples, testOutput);
    // we now have 5 classes
    std::vector<std::vector<int> > confusionMatrix(5, std::vector<int>(5));
    for (int i = 0; i < testOutput.rows; i++)
    {
        int predictedClass = getPredictedClass(testOutput.row(i));
        int expectedClass = testOutputExpected.at(i);
        cout << expectedClass << " " << predictedClass;
        confusionMatrix[expectedClass][predictedClass]++;
    }
    return confusionMatrix;
}

void annTrain::printConfusionMatrix(const std::vector<std::vector<int> >& confusionMatrix, const std::set<std::string>& classes)
{
    for (auto it = classes.begin(); it != classes.end(); ++it)
    {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < confusionMatrix.size(); i++)
    {
        for (size_t j = 0; j < confusionMatrix[i].size(); j++)
        {
            std::cout << confusionMatrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// get the accuracy of the model
float annTrain::getAccuracy(const std::vector<std::vector<int> >& confusionMatrix)
{
    int hits = 0;
    int total = 0;
    for (size_t i = 0; i < confusionMatrix.size(); i++)
    {
        for (size_t j = 0; j < confusionMatrix.at(i).size(); j++)
        {
            if (i == j) hits += confusionMatrix.at(i).at(j);
            total += confusionMatrix.at(i).at(j);
        }
    }
    return hits / (float)total;
}

void annTrain::saveModels(const cv::Mat& vocabulary, const std::set<std::string>& classes)
{
    mlp->save(ofToDataPath("mlp.yaml", true));
    cv::FileStorage fs(ofToDataPath("vocabulary.yaml", true), cv::FileStorage::WRITE);
    fs << "vocabulary" << vocabulary;
    fs.release();
    std::ofstream classesOutput(ofToDataPath("classes.txt", true));
    for (auto it = classes.begin(); it != classes.end(); ++it)
    {
        classesOutput << getClassId(classes, *it) << "\t" << *it << std::endl;
    }
    classesOutput.close();
}

int annTrain::train(std::string imagesDir, int networkInputSize, float testRatio)
{

    std::cout << "Reading training set..." << std::endl;
    uint64 start = ofGetElapsedTimeMillis();
    std::vector<std::string> files = getFilesInDirectory(imagesDir);
    std::random_shuffle(files.begin(), files.end());
    
    cv::Mat img;
    
    for (auto it = files.begin(); it != files.end(); ++it)
    {
        std::string filename = *it;
        //std::cout << "Reading image " << filename << "..." << std::endl;
        img = cv::imread(filename, 0);

        if (img.empty())
        {
            std::cerr << "WARNING: Could not read image." << std::endl;
            continue;
        }
        std::string classname = getClassName(filename);
        cv::Mat descriptors = getDescriptors(img);
        processClassAndDesc(classname, descriptors);
    }
    
    std::cout << " Seconds : " << (ofGetElapsedTimeMillis() - start) / 1000.0 << std::endl;
    
    std::cout << "Creating vocabulary..." << std::endl;
    start = ofGetElapsedTimeMillis();
    cv::Mat labels;
    cv::Mat vocabulary;
    // Use k-means to find k centroids (the words of our vocabulary)
    cv::kmeans(descriptorsSet, networkInputSize, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 0.01), 1, cv::KMEANS_PP_CENTERS, vocabulary);
    // No need to keep it on memory anymore
    descriptorsSet.release();
    std::cout << " Seconds : " << (ofGetElapsedTimeMillis() - start) / 1000.0 << std::endl;
    
    // Convert a set of local features for each image in a single descriptors
    // using the bag of words technique
    std::cout << "Getting histograms of visual words..." << std::endl;
    int* ptrLabels = (int*)(labels.data);
    int size = labels.rows * labels.cols;
    for (int i = 0; i < size; i++)
    {
        int label = *ptrLabels++;
        ImageData* data = descriptorsMetadata[i];
        data->bowFeatures.at<float>(label)++;
    }
    
    // Filling matrixes to be used by the neural network
    std::cout << "Preparing neural network..." << std::endl;
    std::set<ImageData*> uniqueMetadata(descriptorsMetadata.begin(), descriptorsMetadata.end());
    for (auto it = uniqueMetadata.begin(); it != uniqueMetadata.end(); )
    {
        ImageData* data = *it;
        cv::Mat normalizedHist;
        cv::normalize(data->bowFeatures, normalizedHist, 0, data->bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
        trainSamples.push_back(normalizedHist);
        trainResponses.push_back(getClassCode(classes, data->classname));
        delete *it; // clear memory
        it++;
    }
    descriptorsMetadata.clear();
    
    // Training neural network
    std::cout << "Training neural network..." << std::endl;
    start = ofGetElapsedTimeMillis();
    mlp = getTrainedNeuralNetwork(trainSamples, trainResponses);
    std::cout << " Seconds : " << (ofGetElapsedTimeMillis() - start) / 1000.0 << std::endl;
    
    // We can clear memory now
    trainSamples.release();
    trainResponses.release();
    
    // Train FLANN
    std::cout << "Training FLANN..." << std::endl;
    start = ofGetElapsedTimeMillis();
    
    flann = cv::Ptr<cv::FlannBasedMatcher>(new cv::FlannBasedMatcher());
    
    flann->add(vocabulary);
    flann->train();
    std::cout << " Seconds : " << (ofGetElapsedTimeMillis() - start) / 1000.0 << std::endl;
    
    // Reading test set
    std::cout << "Reading test set..." << std::endl;
    start = ofGetElapsedTimeMillis();
    readImagesToTest(files.begin() + (size_t)(files.size() * testRatio), files.end());
    std::cout << " Seconds : " << (ofGetElapsedTimeMillis() - start) / 1000.0 << std::endl;
    
    // Get confusion matrix of the test set
    std::vector<std::vector<int> > confusionMatrix = getConfusionMatrix();
    
    // how accurate is our model
    std::cout << "Confusion matrix " << std::endl;
    printConfusionMatrix(confusionMatrix, classes);
    std::cout << "Accuracy " << getAccuracy(confusionMatrix) << std::endl;
    
    // now save everything
    std::cout << "saving models" << std::endl;
    saveModels(vocabulary, classes);
    
    return 0;
}

void annTrain::processImage( string file )
{
    
    cout << file << endl;
    
    kazeImage.load(file);
    
    long millis = ofGetElapsedTimeMillis();
    
    cv::Mat input = ofxCv::toCv(kazeImage.getPixels());
    cv::Mat greyInput;
    
    if( input.channels() > 2 )
    {
        cv::cvtColor(input, greyInput, CV_BGR2GRAY);
    }
    else
    {
        greyInput = input;
    }

    cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();

    kaze->detect(greyInput, kazeKeypoints);
    kaze->compute(greyInput, kazeKeypoints, kazeDescriptors);

    
}