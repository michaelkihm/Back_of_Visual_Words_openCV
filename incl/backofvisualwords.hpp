#ifndef BOW_H
#define BOW_H

#include <string>
#include <vector>
#include <iostream>
#include <memory>

#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

#include "datasethandler.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

class BackOfVisualWordsClassifier
{
    public:
    BackOfVisualWordsClassifier() : vocabulary_size(320) { }
    BackOfVisualWordsClassifier(const int _vocabulary_size):vocabulary_size(_vocabulary_size) {}
    void fit();
    void fit(vector<DataSample>& _dataset);
    vector<uint> predict(vector<DataSample>& test_data);
    vector<uint> predict();
    
    private:
    vector<DataSample> dataset;
    const int vocabulary_size;
    vector<int> class_per_image;
    vector<Mat> descriptors_per_image;
    unique_ptr<Mat> SIFT_descriptors;
    Mat k_centers; //centers of kmeans clustering
    Mat vocuabulary_histogram;
    Ptr<SVM> svm_calssifier;

    //methods
    void computeSiftFeatures();
    void computeVocabulary();
    Mat getHistogram(Mat &SIFT_descriptors, Mat &centers);
    void trainClassifier();



};

#endif // BOW_H
