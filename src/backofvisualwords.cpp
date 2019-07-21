#include "backofvisualwords.hpp"


void BackOfVisualWordsClassifier::fit(vector<DataSample> &_dataset)
{
    //init 
    dataset = _dataset;
    class_per_image.reserve(dataset.size());
    descriptors_per_image.reserve(dataset.size());
    computeSiftFeatures();
    computeVocabulary(); 
    trainClassifier();
    cout << "finished fitting the classifier!" << endl;


}


/** 
*  computeSiftFeatures. 
* Computes SIFT matrix for each test image
* and creates vocabularay
*/
void BackOfVisualWordsClassifier::computeSiftFeatures()
{
    SIFT_descriptors = make_unique<Mat>();
    Ptr<xfeatures2d::SIFT> siftptr;
	siftptr = xfeatures2d::SIFT::create();
    vector<KeyPoint> keypoints;
	Mat descriptors;
	
    for(auto& it : dataset)
    {
        siftptr->detectAndCompute(it.image, noArray(), keypoints, descriptors);
        SIFT_descriptors->push_back(descriptors);
        descriptors_per_image.push_back(descriptors);
        class_per_image.push_back(it.label);
    }
    cout << "rows: " << SIFT_descriptors->rows << " cols "  << SIFT_descriptors->cols << endl;



}

/** 
* computeVocabulary    
* Generate vocabulary with k-means clustering
*/
void BackOfVisualWordsClassifier::computeVocabulary()
{
    Mat k_labels;
    int attempts = 5, iterationNumber = 16;//1000;

	kmeans(*SIFT_descriptors, vocabulary_size, k_labels, 
            TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, iterationNumber, 1e-4), 
            attempts, KMEANS_PP_CENTERS, k_centers);

    //vocuabularyHistogram = Mat(descriptors_per_image.size(), vocabulary_size,CV_32FC1);
   
    for(auto &it : descriptors_per_image)
    {
	   vocuabulary_histogram.push_back(getHistogram(it,k_centers));
    }




}

/** 
* getHistogram    
* computes histogram of visual words for one training image
*/
Mat BackOfVisualWordsClassifier::getHistogram(Mat &SIFT_descriptors, Mat &centers)
{
    Mat histogram(1,vocabulary_size,CV_32FC1);
    BFMatcher matcher;
	vector<DMatch> matches;
    matcher.match(SIFT_descriptors, centers, matches);

    for (uint i =0; i < matches.size(); i++) 
    {
		histogram.at<float>(0, matches.at(i).trainIdx) +=  + 1;
	}
    return histogram;

}


/** 
* trainClassifier    
* creates openCV training data struct and trains the classifier
*/
void BackOfVisualWordsClassifier::trainClassifier()
{
    //create trainings data
    assert((int)class_per_image.size() == vocuabulary_histogram.rows);
    Mat test(class_per_image.size(),1,CV_32S, class_per_image.data());
    // for(uint i=0; i < class_per_image.size(); i++)
    //     test.at<int>(i,0) = class_per_image[i];


    //cout << test << endl;
    Ptr<TrainData> train_data = TrainData::create(vocuabulary_histogram, ml::ROW_SAMPLE, test);
  

    svm_calssifier = SVM::create();
	svm_calssifier->setType(SVM::C_SVC);
	svm_calssifier->setKernel(SVM::LINEAR);
	svm_calssifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e4, 1e-6));
	// Train the SVM with given parameters
	
    svm_calssifier->train(train_data);

}


/** 
*  computeSiftFeatures. 
* Computes SIFT matrix for each test image
* and creates vocabularay
*/
  vector<uint> BackOfVisualWordsClassifier::predict(vector<DataSample>& test_data)
 {
    Ptr<xfeatures2d::SIFT> siftptr;
	siftptr = xfeatures2d::SIFT::create();
    Mat descriptors, image_histogram;
    vector<KeyPoint> keypoints;
    vector<uint> predictions;
    predictions.reserve(test_data.size());

    for(auto& it : test_data)
    {
        siftptr->detectAndCompute(it.image, noArray(), keypoints, descriptors);
        image_histogram = getHistogram(descriptors,k_centers);
        assert(image_histogram.cols == vocuabulary_histogram.cols && image_histogram.rows == 1);
        predictions.push_back( static_cast<uint>(svm_calssifier->predict(image_histogram)) );

    }

    return predictions;

 }
