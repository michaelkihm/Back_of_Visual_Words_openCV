#include "backofvisualwords.hpp"


void BackOfVisualWordsClassifier::fit(vector<DataSample> &_dataset)
{
    //init 
    dataset = _dataset;
    class_per_image.reserve(dataset.size());
    descriptors_per_image.reserve(dataset.size());
    computeSiftFeatures();
    computeVocabulary();


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
    Mat k_labels, k_centers;
    int attempts = 5, iterationNumber = 15;//= 1e3;

	kmeans(*SIFT_descriptors, vocabulary_size, k_labels, 
            TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, iterationNumber, 1e-4), 
            attempts, KMEANS_PP_CENTERS, k_centers);

    //vocuabularyHistogram = Mat(descriptors_per_image.size(), vocabulary_size,CV_32FC1);
   
    for(auto &it : descriptors_per_image)
    {
	   vocuabularyHistogram.push_back(getHistogram(it,k_centers));
    }




}

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