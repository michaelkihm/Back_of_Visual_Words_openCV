#ifndef DATASET_HANDLER_H
#define DATASET_HANDLER_H

#include <string>
#include <vector>
#include <algorithm> 
#include <tuple>
#include "opencv2/opencv.hpp"
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;

struct DataSample{
     Mat image;
     uint label;
     string label_str;
    DataSample(Mat _image, uint _label, string _label_str):image(_image), label(_label), label_str(_label_str) { }

};

vector<DataSample> createDataset(const string &data_rooth_dir, vector<string> categories);
void addDataSamples( string &dir,  string &label_str,  uint label, vector<DataSample> &dataset);
tuple< vector<DataSample>, vector<DataSample> > train_test_split(vector<DataSample> &dataset, float test_size = 0.2);

#endif // DATASET_HANDLER_H
