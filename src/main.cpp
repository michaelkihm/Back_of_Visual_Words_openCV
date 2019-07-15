#include <iostream>
#include "datasethandler.hpp"
#include "backofvisualwords.hpp"

using namespace std;

int main()
{
   vector<DataSample> dataset, train, test;
   const string dataset_path { "/home/michael/Documents/Datasets/Caltech_101/"};
   vector<string> categories {"kangaroo", "dolphin", "chair", "soccer_ball"};

   dataset = createDataset(dataset_path, categories);
   tie(train, test) = train_test_split(dataset);


   cout << "finish"<<endl;
   return 0;




}