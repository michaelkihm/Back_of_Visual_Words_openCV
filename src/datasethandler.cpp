#include "datasethandler.hpp"

vector<DataSample> createDataset(const string &data_rooth_dir, vector< string> categories)
{
    string dataset_path;
    vector<DataSample> dataset;
    for(uint i =0; i < categories.size(); i++ )
    {
        dataset_path = data_rooth_dir + categories[i];
        addDataSamples(dataset_path, categories[i], i, dataset);
    }



    return dataset;
}

void addDataSamples( string &dir,  string &label_str,  uint label, vector<DataSample> &dataset)
{
    for (auto i = boost::filesystem::directory_iterator(dir); i != boost::filesystem::directory_iterator(); i++)
    {
        if (!boost::filesystem::is_directory(i->path())) //eliminate directories
        {
            dataset.push_back(DataSample(imread(i->path().string()),label, label_str ));
        }
        else
            continue;
    }
}

tuple< vector<DataSample>, vector<DataSample> > train_test_split(vector<DataSample> &dataset, float test_size)
{
    vector<DataSample> temp;
    copy(dataset.begin(), dataset.end(), back_inserter(temp));
    
    random_shuffle(temp.begin(), temp.end());
    vector<DataSample> test, train;
    for(uint i =0; i < temp.size(); i++)
    {
        if(i < static_cast<uint>(temp.size()*test_size))
            test.push_back(temp[i]);
        else
            train.push_back(temp[i]);
    }
    return make_tuple(train, test);

}
