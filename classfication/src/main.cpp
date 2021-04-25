#include <iostream>
#include <fstream> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <unistd.h>
#include <vector>
#include <array>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <stdio.h>
#include <map>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/istreamwrapper.h"


#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
//#include <core/providers/tensorrt/tensorrt_provider_factory.h>

using namespace cv;
using namespace std;

using namespace rapidjson;

Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "test" };
Ort::SessionOptions session_option{nullptr};

void GetOnnxModelInputInfo(Ort::Session& session_net, std::vector<const char*> &input_node_names, std::vector<int64_t> &input_node_dims, \
                                std::vector<const char*> &output_node_names, std::vector<int64_t> &output_node_dims)
{
    size_t num_input_nodes = session_net.GetInputCount();
    input_node_names.resize(num_input_nodes);
    //std::vector<int64_t> input_node_dims;
    //char* input_name = nullptr;

    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    printf("Number of inputs = %zu\n", num_input_nodes);
    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) 
    {
        // print input node names
        char* input_name = session_net.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = session_net.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
        {
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
        }

        //allocator.Free(input_name);
    }

    size_t num_output_nodes = session_net.GetOutputCount();
    output_node_names.resize(num_output_nodes);
    //std::vector<int64_t> output_node_dims;
    //char* output_name = nullptr;

    printf("Number of outputs = %zu\n", num_output_nodes);
    // iterate over all output nodes
    for (int i = 0; i < num_output_nodes; i++) 
    {
        // print output node names
        char* output_name = session_net.GetOutputName(i, allocator);
        printf("output %d : name=%s\n", i, output_name);
        output_node_names[i] = output_name;

        // print output node types
        Ort::TypeInfo type_info = session_net.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        // print output shapes/dims
        output_node_dims = tensor_info.GetShape();
        printf("output %d : num_dims=%zu\n", i, output_node_dims.size());
        for (int j = 0; j < output_node_dims.size(); j++)
        {
            printf("output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
        }

       //allocator.Free(output_name);
    }

}

template<typename T> 
vector<int> argsort(const std::vector<T>& array)
{
	const int array_len(array.size());
	vector<int> array_index(array_len, 0);
	for (int i = 0; i < array_len; i++)
		array_index[i] = i;

	sort(array_index.begin(), array_index.end(),
		[&array](int pos1, int pos2) {return (array[pos1] < array[pos2]);});

	return array_index;
}


int main(int argc, char* argv[])
{
    Ort::SessionOptions session_option;
    session_option.SetIntraOpNumThreads(1);
    session_option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(session_option, 0));
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

    std::string m_pModel_dir = "./alexnet.onnx";
    Ort::Session session_net(env, m_pModel_dir.c_str(), session_option);
    std::vector<const char*> m_InputNodeNames;
    std::vector<int64_t> m_InputNodeDims;
    std::vector<const char*> m_OutputNodeNames;
    std::vector<int64_t> m_OnputNodeDims;
    GetOnnxModelInputInfo(session_net, m_InputNodeNames, m_InputNodeDims, m_OutputNodeNames, m_OnputNodeDims);

    //读取分类标签信息
    ifstream f_val_label;
    f_val_label.open("./my_label.txt", ios::in);
    if (!f_val_label.is_open() || f_val_label.fail())
    {
        printf("my_label can't open, error!!!\n");
        return 0;
    }

    vector<int> m_VecLabelValue;
    while(!f_val_label.eof())
    {
        string tempValue;
        while(getline(f_val_label, tempValue))
        {
            //printf("tempValue:%s\n", tempValue.c_str());
            m_VecLabelValue.push_back(stoi(tempValue));
        }
    }
    f_val_label.close();
    printf("m_VecLabelValue.size: %d\n", (int)m_VecLabelValue.size());

    //得到文件夹下的所有文件名称
    vector<string> m_vecFilesName;
    DIR *dir = nullptr;
    struct dirent *ptr;
    string basePath = "/mnt/share/classfication_inference/img_val_resize";
    if ((dir = opendir(basePath.c_str())) == nullptr)
    {
        printf("opendir dir[%s] failed !!!\n", basePath.c_str());
        return 0;
    }
    while ((ptr = readdir(dir)) != NULL)
    {
        if((strcmp(ptr->d_name,".") == 0) || (strcmp(ptr->d_name,"..") == 0))    //current dir OR parrent dir
            continue;
        else if(ptr->d_type == 8)    //file
        {
            //printf("d_name:%s/%s\n", basePath.c_str(), ptr->d_name);
            m_vecFilesName.push_back(ptr->d_name);
        }
    }
    closedir(dir);
    sort(m_vecFilesName.begin(), m_vecFilesName.end());
    printf("m_vecFilesName.size: %d\n", (int)m_vecFilesName.size());

    //记录准确率
    int total_num = 0;
    int top1_num  = 0;
    int top5_num  = 0;

    map<string, string> m_results; //picname, Json

    for (auto picname : m_vecFilesName)
    {
        printf("%s\n", picname.c_str());
        total_num += 1;

        //cv读取图片数据为BGR格式
        string picPath = basePath + "/" + picname;
        printf("%s\n", picPath.c_str());

        size_t input_tensor_size = 1 * 3 * 224 * 224;
        std::vector<float> input_image_(input_tensor_size);
        std::array<int64_t, 4> input_shape_{ 1, 3, 224, 224 };
        float* output = input_image_.data();
        fill(input_image_.begin(), input_image_.end(), 0.f);
        Mat img = imread(picPath);
        if ((img.rows != 224) || (img.cols != 224))
        {
            resize(img, img, Size(224, 224));
        }
        int ws = img.rows;
        int hs = img.cols;
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < hs; i++) {
                for (int j = 0; j < ws; j++) {
                    if (c == 0) {
                        output[c*hs*ws + i * ws + j] = (img.ptr<uchar>(i)[j * 3 + c]) - 103.939;
                    }
                    if (c == 1) {
                        output[c*hs*ws + i * ws + j] = (img.ptr<uchar>(i)[j * 3 + c]) - 116.779;
                    }
                    if (c == 2) {
                        output[c*hs*ws + i * ws + j] = (img.ptr<uchar>(i)[j * 3 + c]) - 123.68;
                    }
                }
            }
        }

        Ort::Value input_tensor_net = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
        
        auto output_tensors_net = session_net.Run(Ort::RunOptions{nullptr}, m_InputNodeNames.data(), &input_tensor_net, 1, m_OutputNodeNames.data(), m_OutputNodeNames.size());
        printf("output_tensors_net.size: %d\n", (int)output_tensors_net.size());

        Ort::TypeInfo type_info = output_tensors_net[0].GetTypeInfo();
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        size_t tensor_size = tensor_info.GetElementCount();
        float* probs = output_tensors_net[0].GetTensorMutableData<float>();
        vector<float> out{probs, probs + tensor_size};

        // for (int i = 0; i < tensor_size; i++)
        // {
        //     printf(" %f ", out[i]);
        // }
        // printf("\n");

        vector<int> m_VecArgsort = argsort(out);
        // for (int i = 0; i < tensor_size; i++)
        // {
        //     printf(" %d ", m_VecArgsort[i]);
        // }
        // printf("\n");

        vector<int> m_VecSortIdx;
        m_VecSortIdx.assign(m_VecArgsort.rbegin(), m_VecArgsort.rend());
        // for (int i = 0; i < tensor_size; i++)
        // {
        //     printf(" %d ", m_VecSortIdx[i]);
        // }
        // printf("\n");

        vector<int> result(m_VecSortIdx.begin(), m_VecSortIdx.begin() + 5);
        // for (int i = 0; i < 5; i++)
        // {
        //     printf(" %d ", result[i]);
        // }
        // printf("\n");

        int picture_name_index = 0;
        sscanf(picname.c_str(), "ILSVRC2012_val_%d.jpg", &picture_name_index);
        // printf(" %d \n", picture_name_index);

        int label = m_VecLabelValue[picture_name_index-1];

        Document jsonDoc;
        Document::AllocatorType &allocator = jsonDoc.GetAllocator();
        jsonDoc.SetObject();

        Value value(kObjectType);
        stringstream ss;
        string testString;
        copy(result.begin(),result.end(), ostream_iterator<int>(ss," "));
        testString = ss.str();

        Value str_val;
        str_val.SetString(testString.c_str(), testString.length(), allocator);
        value.AddMember("result", str_val, allocator);
        value.AddMember("label", label, allocator);

        vector<int>::iterator it;
        it = find(result.begin(), result.end(), label);
        if (it == result.end())
        {
            //False
            value.AddMember("top5", "False", allocator);
        }
        else
        {
            //True
            top5_num += 1;
            value.AddMember("top5", "True", allocator);
        }
        
        if (label == m_VecSortIdx.front())
        {
            //True
            top1_num += 1;
            value.AddMember("top1", "True", allocator);
        }
        else
        {
            //False
            value.AddMember("top1", "False", allocator);
        }

        StringBuffer str;
        Writer<StringBuffer> writer(str);
        value.Accept(writer);
        string strJson = str.GetString();
        cout<<"value:"<<strJson.c_str()<<endl;

        char picture_name[1024];
        sscanf(picname.c_str(), "%[0-9a-zA-Z_]", picture_name);
        //cout << "picture_name: " << picture_name << endl;
        m_results.insert(make_pair(picture_name, strJson));

        printf("total: %d\n", total_num);
        printf("top1_accuracy_rate: %f\n", (float)top1_num/total_num);
        printf("top5_accuracy_rate: %f\n", (float)top5_num/total_num);

    }

    ofstream result_file;
    result_file.open("./onnx_run_reslut.txt", ios::out | ios::app);
    if (!result_file.is_open() || result_file.fail())
    {
        printf("result_file can't open, error!!!\n");
        return 0;
    }

    auto it = m_results.begin();
    while (it != m_results.end())
    {
        char temp[1024];
        sprintf(temp, "img[%s]  %s", it->first.c_str(), it->second.c_str());
        //cout << "picture_name: " << it->first.c_str() << endl;
        result_file << temp;
        result_file << endl;
        it++;
    }
    result_file.close();

    char pwd[1000];
    getcwd(pwd, 1000);
    printf("test dir: %s\n", pwd);
    printf("total: %d\n", total_num);
    printf("top1 hit: %d\n", top1_num);
    printf("top5 hit: %d\n", top5_num);
    printf("top1_accuracy_rate: %f\n", (float)top1_num/total_num);
    printf("top5_accuracy_rate: %f\n", (float)top5_num/total_num);

    return 0;
}