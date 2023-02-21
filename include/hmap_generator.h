//
// Created by yjunj on 2023/2/10.
//

#ifndef HMAP_GENERATOR_HMAP_GENERATOR_H
#define HMAP_GENERATOR_HMAP_GENERATOR_H


#include "c_api.h"
#include <opencv2/opencv.hpp>

class HeatMapGenerator {
public:
    HeatMapGenerator() = default;
    ~HeatMapGenerator();

    void Init(const std::string& model_path, const std::string& context_name);

    cv::Mat InferFP32(cv::Mat& image);

    cv::Mat InferUInt8(cv::Mat& image);

    void Infer(const char* image_path, const char* heatmap_path);

    float input_scale = 0.f;
    int input_zero_point = 0;
    float output_scale = 0.f;
    int output_zero_point = 0;

private:
    graph_t graph_;
    options opt_;
    tensor_t input_tensor_;
    tensor_t output_tensor_;
    int input_buffer_size_;
    int out_dim_[4];

};


#endif //HMAP_GENERATOR_HMAP_GENERATOR_H
