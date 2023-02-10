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

    void Init(const char* model_path);

    cv::Mat Infer(cv::Mat& image);

    void Infer(const char* image_path, const char* heatmap_path);


private:
    graph_t graph_;
    options opt_;
};


#endif //HMAP_GENERATOR_HMAP_GENERATOR_H
