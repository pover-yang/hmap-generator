#include <iostream>
#include "hmap_generator.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>


cv::Mat preprocess(const cv::Mat &image, float mean, float var, float scale, int zero_point) {
    cv::Mat dst_img;
//    float tmp = var * scale;
//    cv::resize(image, dst_img, cv::Size(640, 400));
//    dst_img.convertTo(dst_img, CV_32FC1, 1.0 / 255);
//    dst_img = (dst_img - cv::Scalar(mean)) / cv::Scalar(tmp) + cv::Scalar(zero_point);
//    dst_img.convertTo(dst_img, CV_8UC1);

    cv::resize(image, dst_img, cv::Size(640, 400));
    dst_img = dst_img * scale + zero_point;
    return dst_img;
}


cv::Mat postprocess(const cv::Mat &image, float scale, int zero_point) {
    cv::Mat dst_img;
//    image.convertTo(dst_img, CV_32FC3);
//    dst_img = (dst_img - zero_point) * scale;
//    dst_img.convertTo(dst_img, CV_8UC3, 255);
//    cv::resize(dst_img, dst_img, cv::Size(1280, 800));
    cv::resize(image, dst_img, cv::Size(1280, 800), 0, 0, cv::INTER_NEAREST);
    return dst_img;
}


int main(int argc, char **argv) {
    std::string model_path = argv[1];
    std::string image_path = argv[2];

    HeatMapGenerator hmap_generator = HeatMapGenerator();
    hmap_generator.Init(model_path, "cpu");
    float input_scale = hmap_generator.input_scale;
    int input_zero_point = hmap_generator.input_zero_point;
    float output_scale = hmap_generator.output_scale;
    int output_zero_point = hmap_generator.output_zero_point;

    cv::Mat src_img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    printf("image size (h,w,c): %d, %d, %d\n", src_img.rows, src_img.cols, src_img.channels());

    // calculate the time of inference
    int64 start = cv::getTickCount();
    cv::Mat in_img = preprocess(src_img, 0.4330, 0.2349, input_scale, input_zero_point);
    int64 end = cv::getTickCount();
    double preprocess_time = double(end - start) / cv::getTickFrequency();
    fprintf(stdout, "Preprocess time: %f ms\n", preprocess_time * 1000);

    start = cv::getTickCount();
    cv::Mat heatmap = hmap_generator.InferUInt8(in_img);
    end = cv::getTickCount();
    double inference_time = double(end - start) / cv::getTickFrequency();
    fprintf(stdout, "Inference time: %f ms\n", inference_time * 1000);

    start = cv::getTickCount();
    cv::Mat heatmap_post = postprocess(heatmap, output_scale, output_zero_point);
    end = cv::getTickCount();
    double time = double(end - start) / cv::getTickFrequency();
    fprintf(stdout, "Postprocess time: %f ms\n", time * 1000);

    cv::imwrite("heatmap.png", heatmap_post);

    cv::Mat blend_img;
    cv::cvtColor(src_img, src_img, cv::COLOR_GRAY2BGR);
    cv::cvtColor(heatmap_post, heatmap_post, cv::COLOR_RGB2BGR);
    cv::addWeighted(src_img, 0.5, heatmap_post, 0.5, 0, blend_img);
    cv::imshow("result", blend_img);
    cv::waitKey(0);

    return 0;
}

