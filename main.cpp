#include <iostream>
#include "hmap_generator.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

cv::Mat preprocess(const cv::Mat &image) {
    printf("image size (h,w,c): %d, %d, %d\n", image.rows, image.cols, image.channels());
    printf("image channels: %d\n", image.channels());

    cv::Mat dst_img;
    cv::resize(image, dst_img, cv::Size(640, 400));
    dst_img.convertTo(dst_img, CV_32FC1, 1.0 / 255);
    dst_img = dst_img - cv::Scalar(0.4430);
    dst_img = dst_img / cv::Scalar(0.2349);

    return dst_img;
}


cv::Mat postprocess(const cv::Mat &image) {
    cv::Mat dst_img;
    image.convertTo(dst_img, CV_8UC1, 255);
    cv::resize(dst_img, dst_img, cv::Size(1280, 800));
    return dst_img;
}


int main() {
    HeatMapGenerator hmap_generator = HeatMapGenerator();
    hmap_generator.Init("../test/hmap-v3-e99.tmfile");

    cv::Mat src_img = cv::imread("../test/barcode-test.png", cv::IMREAD_GRAYSCALE);
    cv::Mat in_img = preprocess(src_img);

    cv::Mat heatmap = hmap_generator.Infer(in_img);
    cv::Mat heatmap_post = postprocess(heatmap);

    cv::Mat blend_img;
    cv::cvtColor(src_img, src_img, cv::COLOR_GRAY2BGR);
    cv::cvtColor(heatmap_post, heatmap_post, cv::COLOR_RGB2BGR);
    cv::addWeighted(src_img, 0.5, heatmap_post, 0.5, 0, blend_img);
    cv::imshow("result", blend_img);
    cv::waitKey(0);

    return 0;
}

