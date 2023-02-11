//
// Created by yjunj on 2023/2/10.
//

#include "hmap_generator.h"


void HeatMapGenerator::Init(const char *model_path) {
    /* init tengine */
    if (init_tengine() != 0) {
        fprintf(stderr, "Init tengine failed.\n");
        exit(1);
    }

    /*set runtime options*/
    opt_.num_thread = 4;
    opt_.cluster = TENGINE_CLUSTER_ALL;
    opt_.precision = TENGINE_MODE_UINT8;
    opt_.affinity = 255;

    /* create VeriSilicon TIM-VX backend */
    context_t timvx_context = create_context("timvx", 1);
    int rtt = set_context_device(timvx_context, "TIMVX", nullptr, 0);
    if (0 > rtt)
    {
        fprintf(stderr, " add_context_device VSI DEVICE failed.\n");
        exit(1);
    }

    /* load model */
    graph_ = create_graph(timvx_context, "tengine", model_path);
    if (graph_ == nullptr) {
        fprintf(stderr, "Create graph failed.\n");
        exit(1);
    }

    input_tensor_ = get_graph_input_tensor(graph_, 0, 0);
    if (input_tensor_ == nullptr) {
        fprintf(stderr, "Get input tensor failed\n");
        exit(1);
    }

    get_tensor_quant_param(input_tensor_, &input_scale, &input_zero_point, 1);

    int img_h = 400;
    int img_w = 640;
    int img_c = 1;
    input_buffer_size_ = img_h * img_w * img_c;
    int in_dims[4] = {1, img_c, img_h, img_w}; // nchw
    if (set_tensor_shape(input_tensor_, in_dims, 4) < 0) {
        fprintf(stderr, "Set input tensor shape failed\n");
        exit(1);
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph_, opt_) < 0) {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        exit(1);
    }

    output_tensor_ = get_graph_output_tensor(graph_, 0, 0);
    if (get_tensor_shape(output_tensor_, out_dim_, 4) < 0) {
        fprintf(stderr, "Get output tensor shape failed\n");
        exit(1);
    }
    get_tensor_quant_param(output_tensor_, &output_scale, &output_zero_point, 1);
}


cv::Mat HeatMapGenerator::InferFP32(cv::Mat &image) {
    /* prepare input data */
    int img_h = image.rows;
    int img_w = image.cols;
    int img_c = image.channels();
    int img_size = img_h * img_w * img_c;
    int in_dims[4] = {1, img_c, img_h, img_w}; // nchw
//    auto *input_data = (float *) malloc(img_size * sizeof(float));
    auto *image_data = (float *) image.data;
    int buffer_size = img_size * int(sizeof(float));

    if (image_data == nullptr) {
        fprintf(stderr, "Malloc input data failed.\n");
        exit(1);
    }

    /* set the shape, data buffer of input_tensor of the graph */
    tensor_t input_tensor = get_graph_input_tensor(graph_, 0, 0);
    if (input_tensor == nullptr) {
        fprintf(stderr, "Get input tensor failed\n");
        exit(1);
    }

    if (set_tensor_shape(input_tensor, in_dims, 4) < 0) {
        fprintf(stderr, "Set input tensor shape failed\n");
        exit(1);
    }

    if (set_tensor_buffer(input_tensor, image_data, buffer_size) < 0) {
        fprintf(stderr, "Set input tensor buffer failed\n");
        exit(1);
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph_, opt_) < 0) {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        exit(1);
    }

    /* run the graph */
    if (run_graph(graph_, 1) < 0) {
        fprintf(stderr, "Run graph failed.\n");
        exit(1);
    }

    /* get the shape, data buffer of output_tensor of the graph */
    tensor_t output_tensor = get_graph_output_tensor(graph_, 0, 0);
    if (output_tensor == nullptr) {
        fprintf(stderr, "Get output tensor failed\n");
        exit(1);
    }

    int out_dim[4];
    if (get_tensor_shape(output_tensor, out_dim, 4) < 0) {
        fprintf(stderr, "Get output tensor shape failed\n");
        exit(1);
    }

    auto *output_data = (float *) get_tensor_buffer(output_tensor);
    if (output_data == nullptr) {
        fprintf(stderr, "Get output data failed\n");
        exit(1);
    }

    cv::Mat heatmap = cv::Mat(out_dim[1], out_dim[2], CV_32FC3, output_data);
    fprintf(stdout, "heatmap size (h,w,c): %d, %d, %d\n", heatmap.rows, heatmap.cols, heatmap.channels());

    return heatmap;
}


cv::Mat HeatMapGenerator::InferUInt8(cv::Mat &image) {
    /* 1. set image data to input tensor */
    auto *image_data = image.data;
    if (set_tensor_buffer(input_tensor_, image_data, input_buffer_size_) < 0) {
        fprintf(stderr, "Set input tensor buffer failed\n");
        exit(1);
    }


    /* 2. run the graph */
    if (run_graph(graph_, 1) < 0) {
        fprintf(stderr, "Run graph failed.\n");
        exit(1);
    }

    /* 3. get the heatmap data from output tensor and transform to cv::Mat */
    auto output_uint8 = (uint8_t *) get_tensor_buffer(output_tensor_);
    if (output_uint8 == nullptr) {
        fprintf(stderr, "Get output data failed\n");
        exit(1);
    }

    cv::Mat heatmap = cv::Mat(out_dim_[1], out_dim_[2], CV_8UC3, output_uint8);
    fprintf(stdout, "heatmap size (h,w,c): %d, %d, %d\n", heatmap.rows, heatmap.cols, heatmap.channels());

    return heatmap;
}


void HeatMapGenerator::Infer(const char *image_path, const char *heatmap_path) {

}

HeatMapGenerator::~HeatMapGenerator() {
    /* release tengine */
    postrun_graph(graph_);
    destroy_graph(graph_);
    release_tengine();
}
