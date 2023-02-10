//
// Created by yjunj on 2023/2/10.
//

#include "hmap_generator.h"


void chw_to_hwc(cv::InputArray src, cv::OutputArray dst) {
  const auto& src_size = src.getMat().size;
  const int src_c = src_size[0];
  const int src_h = src_size[1];
  const int src_w = src_size[2];

  auto c_hw = src.getMat().reshape(0, {src_c, src_h * src_w});

  dst.create(src_h, src_w, CV_MAKETYPE(src.depth(), src_c));
  cv::Mat dst_1d = dst.getMat().reshape(src_c, {src_h, src_w});

  cv::transpose(c_hw, dst_1d);
}


void HeatMapGenerator::Init(const char *model_path) {
    /* init tengine */
    if (init_tengine() != 0) {
        fprintf(stderr, "Init tengine failed.\n");
        exit(1);
    }

    /*set runtime options*/
    opt_.num_thread = 1;
    opt_.cluster = TENGINE_CLUSTER_ALL;
    opt_.precision = TENGINE_MODE_FP32;
    opt_.affinity = 255;

    /* load model */
    graph_ = create_graph(NULL, "tengine", model_path);
    if (graph_ == nullptr) {
        fprintf(stderr, "Create graph failed.\n");
        exit(1);
    }
}


cv::Mat HeatMapGenerator::Infer(cv::Mat &image) {
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
    printf("heatmap size (h,w,c): %d, %d, %d\n", heatmap.rows, heatmap.cols, heatmap.channels());

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
