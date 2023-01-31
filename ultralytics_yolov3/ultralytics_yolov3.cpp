#include <iostream>


#include "memory"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <fstream>
#include <iterator>
#include <algorithm>


typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            if(inter > 0){
                float ovr = inter / (vArea[i] + vArea[j] - inter);
                if (ovr >= NMS_THRESH) {
                    input_boxes.erase(input_boxes.begin() + j);
                    vArea.erase(vArea.begin() + j);
                }
                else {
                    j++;
                }
            }else{
                j++;
            }

        }
    }
}

const int coco_color_list[80][3] =
        {
                //{255 ,255 ,255}, //bg
                {170 ,  0 ,255},
                { 84 , 84 ,  0},
                { 84 ,170 ,  0},
                { 84 ,255 ,  0},
                {170 , 84 ,  0},
                {170 ,170 ,  0},
                {118 ,171 , 47},
                {238 , 19 , 46},
                {216 , 82 , 24},
                {236 ,176 , 31},
                {125 , 46 ,141},
                { 76 ,189 ,237},
                { 76 , 76 , 76},
                {153 ,153 ,153},
                {255 ,  0 ,  0},
                {255 ,127 ,  0},
                {190 ,190 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 ,255},

                {170 ,255 ,  0},
                {255 , 84 ,  0},
                {255 ,170 ,  0},
                {255 ,255 ,  0},
                {  0 , 84 ,127},
                {  0 ,170 ,127},
                {  0 ,255 ,127},
                { 84 ,  0 ,127},
                { 84 , 84 ,127},
                { 84 ,170 ,127},
                { 84 ,255 ,127},
                {170 ,  0 ,127},
                {170 , 84 ,127},
                {170 ,170 ,127},
                {170 ,255 ,127},
                {255 ,  0 ,127},
                {255 , 84 ,127},
                {255 ,170 ,127},
                {255 ,255 ,127},
                {  0 , 84 ,255},
                {  0 ,170 ,255},
                {  0 ,255 ,255},
                { 84 ,  0 ,255},
                { 84 , 84 ,255},
                { 84 ,170 ,255},
                { 84 ,255 ,255},
                {170 ,  0 ,255},
                {170 , 84 ,255},
                {170 ,170 ,255},
                {170 ,255 ,255},
                {255 ,  0 ,255},
                {255 , 84 ,255},
                {255 ,170 ,255},
                { 42 ,  0 ,  0},
                { 84 ,  0 ,  0},
                {127 ,  0 ,  0},
                {170 ,  0 ,  0},
                {212 ,  0 ,  0},
                {255 ,  0 ,  0},
                {  0 , 42 ,  0},
                {  0 , 84 ,  0},
                {  0 ,127 ,  0},
                {  0 ,170 ,  0},
                {  0 ,212 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 , 42},
                {  0 ,  0 , 84},
                {  0 ,  0 ,127},
                {  0 ,  0 ,170},
                {  0 ,  0 ,212},
                {  0 ,  0 ,255},
                {  0 ,  0 ,  0},
                { 36 , 36 , 36},
                { 72 , 72 , 72},
                {109 ,109 ,109},
                {145 ,145 ,145},
                {182 ,182 ,182},
                {218 ,218 ,218},
                {  0 ,113 ,188},
                { 80 ,182 ,188},
                {127 ,127 ,  0},
        };


void draw_coco_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes)
{
    static const char* class_names[] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                         "train", "truck", "boat", "traffic light", "fire hydrant",
                                         "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                         "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                         "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                         "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                         "baseball glove", "skateboard", "surfboard", "tennis racket",
                                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                         "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                         "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                         "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                         "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                         "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                         "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    cv::Mat image = bgr;
    int src_w = image.cols;
    int src_h = image.rows;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(coco_color_list[bbox.label][0],
                                      coco_color_list[bbox.label][1],
                                      coco_color_list[bbox.label][2]);
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
        //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);

        cv::rectangle(image, cv::Point(bbox.x1, bbox.y1), cv::Point(bbox.x2, bbox.y2), color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = bbox.x1;
        int y = bbox.y1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }
//    cv::imwrite("/Users/yang/CLionProjects/test_openvino/images/det_image.jpg", image);
    cv::imshow("image", image);
}


cv::Mat resize_padding(cv::Mat &origin_image, float &ratio, int target_h, int target_w, int constant_value) {
    cv::Mat out_image(target_h, target_w, CV_8UC3);
    int width = origin_image.cols, height = origin_image.rows;
    ratio = std::min((float)target_w /(float)width, (float)target_h / (float)height);

    if(width == target_w && height == target_h)
    {
        origin_image.copyTo(out_image);
        return out_image;
    }

    memset(out_image.data, constant_value, target_h * target_w * 3);
    int new_width = (int)(ratio * (float )width), new_height = (int)(ratio * (float )height);

    cv::Mat resize_image;
    cv::resize(origin_image, resize_image, cv::Size(new_width, new_height));
    // resize_image替换rect_image的同时,也会替换out_image中的对应部分
    cv::Mat rect_image = out_image(cv::Rect(0, 0, new_width, new_height));
    resize_image.copyTo(rect_image);
    return out_image;
}


int main() {

    std::string image_file("/Users/yang/CLionProjects/test_tvm/images/bus.jpeg");
    cv::Mat image = cv::imread(image_file);
    float ratio = 0.0;
    cv::Mat input_image = resize_padding(image, ratio, 640, 640, 114);
    input_image.convertTo(input_image, CV_32F, 1.0/255.0);

    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile("/Users/yang/CLionProjects/test_tvm/ultralytics_yolov3/yolov3_tvm/ultralytics_yolov3.so");
    // json graph
    std::ifstream json_in("/Users/yang/CLionProjects/test_tvm/ultralytics_yolov3/yolov3_tvm/ultralytics_yolov3.json", std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();

    // parameters in binary
    std::ifstream params_in("/Users/yang/CLionProjects/test_tvm/ultralytics_yolov3/yolov3_tvm/ultralytics_yolov3.params", std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    // parameters need to be TVMByteArray type to indicate the binary data
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;

    // get global function module for graph runtime
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(json_data, mod_syslib, device_type, device_id);

    DLTensor* x;
    int in_ndim = 4;
    int64_t in_shape[4] = {1, 3, 640, 640};
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);

    cv::Mat image_channels[3];
    cv::split(input_image, image_channels);
    for (int j = 0; j < 3; j++) {
        memcpy((float*)(x->data) + 640*640 * j, image_channels[j].data,640*640 * sizeof(float));
    }

    // load image data saved in binary
//    std::ifstream data_fin("cat.bin", std::ios::binary);
//    data_fin.read(static_cast<char*>(x->data), 3 * 224 * 224 * 4);

    // get the function from the module(set input data)
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    set_input("images", x);

    // get the function from the module(load patameters)
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    // get the function from the module(run it)
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    run();

    DLTensor* y;
    int out_ndim = 3;
    int64_t out_shape[3] = {1,6000,85};
    TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

    // get the function from the module(get output data)
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    get_output(0, y);

    std::vector<BoxInfo> boxes;
    int num_classes = 80;
    float score_threshold = 0.1;
    float* out_data = static_cast<float *>(y->data);

    for(int i = 0; i < out_shape[1]; i++){
        float obj_conf = *(out_data + (i * (num_classes + 5)) + 4);

        float cls_conf = *(out_data + (i * (num_classes + 5))+5);
        int label = 0;
        for (unsigned int j = 0; j < num_classes; ++j)
        {
//            float tmp_conf = offset_obj_cls_ptr[j + 5];
            float tmp_conf = *(out_data + (i * (num_classes + 5))+j+5);
            if (tmp_conf > cls_conf)
            {
                cls_conf = tmp_conf;
                label = j;
            }
        } // argmax

        float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
        std::cout << conf << std::endl;
        if (conf < score_threshold) continue; //
        std::cout << "obj_conf " << obj_conf << " cls_conf " << cls_conf << " conf: " << conf << std::endl;

        float cx = *(out_data + (i * (num_classes + 5))+0);
//        float cy = offset_obj_cls_ptr[1];
        float cy = *(out_data + (i * (num_classes + 5))+1);
//        float w = output + (i * (num_classes + 5))[2];
        float w = *(out_data + (i * (num_classes + 5))+2);
//        float h = offset_obj_cls_ptr[3];
        float h = *(out_data + (i * (num_classes + 5))+3);
//        float x1 = (cx - w / 2.f) * w_scale;
//        float y1 = (cy - h / 2.f) * h_scale;
//        float x2 = (cx + w / 2.f) * w_scale;
//        float y2 = (cy + h / 2.f) * h_scale;
        float x1 = (cx - w / 2.f) / ratio;
        float y1 = (cy - h / 2.f) / ratio;
        float x2 = (cx + w / 2.f) / ratio;
        float y2 = (cy + h / 2.f) / ratio;

        BoxInfo box;
        box.x1 = std::max(0.f, x1);
        box.y1 = std::max(0.f, y1);
        box.x2 = std::min(x2, (float) image.cols - 1.f);
        box.y2 = std::min(y2, (float) image.rows - 1.f);
        box.score = conf;
        box.label = label;
        boxes.push_back(box);

    }


    nms(boxes, 0.2);
    draw_coco_bboxes(image, boxes);
    cv::waitKey(0);

    TVMArrayFree(x);
    TVMArrayFree(y);

    std::cout << "done!" << std::endl;
    return 0;
}
