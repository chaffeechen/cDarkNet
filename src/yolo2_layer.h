#ifndef YOLO2_LAYER_H
#define YOLO2_LAYER_H

//#include "darknet.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_yolo2_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes);
layer make_yolo2_layer2(int batch, int w, int h, int n, int total, int *mask, float *class_weights , int classes, int max_boxes);
void forward_yolo2_layer(const layer l, network_state state);
void backward_yolo2_layer(const layer l, network_state state);
void resize_yolo2_layer(layer *l, int w, int h);
int yolo2_num_detections(layer l, float thresh);
int get_yolo2_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter);
int get_yolo2_detections_double_thresh(layer l, int w, int h, int netw, int neth, float thresh, float iou_thresh ,int *map, int relative, detection *dets, int letter);
void correct_yolo2_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter);

#ifdef GPU
void forward_yolo2_layer_gpu(const layer l, network_state state);
void backward_yolo2_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
#endif
