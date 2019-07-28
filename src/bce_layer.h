#ifndef BCE_LAYER_H
#define BCE_LAYER_H
#include "layer.h"
#include "network.h"

typedef layer bce_layer;

#ifdef __cplusplus
extern "C" {
#endif
// void softmax_array(float *input, int n, float temp, float *output);
bce_layer make_bce_layer(int batch, int inputs, int groups , float* class_weights );
void forward_bce_layer(const bce_layer l, network_state state);
void backward_bce_layer(const bce_layer l, network_state state);

#ifdef GPU
void pull_bce_layer_output(const bce_layer l);
void forward_bce_layer_gpu(const bce_layer l, network_state state);
void backward_bce_layer_gpu(const bce_layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
#endif
