#include "bce_layer.h"
#include "blas.h"
#include "dark_cuda.h"
#include "utils.h"
#include "blas.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define SECRET_NUM -1234

bce_layer make_bce_layer(int batch, int inputs, int groups , float* class_weights )
{
    assert(inputs%groups == 0);
    fprintf(stderr, "bce                                       %4d\n",  inputs);
    bce_layer l = { (LAYER_TYPE)0 };
    l.type = BCE;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = (float*)calloc(inputs * batch, sizeof(float));
    l.output = (float*)calloc(inputs * batch, sizeof(float));
    l.delta = (float*)calloc(inputs * batch, sizeof(float));
    l.cost = (float*)calloc(1, sizeof(float));

    float* cls_weights = (float*)calloc(inputs, sizeof(float));
    if(!class_weights){
        for ( int i = 0 ; i < inputs ; i++ ) 
            cls_weights[i] = 1.0;
        // free(class_weights);
    } else {
        memcpy(cls_weights , class_weights , inputs*sizeof(float));
    }

    float* clsw = (float*)calloc(inputs*batch,sizeof(float));
    for(int i = 0 ; i < batch ; i++ ) 
        memcpy(clsw+i*inputs , cls_weights , inputs*sizeof(float));
    l.class_weights = clsw;
    if(cls_weights) free(cls_weights);//+20190723 bugfix

    l.forward = forward_bce_layer;
    l.backward = backward_bce_layer;
    #ifdef GPU
    l.forward_gpu = forward_bce_layer_gpu;
    l.backward_gpu = backward_bce_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
    l.class_weights_gpu = cuda_make_array(l.class_weights ,inputs*batch);

    #endif
    return l;
}
//+20190716 class_weights supportted on cpu
void forward_bce_layer(const bce_layer l, network_state net)
{
    logistic_cpu(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);

    if(net.truth && !l.noloss){
        // softmax_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        logistic_x_ent_cpu_clsw(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss, l.class_weights);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}
//将softmax的残差 传递给前一层网络
void backward_bce_layer(const bce_layer l, network_state net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_bce_layer_output(const bce_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
    // cuda_pull_array(layer.class_weights_gpu, layer.class_weights, layer.inputs);
}

void forward_bce_layer_gpu(const bce_layer l, network_state net)
{
	logistic_gpu_new_api(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output_gpu);

    if(net.truth && !l.noloss){
        // softmax_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth, l.delta_gpu, l.loss_gpu);
        logistic_x_ent_gpu_clsw(l.batch*l.inputs, l.output_gpu, net.truth, l.delta_gpu, l.loss_gpu, l.class_weights_gpu);
        cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_bce_layer_gpu(const bce_layer layer, network_state net)
{
	axpy_ongpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, net.delta, 1);
}

#endif
