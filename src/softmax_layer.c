#include "softmax_layer.h"
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

void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output)
{
	int b;
	for (b = 0; b < batch; ++b) {
		int i;
		int count = 0;
		for (i = 0; i < hierarchy->groups; ++i) {
			int group_size = hierarchy->group_size[i];
			softmax(input + b*inputs + count, group_size, temp, output + b*inputs + count, 1);
			count += group_size;
		}
	}
}

softmax_layer make_softmax_layer(int batch, int inputs, int groups , float* class_weights )
{
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n",  inputs);
    softmax_layer l = { (LAYER_TYPE)0 };
    l.type = SOFTMAX;
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

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;
    #ifdef GPU
    l.forward_gpu = forward_softmax_layer_gpu;
    l.backward_gpu = backward_softmax_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
    l.class_weights_gpu = cuda_make_array(l.class_weights ,inputs*batch);

    #endif
    return l;
}
//+20190716 class_weights supportted on cpu
void forward_softmax_layer(const softmax_layer l, network_state net)
{
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        }
    } else {
        softmax_cpu(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
    }

    if(net.truth && !l.noloss){
        // softmax_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        softmax_x_ent_cpu_clsw(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss, l.class_weights);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}
//将softmax的残差 传递给前一层网络
void backward_softmax_layer(const softmax_layer l, network_state net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_softmax_layer_output(const softmax_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
    // cuda_pull_array(layer.class_weights_gpu, layer.class_weights, layer.inputs);
}

void forward_softmax_layer_gpu(const softmax_layer l, network_state net)
{
    if(l.softmax_tree){
		softmax_tree_gpu(net.input, 1, l.batch, l.inputs, l.temperature, l.output_gpu, *l.softmax_tree);
		/*
		int i;
		int count = 0;
		for (i = 0; i < l.softmax_tree->groups; ++i) {
		int group_size = l.softmax_tree->group_size[i];
		softmax_gpu(net.input_gpu + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output_gpu + count);
		count += group_size;
		}
		*/
    } else {
        if(l.spatial){
			softmax_gpu_new_api(net.input, l.c, l.batch*l.c, l.inputs/l.c, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu);
        }else{
			softmax_gpu_new_api(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output_gpu);
        }
    }
    if(net.truth && !l.noloss){
        // softmax_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth, l.delta_gpu, l.loss_gpu);
        softmax_x_ent_gpu_clsw(l.batch*l.inputs, l.output_gpu, net.truth, l.delta_gpu, l.loss_gpu, l.class_weights_gpu);
        if(l.softmax_tree){
			mask_gpu_new_api(l.batch*l.inputs, l.delta_gpu, SECRET_NUM, net.truth, 0);
			mask_gpu_new_api(l.batch*l.inputs, l.loss_gpu, SECRET_NUM, net.truth, 0);
        }
        cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_softmax_layer_gpu(const softmax_layer layer, network_state net)
{
	axpy_ongpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, net.delta, 1);
}

#endif
