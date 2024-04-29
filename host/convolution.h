#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

typedef struct _Layer {
 int m, n, c;
 double* weights;
 
} Layer;

Layer* make_layer(int, int, int);
void make_convolution(Layer* input, Layer* kernel, int padding, Layer* output);
void make_max_pooling(Layer* input, int window_size_m, int window_size_n, int stride, Layer* output);
void make_fully_connected(Layer* input, Layer* w_and_b, Layer* output);
void destroy_layer(Layer*);
void set_weight(double val, Layer* l, int m, int n, int c);
double get_weight(Layer* l, int m, int n, int c);

#endif