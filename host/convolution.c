#include <stdlib.h>
#include <float.h>

#include "convolution.h"
#include <logging.h>

/**
 * Retrieves the weight value from the given layer at the specified indices.
 *
 * @param l     Pointer to the Layer structure.
 * @param c     Index of the channel.
 * @param m     Index of the row.
 * @param n     Index of the column.
 * @return      The weight value at the specified indices.
 *              Returns 0 if the layer is NULL or if the indices are invalid.
 */
double get_weight(Layer* l, int c, int m, int n){
    if(!l)  return 0;
    int invalid = (m < 0 || n < 0 || c < 0) || (m > l->m || n > l->n || c > l->c);
    if(invalid) return 0;
    return l->weights[(c * l->m * l->n) + (m * l->n) + n];
}

/**
 * Sets the weight value for a specific position in the layer.
 *
 * @param val The value to set.
 * @param l Pointer to the Layer structure.
 * @param c The channel index.
 * @param m The row index.
 * @param n The column index.
 */
void set_weight(double val, Layer* l, int c, int m, int n) {
    if(!l)  return;
    int invalid = (m < 0 || n < 0 || c < 0) || (m > l->m || n > l->n || c > l->c);
    if(invalid) return;
    l->weights[(c * l->m * l->n) + (m * l->n) + n] = val;
}

/**
 * @brief Creates a layer with specified dimensions and initializes the weights to zero.
 *
 * This function creates a layer with the specified dimensions (m, n, c) and initializes
 * the weights to zero. It returns a pointer to the created layer.
 *
 * @param m The number of rows in the layer.
 * @param n The number of columns in the layer.
 * @param c The number of channels in the layer.
 * @return A pointer to the created layer, or NULL if the dimensions are invalid.
 */

Layer* make_layer(int m, int n, int c){
    if( !m || !n || !c ) {
        return NULL;
    }

    Layer* layer = (Layer*) calloc(1, sizeof(*layer));
    *layer = (Layer) { .m=m, .n=n, .c=c, .weights=NULL };
    
    // creates a 3d matrix of zeros size (m, n, c)
    double* weights = (double*) calloc(m * n * c, sizeof(*weights));
    layer->weights = weights;
    for(int i=0; i < m; i++){
        for(int j = 0; j < n; j++){
            for(int k = 0; k < c; k++){
                set_weight(0, layer, k, i, j);
            }
        }
    }
    
    return layer;
}

/**
 * Calculates the dot product of a 3D input layer and a kernel layer.
 * 
 * @param input The input layer.
 * @param kernel The kernel layer.
 * @param offset_c The offset for the channel dimension.
 * @param offset_i The offset for the height dimension.
 * @param offset_j The offset for the width dimension.
 * @return The dot product of the input and kernel layers.
 */
static double dot_3d(Layer* input, Layer* kernel, int offset_c, int offset_i, int offset_j) {
    double product = 0;
    for (int i = 0; i < kernel->m; i++) {
        for(int j = 0; j < kernel->n; j++) {
            for(int k = 0; k < kernel->c; k++) {
                product += get_weight(input, k + offset_c, i + offset_i, j + offset_j) * get_weight(kernel, k, i, j);
            }
        }
    }
    return product;
}

/**
 * Calculates the maximum value within a specified window of a given layer.
 *
 * @param input The input layer.
 * @param window_size_m The size of the window in the m dimension.
 * @param window_size_n The size of the window in the n dimension.
 * @param c The channel index.
 * @param offset_1 The offset in the m dimension.
 * @param offset_2 The offset in the n dimension.
 * @return The maximum value within the specified window.
 */
static double get_max(Layer* input, int window_size_m, int window_size_n, int c, int offset_1, int offset_2) {
    double max = -DBL_MAX;
    double curr;
    for (int i = 0; i < window_size_m; i++) {
        for(int j = 0; j < window_size_n; j++) {
            curr = get_weight(input, c, i + offset_1, j + offset_2);
            if(max < curr) {
                max = curr;
            }
        }
    }
    return max;   
}


/**
 * Applies max pooling operation to the input layer and stores the result in the output layer.
 *
 * @param input The input layer.
 * @param window_size_m The size of the pooling window in the m dimension.
 * @param window_size_n The size of the pooling window in the n dimension.
 * @param stride The stride value for the pooling operation.
 * @param output The output layer.
 */

void make_max_pooling(Layer* input, int window_size_m, int window_size_n, int stride, Layer* output) {

    for(int i = 0; i < output->m; i++){
        for(int j = 0; j < output->n; j++) {
            for(int k = 0; k < output->c; k++) {
                double max = get_max(input, window_size_m, window_size_n, k, i * stride, j * stride);
                set_weight(max, output, k, i, j);
            }
        }
    }
}

/**
 * Applies the convolution operation to the input layer using the kernel layer and stores the result in the output layer.
 *
 * @param input The input layer.
 * @param kernel The kernel layer.
 * @param padding The padding value for the convolution operation.
 * @param output The output layer.
 */

void make_convolution(Layer* input, Layer* kernel, int padding, Layer* output){ 

    double dot;
    for(int i = 0; i < output->m; i++){
        for(int j = 0; j < output->n; j++) {
            for(int k = 0; k < output->c; k++) {
                dot = dot_3d(input, kernel, k, i - padding, j - padding);
                set_weight(dot, output, k, i, j);
            }
        }
    }
}

void make_fully_connected(Layer* input, Layer* w_and_b, Layer* output){
    for (int i = 0; i < w_and_b->n; i += 2){
        for(int j = 0; j < input->m * input->n * input->c; j++){
            output->weights[i/2] += input->weights[j] * get_weight(w_and_b, 0, j, i);
            output->weights[i/2] += get_weight(w_and_b, 0, j, i+1);
        }
    }
}

void destroy_layer(Layer *layer){
    /*
    Takes a layer and frees all the mem required for it
    */
    if( !layer ) return;
    
    free(layer->weights);
    free(layer);
}