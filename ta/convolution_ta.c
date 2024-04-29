#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <convolution_ta.h>

// #define DEBUGGING
#include <logging.h>

/************************************************************
*   beginning of prototypes
************************************************************/
TEE_Result make_conv5_pool_TEE(uint32_t param_types, TEE_Param params[4]);
TEE_Result make_fully_connected_TEE(uint32_t param_types, TEE_Param params[4]);

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

/************************************************************
*   beginning of convolution functions
************************************************************/
double get_weight(Layer* l, int c, int m, int n){
    if(!l)  return 0;
    int invalid = (m < 0 || n < 0 || c < 0) || (m > l->m || n > l->n || c > l->c);
    if(invalid) return 0;
    return l->weights[(c * l->m * l->n) + (m * l->n) + n];
}

void set_weight(double val, Layer* l, int c, int m, int n) {
    if(!l)  return;
    int invalid = (m < 0 || n < 0 || c < 0) || (m > l->m || n > l->n || c > l->c);
    if(invalid) return;
    l->weights[(c * l->m * l->n) + (m * l->n) + n] = val;
}

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

static double dot_3d(Layer* input, Layer* kernel, int offest_c, int offset_i, int offset_j) {
    double product = 0;
    for (int i = 0; i < kernel->m; i++) {
        for(int j = 0; j < kernel->n; j++) {
            for(int k = 0; k < kernel->c; k++) {
                product += get_weight(input, k + offest_c, i + offset_i, j + offset_j) * get_weight(kernel, k, i, j);
            }
        }
    }
    return product;
}

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

void make_max_pooling(Layer* input, int window_size_m, int window_size_n, int stride, Layer* output) {
    /*
    Takes an input layer, window shape, and stride. 
    Max pooling is then performed to create the final_out layer.
    NOTE: this assumes the window shape and stride will match the input 
          (this is NOT memory safe if it does not!!!!)
    */
    for(int i = 0; i < output->m; i++){
        for(int j = 0; j < output->n; j++) {
            for(int k = 0; k < output->c; k++) {
                double max = get_max(input, window_size_m, window_size_n, k, i * stride, j * stride);
                set_weight(max, output, k, i, j);
            }
        }
    }
}

void make_convolution(Layer* input, Layer* kernel, int padding, Layer* output){ 
    /*
    takes an input layer and a kernel and then convolutes it 
    into a output layer. 
    NOTE: this assumes the kernal to only have one channel
    */
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
/************************************************************
*   end of convolution functions
************************************************************/

TEE_Result TA_CreateEntryPoint(void)
{
    DMSG("has been called");

    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void)
{
    DMSG("has been called");
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
                                    TEE_Param __maybe_unused params[4],
                                    void __maybe_unused **sess_ctx)
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    /* Unused parameters */
    (void)&params;
    (void)&sess_ctx;

    IMSG("secure world opened!\n");
    return TEE_SUCCESS;
}


void TA_CloseSessionEntryPoint(void __maybe_unused *sess_ctx)
{
    (void)&sess_ctx; /* Unused parameter */
    IMSG("Goodbye!\n");
}

TEE_Result make_conv5_pool_TEE(uint32_t param_types, TEE_Param params[4]){ 
    /*
    takes an input layer and a kernel and then convolutes it 
    into a output layer. 
    */
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                                TEE_PARAM_TYPE_MEMREF_INPUT,
                                                TEE_PARAM_TYPE_MEMREF_INPUT,
                                                TEE_PARAM_TYPE_MEMREF_INPUT);

    DBG_LOG_BLUE("make_conv5_pool_TEE has been called\n");
    if (param_types != exp_param_types) {
        LOG_RED("bad params recieved\n");
        return TEE_ERROR_BAD_PARAMETERS;
    }

    DBG_LOG_BLUE("Setting Parameters to vars\n");
    Layer* layers      = params[0].memref.buffer;
    Layer* input       = &layers[0];
    Layer* kernel      = &layers[1];
    Layer* final_out   = &layers[2];

    final_out->weights = params[1].memref.buffer;
    kernel->weights    = params[2].memref.buffer;
    input->weights     = params[3].memref.buffer;
    
    DBG_LOG_BLUE("input and kernel layers layer:\n");
    DBG_PRINT_SHAPE(input);
    DBG_PRINT_LAYER(input, 0);
    DBG_PRINT_SHAPE(kernel);
    DBG_PRINT_LAYER(kernel, 0);
    DBG_PRINTF("\n");

    int padding = 1;
    Layer* conv_5 = make_layer(input->m - kernel->m + 1 + 2 * padding, 
                                input->n - kernel->n + 1 + 2 * padding,
                                input->c - kernel->c + 1);
    make_convolution(input, kernel, padding, conv_5);
    
    DBG_LOG_BLUE("conv_5 from convolution:\n");
    DBG_PRINT_SHAPE(conv_5);
    DBG_PRINT_LAYER(conv_5, 0);
    DBG_PRINTF("\n");

    make_max_pooling(conv_5, 3, 3, 2, final_out); 
    
    DBG_LOG_BLUE("output from Max Pooling:\n");
    DBG_PRINT_SHAPE(final_out);
    DBG_PRINT_LAYER(final_out, 0);
    DBG_PRINTF("\n");

    destroy_layer(conv_5);

    DBG_LOG_BLUE("returning from make_conv5_pool_TEE\n");
    return TEE_SUCCESS;
}

TEE_Result make_fully_connected_TEE(uint32_t param_types, TEE_Param params[4]){ 
    /*
    takes an input layer and a kernel and then convolutes it 
    into a output layer. 
    */
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                                TEE_PARAM_TYPE_MEMREF_INPUT,
                                                TEE_PARAM_TYPE_MEMREF_INPUT,
                                                TEE_PARAM_TYPE_MEMREF_INPUT);

    DBG_LOG_BLUE("make_fully_connected_TEE has been called\n");
    if (param_types != exp_param_types) {
        LOG_RED("bad params recieved\n");
        return TEE_ERROR_BAD_PARAMETERS;
    }

    DBG_LOG_BLUE("Setting Parameters to vars\n");
    Layer* layers      = params[0].memref.buffer;
    Layer* input       = &layers[0];
    Layer* w_and_b      = &layers[1];
    Layer* final_out   = &layers[2];

    final_out->weights = params[1].memref.buffer;
    w_and_b->weights    = params[2].memref.buffer;
    input->weights     = params[3].memref.buffer;
    
    DBG_LOG_BLUE("input and w_and_b layers layer:\n");
    DBG_PRINT_SHAPE(input);
    // DBG_PRINT_LAYER(input, 0);
    DBG_PRINT_SHAPE(w_and_b);
    // DBG_PRINT_LAYER_SML(w_and_b, 0);
    DBG_PRINTF("\n");

    make_fully_connected(input, w_and_b, final_out);

    DBG_LOG_BLUE("output from fully connected layer:\n");
    DBG_PRINT_SHAPE(final_out);
    DBG_PRINT_ARR_SML(final_out->weights, final_out->m);
    DBG_PRINTF("\n");

    return TEE_SUCCESS;
}

TEE_Result TA_InvokeCommandEntryPoint(void __maybe_unused *sess_ctx,
                                      uint32_t cmd_id,
                                      uint32_t param_types, TEE_Param params[4])
{
    (void)&sess_ctx; /* Unused parameter */

    switch (cmd_id) {
        case CONV_POOL_CMD:
        return make_conv5_pool_TEE(param_types, params);
        case FULL_CONN_CMD:
        return make_fully_connected_TEE(param_types, params);

        default:
        return TEE_ERROR_BAD_PARAMETERS;

    }
}