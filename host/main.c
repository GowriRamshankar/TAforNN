#include <stdio.h>
#include <string.h>
#include <err.h>
#include <tee_client_api.h>
#include <convolution_ta.h>
#include <time.h>

#include "convolution.h"

//#define DEBUGGING
#include <logging.h>

#define MY_RAND_MIN -10
#define MY_RAND_MAX  10

#define CONV_POOL_CMD 1
#define FULL_CONN_CMD 2

#define TA_CONV_UUID \
	{ 0x7fc5c039, 0x0542, 0x4ee1, \
		{ 0x80, 0xaf, 0xb4, 0xea, 0xb2, 0xf1, 0x99, 0x8d} }

/************************************************************
*   global variable declaratoins
************************************************************/
TEEC_Context ctx;
TEEC_Session sess;
TEEC_SharedMemory layer_SM;
TEEC_SharedMemory weights_SM;

/************************************************************
*   function declarations
************************************************************/
static void assign_random(Layer* l);
void run_TA_func(Layer* input, Layer* params, Layer* output, int cmd_num);
void prepare_tee_session();
void terminate_tee_session();

/************************************************************
*   begin main
************************************************************/
// Define the number of rows to run at a time for each layer
#define MEM_FACTOR_1 32
#define MEM_FACTOR_2 50
int main(){  
    double *weights;
    Layer *input, *fc_w_and_b_1, *fc_1, *fc_w_and_b_2, *fc_2, *fc_w_and_b_3, *fc_3;
    struct timespec time_start, time_end;
    uint64_t delta_us;

    prepare_tee_session();
    // define the shape of the input layer
    input = make_layer(3, 3, 512);
    assign_random(input);

    DBG_LOG_BLUE("input layer:\n");
    DBG_PRINT_SHAPE(input);
    DBG_PRINT_LAYER(input, 0);
    DBG_PRINTF("\n");

    // running the first fully connected layer just one row at a time
    fc_w_and_b_1 = make_layer(3*3*512, 2 * MEM_FACTOR_1, 1);
    fc_1         = make_layer(1000, 1, 1);  
    weights      = fc_1->weights;
    delta_us = 0;
    for(int i = 0; i < 1000; i += MEM_FACTOR_2){
        assign_random(fc_w_and_b_1);   
        clock_gettime(CLOCK_MONOTONIC_RAW, &time_start);
        // run_TA_func(input, fc_w_and_b_1, fc_1, FULL_CONN_CMD);
        make_fully_connected(input, fc_w_and_b_1, fc_1);
        clock_gettime(CLOCK_MONOTONIC_RAW, &time_end);
        delta_us += (time_end.tv_sec - time_start.tv_sec) * 1000 + (time_end.tv_nsec - time_start.tv_nsec) / 1000000;
        
        fc_1->weights += MEM_FACTOR_1;                          // increments the weights pointer 
    }
    printf("L1: %lu ms\n", delta_us);

    fc_1->weights = weights;
    destroy_layer(input);
    destroy_layer(fc_w_and_b_1);

    DBG_LOG_BLUE("output from fully connected 1 (fc_1 layer):\n");
    DBG_PRINT_SHAPE(fc_1);
    DBG_PRINT_ARR_SML(fc_1->weights, fc_1->m);
    DBG_PRINTF("\n");

    // running the second fully connected layer just one row at a time
    fc_w_and_b_2 = make_layer(4096, 2 * MEM_FACTOR_1, 1);
    fc_2         = make_layer(4096, 1, 1);
    weights      = fc_2->weights;
    delta_us = 0;
    for(int i = 0; i < 4096; i += MEM_FACTOR_1){
        assign_random(fc_w_and_b_2);                            // TODO: this would really be where the row is read  
        clock_gettime(CLOCK_MONOTONIC_RAW, &time_start);     
        run_TA_func(fc_1, fc_w_and_b_2, fc_2, FULL_CONN_CMD);
        // make_fully_connected(fc_1, fc_w_and_b_2, fc_2);
        clock_gettime(CLOCK_MONOTONIC_RAW, &time_end);
        delta_us += (time_end.tv_sec - time_start.tv_sec) * 1000 + (time_end.tv_nsec - time_start.tv_nsec) / 1000000;
        fc_2->weights += MEM_FACTOR_1;                          // incraments the weights pointer 
    }
    
    printf("L2: %lu ms\n", delta_us);
    fc_2->weights = weights;
    destroy_layer(fc_1);
    destroy_layer(fc_w_and_b_2);

    DBG_LOG_BLUE("output from fully connected 2 (fc_2 layer):\n");
    DBG_PRINT_SHAPE(fc_2);
    DBG_PRINT_ARR_SML(fc_2->weights, fc_2->m);
    DBG_PRINTF("\n");

    // running the third fully connected layer just one row at a time
    fc_w_and_b_3 = make_layer(4096, 2 * MEM_FACTOR_2, 1);
    fc_3         = make_layer(1000, 1, 1);
    weights      = fc_3->weights;
    delta_us = 0;
    for(int i = 0; i < 1000; i += MEM_FACTOR_2){
        assign_random(fc_w_and_b_3);                            // TODO: this would really be where the row is read
        clock_gettime(CLOCK_MONOTONIC_RAW, &time_start);       
        run_TA_func(fc_2, fc_w_and_b_3, fc_3, FULL_CONN_CMD);
        // make_fully_connected(fc_2, fc_w_and_b_3, fc_3);
        clock_gettime(CLOCK_MONOTONIC_RAW, &time_end);
        delta_us += (time_end.tv_sec - time_start.tv_sec) * 1000 + (time_end.tv_nsec - time_start.tv_nsec) / 1000000;
        fc_3->weights += MEM_FACTOR_2;                          // incraments the weights pointer 
    }

    printf("L3: %lu ms\n", delta_us);
    fc_3->weights = weights;
    destroy_layer(fc_2);
    destroy_layer(fc_w_and_b_3);


    DBG_LOG_BLUE("output from fully connected 3 (fc_3 layer):\n");
    DBG_PRINT_SHAPE(fc_3);
    DBG_PRINT_ARR_SML(fc_3->weights, fc_3->m);
    DBG_PRINTF("\n");

    destroy_layer(fc_3); 

    terminate_tee_session();

    return 0;
}

/************************************************************
*   begin other function definitons 
************************************************************/
/**
 * Assigns dummy weights to the given layer.
 *
 * @param l The layer to assign  dummy weights to.
 */
static void assign_random(Layer* l) {
    for (int i = 0; i < l->m; i++){
        for(int j = 0; j < l->n; j++){
            for (int k = 0; k < l->c; k++){
                int num = rand() % (MY_RAND_MAX - MY_RAND_MIN + 1) + MY_RAND_MIN;
                set_weight(num, l, k, i, j);
            }
        }
    }
}

/**
 * @brief Runs a Trusted Application (TA) function with the given input, parameters, and output.
 *
 * This function prepares the TEEC_Operation structure and invokes the specified TA command.
 * It sets the parameter types and values required by the TA function.
 *
 * @param input   Pointer to the input Layer structure.
 * @param params  Pointer to the parameters Layer structure.
 * @param output  Pointer to the output Layer structure.
 * @param cmd_num The command number to be invoked in the TA.
 */
void run_TA_func(Layer* input, Layer* params, Layer* output, int cmd_num)
{
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

        memset(&op, 0, sizeof(op));
        op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                                                         TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT);

        Layer layers[] = {*input, *params, *output};

        op.params[0].tmpref.buffer = layers;
        op.params[0].tmpref.size   = sizeof(layers);
        op.params[1].tmpref.buffer = output->weights;
        op.params[1].tmpref.size   = output->c * output->m * output->n * sizeof(*output->weights);
        op.params[2].tmpref.buffer = params->weights;
        op.params[2].tmpref.size   = params->c * params->m * params->n * sizeof(*params->weights);
        op.params[3].tmpref.buffer = input->weights;
        op.params[3].tmpref.size   = input->c * input->m * input->n * sizeof(*input->weights);

        res = TEEC_InvokeCommand(&sess, cmd_num, &op, &origin);

        if (res != TEEC_SUCCESS) {
                errx(1, "TEEC_InvokeCommand(CONV) failed 0x%x origin 0x%x", res, origin);
        }

}


/**
 * @brief Initializes a TEE session and opens a session with the Trusted Application (TA).
 * 
 * This function initializes a context connecting the host application to the Trusted Execution Environment (TEE).
 * It then opens a session with the Trusted Application identified by the UUID TA_CONV_UUID.
 * 
 * @return None.
 */
void prepare_tee_session()
{
    TEEC_UUID uuid = TA_CONV_UUID;
    uint32_t origin;
    TEEC_Result res;

    /* Initialize a context connecting us to the TEE */
    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InitializeContext failed with code 0x%x", res);

    /* Open a session with the TA */
    res = TEEC_OpenSession(&ctx, &sess, &uuid,
                           TEEC_LOGIN_PUBLIC, NULL, NULL, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_Opensession failed with code 0x%x origin 0x%x",
         res, origin);
}

void terminate_tee_session()
{
    TEEC_CloseSession(&sess);
    TEEC_FinalizeContext(&ctx);
}