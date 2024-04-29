#ifndef __LOGGING_H__
    #define __LOGGING_H__
    #include <stdio.h>
    #include <stdlib.h>
    #include <stdbool.h>
    #include <unistd.h>
    
    //_______________________________FUNCTIONAL_TOOLS__________________________
    #define MAX(a,b) ((a) > (b) ? a : b)

    //_______________________________LOGGING_TOOLS__________________________
    #define ANSI_RED   "\x1b[31m"
    #define ANSI_GREEN "\x1b[32m"
    #define ANSI_BLUE  "\x1b[34m"
    #define ANSI_RESET "\x1b[0m"
    #define LOG_OUT stderr
    #define safe_write(sf_ptr, sf_sz, sf_num, sf_fp)           \
        do {                                                   \
            if(sf_fp)                                          \
                fwrite((sf_ptr), (sf_sz), (sf_num), (sf_fp));  \
        } while(false)  

    #define LOG_IS_FILE_OUT 1 
    #define _LOG_OUT(...) printf(__VA_ARGS__)
    #define _LOG_COL(log_col, ...)                     \
            do{                                        \
                if (LOG_IS_FILE_OUT) {                 \
                    _LOG_OUT(log_col __VA_ARGS__);     \
                    _LOG_OUT(ANSI_RESET);	           \
                }                                      \
                else{                                  \
                    _LOG_OUT(__VA_ARGS__);             \
                }                                      \
            }while(false)
    #define LOG_ERR(...)   _LOG_COL(ANSI_RED, __VA_ARGS__)
    #define LOG_RED(...)   _LOG_COL(ANSI_RED, __VA_ARGS__)
    #define LOG_GREEN(...) _LOG_COL(ANSI_GREEN, __VA_ARGS__)
    #define LOG_BLUE(...)  _LOG_COL(ANSI_BLUE, __VA_ARGS__)

    //_______________________________DEBUGGING__________________________
    #ifdef DEBUGGING
        #define DBG_PRINTF(...) printf(__VA_ARGS__) 
        #define DBG_LOG_ERR(...)   _LOG_COL(ANSI_RED, __VA_ARGS__)
        #define DBG_LOG_RED(...)   _LOG_COL(ANSI_RED, __VA_ARGS__)
        #define DBG_LOG_GREEN(...) _LOG_COL(ANSI_GREEN, __VA_ARGS__)
        #define DBG_LOG_BLUE(...)  _LOG_COL(ANSI_BLUE, __VA_ARGS__)
        #define DBG_PRINT_SHAPE(dbg_lr) printf("%s shape: (%d, %d, %d)\n", (#dbg_lr), (dbg_lr)->m, (dbg_lr)->n, (dbg_lr)->c)
        #define DBG_PRINT_ARR(dbg_arr, dbg_sz)                         \
        do {                                                           \
            DBG_PRINTF("[");                                           \
            for(int dbg_i = 0; dbg_i < (dbg_sz) - 1; dbg_i++){         \
                DBG_PRINTF("%5d, ", (int)(dbg_arr)[dbg_i]);            \
            }                                                          \
            DBG_PRINTF("%5d]\n", (int)(dbg_arr)[(dbg_sz) - 1]);        \
        } while(false)

        #define DBG_PRINT_ARR_SML(dbg_arr, dbg_sz)                              \
        do {                                                                    \
            DBG_PRINTF("[");                                                    \
            for(int dbg_i = 0; dbg_i < 3; dbg_i++){                             \
                DBG_PRINTF("%5d, ", (int)(dbg_arr)[dbg_i]);                     \
            }                                                                   \
            DBG_PRINTF("  ...  ");                                              \
            for(int dbg_i = (dbg_sz) - 3; dbg_i < (dbg_sz) - 1; dbg_i++){       \
                DBG_PRINTF("%5d, ", (int)(dbg_arr)[dbg_i]);                     \
            }                                                                   \
            DBG_PRINTF("%5d]\n", (int)(dbg_arr)[(dbg_sz) - 1]);                 \
        } while(false)                                 

        #define DBG_PRINT_LAYER(dbg_grid, dbg_c)                                                  \
        do {                                                                                      \
            for(int dbg_j = 0; dbg_j < (dbg_grid)->m; dbg_j++){                                   \
                DBG_PRINTF("[");                                                                  \
                for(int dbg_i = 0; dbg_i < (dbg_grid)->n - 1; dbg_i++){                           \
                    DBG_PRINTF("%5d, ", (int)get_weight(dbg_grid, dbg_c, dbg_j, dbg_i));          \
                }                                                                                 \
                DBG_PRINTF("%5d]\n", (int)get_weight(dbg_grid, dbg_c, dbg_j, (dbg_grid)->n - 1)); \
            }                                                                                     \
        } while(false)

        #define DBG_PRINT_LAYER_SML(dbg_grid, dbg_c)                                               \
        do {                                                                                       \
            for(int dbg_j = 0; dbg_j < 3; dbg_j++){                                                \
                DBG_PRINTF("[");                                                                   \
                for(int dbg_i = 0; dbg_i < 3; dbg_i++){                                            \
                    DBG_PRINTF("%5d, ", (int)get_weight(dbg_grid, dbg_c, dbg_j, dbg_i));           \
                }                                                                                  \
                DBG_PRINTF("  ...  ");                                                             \
                for(int dbg_i = (dbg_grid)->n - 3; dbg_i < (dbg_grid)->n - 1; dbg_i++){            \
                    DBG_PRINTF("%5d, ", (int)get_weight(dbg_grid, dbg_c, dbg_j, dbg_i));           \
                }                                                                                  \
                DBG_PRINTF("%5d]\n", (int)get_weight(dbg_grid, dbg_c, dbg_j, (dbg_grid)->n - 1));  \
            }                                                                                      \
            DBG_PRINTF("           ...............................           \n");                 \
            for(int dbg_j = (dbg_grid)->m - 3; dbg_j < (dbg_grid)->m; dbg_j++){                    \
                DBG_PRINTF("[");                                                                   \
                for(int dbg_i = 0; dbg_i < 3; dbg_i++){                                            \
                    DBG_PRINTF("%5d, ", (int)get_weight(dbg_grid, dbg_c, dbg_j, dbg_i));           \
                }                                                                                  \
                DBG_PRINTF("  ...  ");                                                             \
                for(int dbg_i = (dbg_grid)->n - 3; dbg_i < (dbg_grid)->n - 1; dbg_i++){            \
                    DBG_PRINTF("%5d, ", (int)get_weight(dbg_grid, dbg_c, dbg_j, dbg_i));           \
                }                                                                                  \
                DBG_PRINTF("%5d]\n", (int)get_weight(dbg_grid, dbg_c, dbg_j, (dbg_grid)->n - 1));  \
            }                                                                                      \
        } while(false)
    #else
        #define DBG_PRINTF(...)
        #define DBG_PRINT_SHAPE(dbg_lr)
        #define DBG_PRINT_ARR(dbg_arr, dbg_sz)
        #define DBG_PRINT_ARR_SML(dbg_arr, dbg_sz)
        #define DBG_PRINT_LAYER(dbg_grid, dbg_c)
        #define DBG_PRINT_LAYER_SML(dbg_grid, dbg_c)
        #define DBG_LOG_ERR(...)   
        #define DBG_LOG_RED(...)   
        #define DBG_LOG_GREEN(...) 
        #define DBG_LOG_BLUE(...)  
    #endif
#endif