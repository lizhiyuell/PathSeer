#ifndef _PRINT_HPP_
#define _PRINT_HPP_

#include<stdio.h>
#include<cstdint>
#include<algorithm>

void print_label(int64_t* array, float* dist_arr, int length, int limit=-1){

    int stop = length;
    if(limit >= 0){
        stop = std::min<int>(stop, limit);
    }

    for(int i=0; i< stop; i++){
        printf("%d: dist=%f, idx=%lu\n", i, dist_arr[i], array[i]);
    }

}

void print_recall(double recall, int idx=-1){
    if(idx!=-1)
        printf("Idx %d, Recall is %f\n", idx, recall);
    else
        printf("Recall is %f\n", recall);
}

void print_arr_class_int(int* class_arr, int idx){
    printf("Class_int of %d is %d\n", idx, class_arr[idx]);
}

void print_class_int(int class_int){
    printf("Class_int is %d\n", class_int);
}

#endif
