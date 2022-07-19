#include "matrix.h"

#include <immintrin.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <thread>

namespace chrono = std::chrono;

void transpose_matrix(float* m, int n){
    int k = (n * (n-1)) / 2;
    int start = 1;
    int plus = 2;
    int j = 0;
    int line = 0;
    float t = 0;
    for (int i = 0; i< k; i++){
        t = m[start + i];
        m[start + i] = m[(j+1)*n + line];
        m[(j+1)*n + line] = t;
        j++;
        if((start + i)% n == (n-1)){
            start = start + plus;
            plus++;
            line++;
            j = line;
        }
    }
}

void final_multi_thread_task(float *a, float *b, float *c, int i, int n){
    __m128 temp = _mm_set1_ps(0);
    __m128 am = _mm_set1_ps(0);
    __m128 bm = _mm_set1_ps(0);
    for (int j = 0; j < n; j++) {
            temp = _mm_set1_ps(0);
            for (int k = 0; k < n; k= k + 4) {
                am = _mm_load_ps(&a[i * n + k]);
                bm = _mm_load_ps(&b[j * n + k]);
                temp = _mm_add_ps(temp, _mm_mul_ps(am, bm));
            }
            c[i * n + j] = temp[0] + temp[1] + temp[2] + temp[3];
        }
}

void naive_matrix_multiply(float *a, float *b, float *c, int n) {
    transpose_matrix(b, n);
    for (int i = 0; i < n; i= i + 4) {
        std::thread th[4];
        for(int t= i; t<i+4; t++){
            th[t-i]= std::thread(final_multi_thread_task, a, b, c, t, n);
        }
        for(int t= 0; t<4; t++){
            th[t].join();
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

void matrix_multiply(float *a, float *b, float *c, int n) {
    naive_matrix_multiply(a, b, c, n);
}

#ifdef __cplusplus
}
#endif
