#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct {
    int rows;
    int columns;
    float **data;
} matrix;


matrix *mat_alloc(int row, int col);
void mat_free(matrix *M);

// Initialization
float RNG(float a , float b);
void mat_init(matrix *A);
void mat_fill(matrix *A, float x);
void mat_rand(matrix *A);
void mat_copy(matrix *dest, matrix *source);

// Math
void mat_a_s(matrix *A, matrix *B, matrix *mat, char sign);
void mat_mul(matrix *A, matrix *B, matrix *mat);
void mat_mul_hadamard(matrix *A, matrix *B, matrix *mat);
void mat_scale(matrix *A, float a, matrix *mat);
void mat_T(matrix *A, matrix *mat);
void mat_size(matrix *A);

void mat_row(matrix *A, int row, matrix *mat);
void mat_column(matrix *A, int column, matrix *mat);
float mat_magnitude(matrix *A);

// Activation Functions
float sigmoid(float x);
void mat_sig(matrix *A, matrix *mat);
void mat_sig_deriv(matrix *A, matrix *mat);
void mat_relu(matrix *A, matrix *mat);
void mat_relu_deriv(matrix *A , matrix *mat);
void mat_softmax(matrix *A, matrix *mat);

void mat_print(matrix *A);
void y_x2_generator(float *input,float *output, int n);
