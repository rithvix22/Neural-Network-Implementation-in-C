#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

float RNG(float a , float b) {
   	
	float x = (float)rand() / (float)RAND_MAX;
	return ((b-a)*x)+a;
}	

matrix *mat_alloc(int row, int col) {
   
    matrix *out = (matrix *)malloc(sizeof(matrix));
    out->rows = row;
    out->columns = col;
    out->data = (float **)malloc(sizeof(float *) * row);
    for (int i = 0; i < row; i++) {
        out->data[i] = (float *)malloc(sizeof(float) * col);
    }
    return out;
}

void mat_a_s(matrix *A, matrix *B, matrix *mat, char sign) {
    
    if ((A->rows != B->rows) || (A->columns != B->columns)) {
        printf("invalid addition/subtraction\n");
        return;
    }
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->columns; j++) {
            if (sign == '+') {
                mat->data[i][j] = A->data[i][j] + B->data[i][j];
            } else {
                mat->data[i][j] = A->data[i][j] - B->data[i][j];
            }
        }
    }
}

void mat_row(matrix *A, int row, matrix *mat) {
    
    
    for (int i = 0; i < A->columns; i++) {
        mat->data[0][i] = A->data[row][i];
    }
}

void mat_column(matrix *A, int column, matrix *mat) {
    for (int i = 0; i < A->rows; i++) {
        mat->data[i][0] = A->data[i][column];
    }
}

void mat_copy(matrix *dest, matrix *source) {
    
    for (int i = 0; i < source->rows; i++) {
        for (int j = 0; j < source->columns; j++) {
            dest->data[i][j] = source->data[i][j];
        }
    }
}

void mat_init(matrix *A) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->columns; j++) {
            scanf("%f", &A->data[i][j]);
        }
    }
}


void mat_size(matrix *A){
	printf("%d x %d\n",A->rows , A->columns);
}

void mat_mul(matrix *A, matrix *B, matrix *mat) {
    if (A->columns != B->rows) {
        printf("invalid multiplication\n");
        return;
    }
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->columns; j++) {
            float sum = 0;
            for (int k = 0; k < A->columns; k++) {
                sum += A->data[i][k] * B->data[k][j];
            }
            mat->data[i][j] = sum;
        }
    }
}

void mat_mul_hadamard(matrix *A, matrix *B, matrix *mat) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->columns; j++) {
            mat->data[i][j] = A->data[i][j] * B->data[i][j];
        }
    }
}

void mat_scale(matrix *A, float a, matrix *mat) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->columns; j++) {
            mat->data[i][j] = A->data[i][j] * a;
        }
    }
}

void mat_print(matrix *A) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->columns; j++) {
            printf("%.2f ", A->data[i][j]);
        }
        printf("\n");
    }
}

void mat_fill(matrix *A, float x) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->columns; j++) {
            A->data[i][j] = x;
        }
    }
}

void mat_T(matrix *A, matrix *mat) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->columns; j++) {
            mat->data[j][i] = A->data[i][j];
        }
    }
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void mat_sig(matrix *A, matrix *mat) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->columns; j++) {
            mat->data[i][j] = sigmoid(A->data[i][j]);
        }
    }
}

// FIXED: A is already sigmoid(Z), so derivative is A*(1-A), no sigmoid call needed.
void mat_sig_deriv(matrix *A, matrix *mat) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->columns; j++) {
            mat->data[i][j] = A->data[i][j] * (1.0f - A->data[i][j]);
        }
    }
}

void mat_relu(matrix *A, matrix *mat) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->columns; j++) {
            	if(A->data[i][j] > 0){
			
			mat->data[i][j] = A->data[i][j];
		}
		else {
			
			mat->data[i][j] = 0;
		}
        }
    }
}

void mat_relu_deriv(matrix *A , matrix *mat){
	
	for(int i =0;i<A->rows;i++){
		for(int j =0 ; j<A->columns ; j++){
			
			if(A->data[i][j] >0){
				mat->data[i][j]= 1;
			}
			else{
				mat->data[i][j] = 0;
			}
		}
	}
}

void mat_softmax(matrix *A, matrix *mat) {
    float sum = 0.0f;
    for (int i = 0; i < A->rows; i++) {
        float e = expf(A->data[i][0]);
        mat->data[i][0] = e;
        sum += e;
    }
    for (int i = 0; i < A->rows; i++) {
        mat->data[i][0] /= sum;
    }
}

void mat_rand(matrix *A) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->columns; j++) {
            A->data[i][j] = RNG(-1.0f,1.0f);
        }
    }
}

float mat_magnitude(matrix *A) {
  
    

    float sum = 0;
    if (A->columns == 1) {
        for (int i = 0; i < A->rows; i++) {
            sum += A->data[i][0] * A->data[i][0];
        }
	return sum;
    }
    else if(A->rows == 1){
        
	 for (int i = 0; i < A->columns; i++) {
           
		 sum += A->data[0][i] * A->data[0][i];
        }
    	return sum;	
    }
    else {
	
	printf("Invalid matrix\n");
	return 0;
    }
    
}

void y_x2_generator(float *input, float *output, int n) {
	
	for(int i=0;i<n;i++){
		
		input[i] = RNG(0,1.0f);
		output[i] = pow(input[i],2);
	}
}

void mat_free(matrix *M) {
    if (M == NULL) return;
    
    for (int i = 0; i < M->rows; i++) {
        free(M->data[i]);
    }
    free(M->data);
    free(M);
}
