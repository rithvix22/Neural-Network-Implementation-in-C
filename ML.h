#include "matrix.h"

struct Layer {

	int neurons;
	matrix *W, *B, *A, *Z, *dB, *dW;
	matrix *delta;
};

typedef struct Layer Layer;

struct Network {
	
	int num_layers;
	Layer *layers; //points to an array of layers.
};

typedef struct Network Network;

Network *create_network(int NUM_LAYERS , int *neurons);

void feed_forward(Network *net , float *input , int n);
// n  = size of input .

void back_prop(Network *net, int epochs , float lr , float **input , float **output , int m );
// m is the size of training data.


float compute_cost(Network *net , float *output , int n);
// computing cost for 1 input
// n = size of output

void save_model(Network *net , const char *filename);
Network *load_model(const char *filename);

void free_layer(Layer layer);

void free_network(Network *net);
