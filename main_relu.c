#include "ML.h"
void free_function(float **data , int m , int n){
	
	for(int i =0 ;i<m;i++){
		
		free(data[i]);
	}

	free(data);

}

int main(){

	srand(time(NULL));	
	int layers[5] = {1,10,12,10,1};
	Network *net = create_network(5,layers);
	
	int m = 2000;
	int n = 1;
	float **input = malloc(sizeof(float *) * m);

	for(int i =0 ;i<m;i++){
		
		input[i] = malloc(sizeof(float) * n);
	}

	float **output = malloc(sizeof(float *) * m);

	for(int i =0 ; i<m ;i++){
		
		output[i] = malloc(sizeof(float ) *n);
	}
	
	float x = 0.0f;

	for(int i =0;i<m;i++){
		
		input[i][0] = x-0.5f;
		output[i][0] = (x-0.5f)*(x-0.5f);
		x += 0.0005f;
	}
	

	//data has been generated now , feeding it to the neural network.

	back_prop(net,100000,0.00001,input,output,2000);

	save_model(net,"network.txt");
	free_function(input,m,n);
	free_function(output,m,n);
	free_network(net);

}
