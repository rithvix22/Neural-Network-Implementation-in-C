#include "ML.h"


Network *create_network(int NUM_LAYERS , int *neurons){
	
	//layers are going to be 0 indexed. W[n-1] represents the weights from (n-2) -> (n-1).
	
	Network *net = (Network *)malloc(sizeof(Network));
	
	net->layers = (Layer *)malloc(sizeof(Layer)*NUM_LAYERS);
	
	net->num_layers = NUM_LAYERS;
	net->layers[0].neurons = neurons[0];
	net->layers[0].W = NULL;
	net->layers[0].B = NULL;
	net->layers[0].A = mat_alloc(neurons[0],1);
	net->layers[0].Z = NULL;
	net->layers[0].dB = NULL;
	net->layers[0].dW = NULL;
	net->layers[0].delta = NULL;

	for(int i =1;i<net->num_layers;i++){
		
		//creation.
		net->layers[i].neurons = neurons[i];
		net->layers[i].W = mat_alloc(net->layers[i].neurons,net->layers[i-1].neurons);
		net->layers[i].B = mat_alloc(net->layers[i].neurons,1);
		net->layers[i].A = mat_alloc(net->layers[i].neurons,1);
		net->layers[i].Z = mat_alloc(net->layers[i].neurons,1);
		net->layers[i].dB = mat_alloc(net->layers[i].neurons,1);
		net->layers[i].dW = mat_alloc(net->layers[i].neurons,net->layers[i-1].neurons);
		net->layers[i].delta = mat_alloc(net->layers[i].neurons,1);

		//initialization.	
		mat_rand(net->layers[i].W);
		mat_scale(net->layers[i].W, sqrtf(1.0f / net->layers[i-1].neurons), net->layers[i].W);
		mat_rand(net->layers[i].B);  // random biases in (-1, 1)
		mat_scale(net->layers[i].B, sqrtf(1.0f / net->layers[i-1].neurons), net->layers[i].B);
	}
	return net;

}

void feed_forward(Network *net , float *input , int n){
	
	// n is the number of entries in input.
	
	if(n != net->layers[0].neurons){
		printf("Invalid size of input\n");
		return;
	}

	for(int i =0;i<n;i++){
		net->layers[0].A->data[i][0] = input[i];
	}
	
	int NUM_LAYERS = net->num_layers;

	for(int i =1;i<NUM_LAYERS;i++){
		
		mat_mul(net->layers[i].W,net->layers[i-1].A,net->layers[i].Z);

		mat_a_s(net->layers[i].Z,net->layers[i].B,net->layers[i].Z,'+');

		if(i!= NUM_LAYERS-1){
			mat_relu(net->layers[i].Z,net->layers[i].A);

		}
		else {
			
			mat_copy(net->layers[i].A , net->layers[i].Z);
		}


	}
	
}

float compute_cost(Network *net , float *output , int n){
	
	int NUM_LAYERS = net->num_layers;

	if(net->layers[NUM_LAYERS-1].neurons != n){
		printf("Invalid output vector\n");
		return 0;
	}
	
	float cost = 0;
	float temp;
	for(int i =0;i<n;i++){
		
		temp = (output[i]-net->layers[NUM_LAYERS-1].A->data[i][0]);
		cost += temp*temp;	
	}

	return cost;
}

void back_prop(Network *net, int epochs, float lr, float **input, float **output, int m){
	
	//upadate , display the costs do it epochs times.
	
	for(int ii =0;ii<epochs;ii++){

		
		int NUM_LAYERS = net->num_layers;
		// calculating the updates.

		for(int l = 1; l < NUM_LAYERS; l++){
			mat_fill(net->layers[l].dW, 0);
			mat_fill(net->layers[l].dB, 0);
		}

		for(int i =0;i<m;i++){
			
			feed_forward(net,input[i],net->layers[0].A->rows);

			float temp = 0;

			for(int j =0; j<net->layers[NUM_LAYERS-1].neurons;j++){
				
				temp = net->layers[NUM_LAYERS-1].A->data[j][0];
				// FIXED: added 2.0f factor, the correct derivative of (a-y)^2 is 2*(a-y).
				net->layers[NUM_LAYERS-1].delta->data[j][0] = 2.0f * (temp-output[i][j]) ;
			}
			
			//now , we have calculated delta[L];
			// hardcode the update for the last layer then use forloop for updating each layer.
			// FIXED: accumulate dB across examples with '+', not overwrite with mat_copy.
			mat_a_s(net->layers[NUM_LAYERS-1].dB, net->layers[NUM_LAYERS-1].delta, net->layers[NUM_LAYERS-1].dB, '+');

			matrix *temp_mat = mat_alloc(net->layers[NUM_LAYERS-2].A->columns,net->layers[NUM_LAYERS-2].A->rows);
			mat_T(net->layers[NUM_LAYERS-2].A,temp_mat);
			matrix *dw_tmp = mat_alloc(net->layers[NUM_LAYERS-1].neurons,net->layers[NUM_LAYERS-2].neurons);
			mat_mul(net->layers[NUM_LAYERS-1].delta,temp_mat,dw_tmp);
			mat_a_s(net->layers[NUM_LAYERS-1].dW,dw_tmp,net->layers[NUM_LAYERS-1].dW,'+');	
			
			mat_free(temp_mat);
			mat_free(dw_tmp);

			//back propagating.
			for(int l = NUM_LAYERS-1; l>1; l--){
			

				// God equation.
				
				matrix *weight_T = mat_alloc(net->layers[l].W->columns,net->layers[l].W->rows);
				
				mat_T(net->layers[l].W,weight_T);

				matrix *deriv = mat_alloc(net->layers[l-1].Z->rows,net->layers[l-1].Z->columns);
				mat_relu_deriv(net->layers[l-1].Z,deriv);

				mat_mul(weight_T , net->layers[l].delta , net->layers[l-1].delta);	
				mat_mul_hadamard(net->layers[l-1].delta,deriv,net->layers[l-1].delta);
				
				mat_free(weight_T);
				mat_free(deriv);
		
				// now , updating for ** l-1 ** .

				// FIXED: accumulate dB across examples with '+', not overwrite with mat_copy.
				mat_a_s(net->layers[l-1].dB, net->layers[l-1].delta, net->layers[l-1].dB, '+');

				matrix *temp_mat = mat_alloc(net->layers[l-2].A->columns , net->layers[l-2].A->rows);
				mat_T(net->layers[l-2].A , temp_mat);
				matrix *dw_tmp = mat_alloc(net->layers[l-1].neurons,net->layers[l-2].neurons);
				mat_mul(net->layers[l-1].delta,temp_mat,dw_tmp);
				mat_a_s(net->layers[l-1].dW,dw_tmp,net->layers[l-1].dW,'+');

				mat_free(temp_mat);	
				mat_free(dw_tmp);

			}
		}

		for(int l =1 ; l<NUM_LAYERS ;l++){
			mat_scale(net->layers[l].dW , (-1)*lr/m , net->layers[l].dW);
			mat_scale(net->layers[l].dB , (-1)*lr/m , net->layers[l].dB);
			mat_a_s(net->layers[l].W,net->layers[l].dW,net->layers[l].W,'+');
			mat_a_s(net->layers[l].B,net->layers[l].dB,net->layers[l].B,'+');
		}

		// displaying the cost
	
		float cost = 0;
		for(int i = 0; i< m; i++){	

			cost += compute_cost(net,output[i],net->layers[NUM_LAYERS-1].neurons);
		}

		cost = cost/m;
		printf("EPOCH : %d , COST : %.6f\n",ii,cost);

	}


	
}

void save_model(Network *net, const char *filename){
    FILE *fp = fopen(filename, "w");

    fprintf(fp, "%d\n", net->num_layers);

    for(int i = 0; i < net->num_layers; i++)
        fprintf(fp, "%d ", net->layers[i].neurons);
    fprintf(fp, "\n");

    for(int i = 1; i < net->num_layers; i++){

        matrix *W = net->layers[i].W;
        matrix *B = net->layers[i].B;

        for(int r = 0; r < W->rows; r++){
            for(int c = 0; c < W->columns; c++)
                fprintf(fp, "%f ", W->data[r][c]);
            fprintf(fp, "\n");
        }

        for(int r = 0; r < B->rows; r++){
            fprintf(fp, "%f\n", B->data[r][0]);
        }
    }

    fclose(fp);
}

Network *load_model(const char *filename){
    FILE *fp = fopen(filename, "r");

    int num_layers;
    fscanf(fp, "%d", &num_layers);

    int *neurons = malloc(sizeof(int) * num_layers);
    for(int i = 0; i < num_layers; i++)
        fscanf(fp, "%d", &neurons[i]);

    Network *net = create_network(num_layers, neurons);

    for(int i = 1; i < num_layers; i++){

        matrix *W = net->layers[i].W;
        matrix *B = net->layers[i].B;

        for(int r = 0; r < W->rows; r++){
            for(int c = 0; c < W->columns; c++)
                fscanf(fp, "%f", &W->data[r][c]);
        }

        for(int r = 0; r < B->rows; r++){
            fscanf(fp, "%f", &B->data[r][0]);
        }
    }

    fclose(fp);
    free(neurons);

    return net;
}


void free_layer(Layer layer){
	
	mat_free(layer.W);
	mat_free(layer.B);
	mat_free(layer.A);
	mat_free(layer.Z);
	mat_free(layer.dB);
	mat_free(layer.dW);
	mat_free(layer.delta);
}

void free_network(Network *net){
	
	for(int i =0;i<net->num_layers;i++){
		
		free_layer(net->layers[i]);
	}
	free(net->layers);
	free(net);

}
