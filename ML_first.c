#include "matrix.h"

struct network{
	
	matrix *a1;

	matrix *w2,*b2;
	matrix *z2,*a2;
	
	matrix *w3,*b3;
	matrix *z3,*a3;

	float learning_rate;
};

typedef struct network network;

void feed_forward(network *net , float input){
	
	net->a1->data[0][0] = input;

	mat_mul(net->w2,net->a1,net->z2);
	mat_a_s(net->z2,net->b2,net->z2,'+');
	mat_sig(net->z2,net->a2);

	mat_mul(net->w3,net->a2,net->z3);
	mat_a_s(net->z3,net->b3,net->z3,'+');
	mat_sig(net->z3,net->a3);

}

void free_net(network *net){
	mat_free(net->a1); 
	
	mat_free(net->w2);
	mat_free(net->b2);
	mat_free(net->z2);
	mat_free(net->a2);

	mat_free(net->w3);
	mat_free(net->b3);
	mat_free(net->z3);
	mat_free(net->a3);

	free(net);
}

float compute_cost(network *net , float *input , float *output ,int n){
	
	float cost = 0;
	for(int i=0;i<n;i++){
		
		feed_forward(net,input[i]);
		cost += pow((net->a3->data[0][0]-output[i]),2);
	}
	
	cost = cost/n;
	return cost;
}

void back_prop(network *net , int epochs , float lr , float *input , float *output , int n ){

	for(int l = 0;l<epochs;l++){
		//B3 : (1,1) update.
		
		float temp = 0;
		float B3_grad = 0;
		for(int i =0;i<n;i++){
					
			feed_forward(net,input[i]);
			temp = net->a3->data[0][0];
			B3_grad += (temp-output[i])*temp*(1-temp);
		}
		
		B3_grad = (B3_grad*2)/n;

		//W3 : (1,n) update
		
		float sum;

		float *W3_grad = (float *)malloc(sizeof(float)*net->w3->columns);
		
		for(int j=0;j<net->w3->columns;j++){
			sum =0;
			for(int i =0;i<n;i++){
					
				feed_forward(net,input[i]);
				temp = net->a3->data[0][0];
				sum += (temp-output[i])*temp*(1-temp)*(net->a2->data[j][0]);
			}
			W3_grad[j] = (sum*2)/n;
		}

		//B2 : (n,1) update.
		
		float aj;
		float *B2_grad = (float *)malloc(sizeof(float ) * net->a2->rows);
		for(int j =0 ; j<net->a2->rows;j++){
			sum  =0;		
			for(int i =0;i<n;i++){
				
				feed_forward(net,input[i]);
				temp = net->a3->data[0][0];
				aj = net->a2->data[j][0];		
				sum += (temp-output[i])*temp*(1-temp)*(net->w3->data[0][j])*(aj)*(1-aj);
			}

			B2_grad[j] = (sum*2)/n ;
		}

		//W2 : (n,1) update.
		
		float *W2_grad = (float *)malloc(sizeof(float ) * net->a2->rows);
		for(int j =0;j<net->w2->rows;j++){
			
			sum =0;
			for(int i =0;i<n;i++){
				
				feed_forward(net,input[i]);
				temp = net->a3->data[0][0];
				float x_input = input[i];

	sum += (temp - output[i]) * temp * (1 - temp)
       * net->w3->data[0][j]
       * net->a2->data[j][0] * (1 - net->a2->data[j][0])
       * x_input;
			}
			W2_grad[j] = (sum*2)/n;
		}
		
		
		net->b3->data[0][0] = net->b3->data[0][0] - (lr)*(B3_grad);
		
		for(int i =0;i<net->w3->columns;i++){
			
			net->w3->data[0][i] += (-1)*lr*W3_grad[i];
		}	
		
		for(int i =0;i<net->a2->rows;i++){
			
			net->b2->data[i][0] += (-1)*lr*B2_grad[i];
		}
		
		for(int j =0;j<net->a2->rows;j++){
			
			net->w2->data[j][0] += (-1)*lr*W2_grad[j];
		}

		float cost = compute_cost(net,input,output,n);
		printf("cost = %.3f EPOCH : {%d}\n",cost,l);
		
		free(W3_grad);
		free(B2_grad);
		free(W2_grad);
	}

	
}

int main(int argc,char *argv[]){
	
	srand(time(NULL));

	int m= atoi(argv[1]) ;
	int n= atoi(argv[2]);

	network *Net = (network *)malloc(sizeof(network));
	
	Net->a1 = mat_alloc(1,1);
	
	Net->w2 = mat_alloc(n,1);
	Net->b2 = mat_alloc(n,1);
	Net->z2 = mat_alloc(n,1);
	Net->a2 = mat_alloc(n,1);

	Net->w3 = mat_alloc(1,n);
	Net->b3 = mat_alloc(1,1);
	Net->z3 = mat_alloc(1,1);
	Net->a3 = mat_alloc(1,1);
	
	mat_rand(Net->w2);
	//mat_print(Net->w2);
	mat_rand(Net->b2);
	mat_rand(Net->w3);
	mat_rand(Net->b3);

	float *INPUT = (float *)malloc(sizeof(float ) * m);
	float *OUTPUT = (float *)malloc(sizeof(float ) * m);

	y_x2_generator(INPUT,OUTPUT,m);
	//printf("%f %f\n",INPUT[0],OUTPUT[0]);
	//printf("%f %f\n",INPUT[1],OUTPUT[1]);
	
	
	//printf("%f\n",compute_cost(Net,INPUT,OUTPUT,n));
	int epochs = 100000;
	float lr = 0.02;

	back_prop(Net,epochs,lr,INPUT,OUTPUT,m);
	
	free_net(Net);
	
	
}
