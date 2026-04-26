#include "ML.h"

int main(){
	Network *net = load_model("network.txt");
	
	int m;
	printf("Enter the number of inputs : ");
	scanf("%d",&m);
	float temp;
	
	float **input = (float **)malloc(sizeof(float *)*m);

	for(int i = 0 ; i<m ;i++){

		input[i] = (float *)malloc(sizeof(float)*1);
		printf("Input : ");
		scanf("%f",&temp);
		input[i][0] = temp;
		feed_forward(net , input[i] ,1);
		printf("Output : %f\n",net->layers[net->num_layers-1].A->data[0][0]);
	}
}
