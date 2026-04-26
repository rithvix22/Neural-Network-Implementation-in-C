# Neural-Network-Implementation-in-C

Welcome to my project
My goal was to build a neural network from scratch in C <br><br>
All these files may look scary at the start <br>
Read the Readme so as to get a better understanding.<br>

# First Attempt :
My main goal was to get a working model <br> not worrying about the time / cost 
On my first attempt , I had hardcoded the update values of the weights and biases without actually using back propagation.
due to which , the time it took to train my neural network was huge . 
I was able to get a cost of 0.001 on 30 minutes of training (MSE).
what this means is that on average , the model gives an error of 0.03 error
so <br>
say x = 0.4 <br>
x<sup>2</sup> will be 0.16 but , my model gives 0.46 (pretty bad).
ML_first.c is the ML code corresponding to this.


# Second Attempt :
I implemented the Back prop Algorithm. for the function y = x<sup>2</sup> <br>
I was able to generate a minimum cost = 0.088812 ≈ 0.09 <br>
this means that for a given input of x between 0-1 , there is an error of &radic;0.09 = 0.3 (which is quite bad) <br> <br>
Note : <br>
1 . I have trained the NN on y = x<sup>2</sup> in the range x = [0,1] <br>
2 . I have limited the number of epochs to 100 so it doesnt take much time to run and , we get results quickly <br>
3 . Minimum error I was able to acheive was 0.0088812 after which it saturates. <br> <br>
4 . Instructions to run : <br>
  `make -f makefile_run` <br>
  `./a.out                ---> this trains the neural network and stores the weights and biases into network.txt` <br>
  `make -f makefile_tester` <br>
  `./a.out                ---> to give the prediction using the learnt weights and biases` <br><br>
5 . unfortunately , the neural network learns a constant line which is ≈ 0.3 . The issue I suspect is vanishing gradient<br><br>

The codes have been attached <br><br>
1 . ML.c is the neural network library<br>
2 . matrix.c is the matrix library <br>
3 . makefile_main is the makefile which gets the learnt weights and biases into network.txt<br>
4 . makefile_tester is for testing the data.<br>
5 . main.c is the application <br>
5 . makefile_main and makefile_lib are the makefiles with c files and makefile_lib_main and makefile_lib_tester are the ones corresponding to the mylib.a

# Third Attempt :
I changed the functions from sigmoid to Relu activation <br>
also , I changed the inputs from -0.5 to +0.5 rather than 0-1 <br>
by doing so , I was able to get the cost down to 0.006 which translates to an error of 0.07 nearly per example which is reasonably good <br><br>
The ML library corresponding to Relu function is ReluML.c , main is main_relu and , makefile is makefile_relu. <br>
Instructions to Run : <br>
`make -f makefile_relu`<br>
`./a.out                ---> training phase , stores the weights and biases into network.txt for future use`<br>
`make -f makefile_tester ` <br>
`./a.out                ---> tests the neural network` 

<br><br>
<img width="443" height="304" alt="image" src="https://github.com/user-attachments/assets/5bb412d3-e677-4dac-adab-d17df52a57b6" />
