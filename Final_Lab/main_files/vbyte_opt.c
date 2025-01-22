#include <stdio.h>
#include <stdlib.h>
#include "platform.h"
#include "xil_printf.h"
#include "xuartps.h"
#include "weights.h"
#include "test_images.h"
#include <xtime_l.h>
#include <time.h>

#include <math.h>

#define n_bias0 64
#define n_weights0 50176
#define n_bias1 32
#define n_weights1 2048
#define n_bias2 10
#define n_weights2 320

typedef short int DATA;
typedef unsigned char vbyte;

#define FIXED2FLOAT(a, qf) (((float) (a)) / (1<<qf))
#define FLOAT2FIXED(a, qf) ((short int) round((a) * (1<<qf)))

#define _MAX_ (1 << (sizeof(DATA)*8-1))-1
#define _MIN_ -(_MAX_+1)

#define CLOCKFREQ 333000000.0 // 333 MHz


/* --- Debugging Settings ---*/
//#define DONTASKVALUES
//#define DEBUGSENDVALUES

/* --- Optimization Settings --- */
#define IMAGETRANOPT // Transfer of images Q0.8 instead of Q8.8
#define VALUESTRANSOPT // Transfer of the bias/weight values with Q1.7 instead of Q8.8
#define MEMORYOPT

// DNN functions to compose your network

void FC_forward(DATA* input, DATA* output, int in_s, int out_s, DATA* weights, DATA* bias, int qf) ;
static inline long long int saturate(long long int mac);
static inline void relu_forward(DATA* input, DATA* output, int size);
static inline void relu_forward2(vbyte* input, vbyte* output, int size);
int resultsProcessing(DATA* results, int size);

DATA readPixelfromUART_opt(){
	unsigned char in;
	DATA out;
	in = XUartPs_RecvByte(STDIN_BASEADDRESS);
	out = (DATA) in;
	return out;
}

vbyte readPixelfromUART_opt2(){
	unsigned char in;
	in = XUartPs_RecvByte(STDIN_BASEADDRESS);
	return in;
}

DATA readQ1_7ValuesFromUart(){
	unsigned char in;
	DATA sign, out;
	in = XUartPs_RecvByte(STDIN_BASEADDRESS);

	if (in & 0x80){
		sign = 0xFF00;
	}
	else{
		sign = 0x0000;
	}
	out = (DATA) (in << 1) | sign;
	return out;
}

vbyte readQ1_7ValuesFromUart2(){
	unsigned char in;
	in = XUartPs_RecvByte(STDIN_BASEADDRESS);
	return in;
}

// implement your function receiving from UART
DATA readDATAfromUART(){ // reads a sequence of bytes and composes the DATA
	unsigned char in1, in2;
	DATA out;
	in1 = XUartPs_RecvByte(STDIN_BASEADDRESS);
	in2 = XUartPs_RecvByte(STDIN_BASEADDRESS);
	out = (in2 << 8) | in1;
	return out;
}

void readVByte(vbyte * pArray, int len){
	for(int i = 0; i < len; i++){
#if defined(VALUESTRANSOPT) && defined(MEMORYOPT)
		pArray[i] = readQ1_7ValuesFromUart2();
#else
		printf("Read vbyte does not work with this setup!\n");
#endif
	}
}

void readDATA(DATA * pArray, int len){
	for(int i = 0; i < len; i++){
#ifdef VALUESTRANSOPT
		pArray[i] = readQ1_7ValuesFromUart();
#else
		pArray[i] = readDATAfromUART();
#endif
	}
}

/* --- Test Images --- */
DATA test_images[10][28*28] = {{imm_test_0},{imm_test_1},{imm_test_2},{imm_test_3},{imm_test_4},{imm_test_5},{imm_test_6},{imm_test_7},{imm_test_8},{imm_test_9}};

// these buffers are for running the test images
DATA gemm0_bias[n_bias0] = {bias0};
DATA gemm0_weights[n_weights0] = {weights0} ;
DATA gemm1_bias[n_bias1] = {bias1};
DATA gemm1_weights[n_weights1] = {weights1};
DATA gemm2_bias[n_bias2] = {bias2};
DATA gemm2_weights[n_weights2] = {weights2};

DATA output_gemm0[64] = {0};
DATA  input_gemm1[64] = {0};
DATA output_gemm1[32] = {0};
DATA  input_gemm2[32] = {0};
DATA output_gemm2[10] = {0};

/* Structure for holding the values for one layer in the DNN */
typedef struct{
	int neurons;
	int input_size;
	int weights_size;
	DATA * input; // size == output of previous layer
	DATA * output; // size == neurons
	DATA * bias; // size == neurons
	DATA * weights; // size == input_size * neurons
} layer;

typedef struct{
	int neurons;
	int input_size;
	int weights_size;
	vbyte * input; // size == output of previous layer
	vbyte * output; // size == neurons
	vbyte * bias; // size == neurons
	vbyte * weights; // size == input_size * neurons
} layer2;

int main(){
	init_platform();

	//UART setup
	XUartPs Uart_1_PS;
	u16 DeviceId_1= XPAR_PS7_UART_1_DEVICE_ID;
	int Status_1;
	XUartPs_Config *Config_1;
	Config_1 = XUartPs_LookupConfig(DeviceId_1);
	if (NULL == Config_1) {
	return XST_FAILURE;
	}
	/*the default configuration is stored in Config and it can be used to initialize the controller */
	Status_1 = XUartPs_CfgInitialize(&Uart_1_PS, Config_1, Config_1->BaseAddress);
	if (Status_1 != XST_SUCCESS) {
	return XST_FAILURE;
	}
	// Set the BAUD rate
	u32 BaudRate = (u32)115200;
	Status_1 = XUartPs_SetBaudRate(&Uart_1_PS, BaudRate);
	if (Status_1 != (s32)XST_SUCCESS) {
	return XST_FAILURE;
	}
	//END UART SETUP
	xil_printf ("Started\r\n");

	runTest(); // Runs the DNN with the values from weights.h


	/* --- DNN Settings --- */ //until further implementation, input the DNN specific parameters here
#define TESTDNN // TESTDNN, GROUP1, ...
#ifdef TESTDNN
	int layers_num = 3;
	int input_image_size = 784;
	int neurons_num[3] = {64, 32, 10};
#endif
#ifdef GROUP0
	int layers_num = 2;
	int input_image_size = 784;
	int neurons_num[2] = {64, 10};
#endif

	/* --- DNN setup --- */
	/* Construct DNN */
#ifdef MEMORYOPT
	layer2* DNN = (layer2*) malloc(layers_num * sizeof(layer2));
#else
	layer* DNN = (layer*) malloc(layers_num * sizeof(layer));
#endif
	if (DNN == NULL){
		printf("Malloc failure: DNN");
		return 1;
	}
	int previous_output_size = input_image_size;
	for(int i = 0; i < layers_num; i++){
		DNN[i].neurons = neurons_num[i];
#ifdef MEMORYOPT
		DNN[i].input = (vbyte*) calloc(previous_output_size, sizeof(vbyte));
#else
		DNN[i].input = (DATA*) calloc(previous_output_size, sizeof(DATA));
#endif
		if (DNN[i].input == NULL){
			printf("Malloc failure: DNN[%d].input", i);
			return 1;
		}
		DNN[i].input_size = previous_output_size;
#ifdef MEMORYOPT
		DNN[i].output = (vbyte*) calloc(neurons_num[i], sizeof(vbyte));
#else
		DNN[i].output = (DATA*) calloc(neurons_num[i], sizeof(DATA));
#endif
		if (DNN[i].output == NULL){
			printf("Malloc failure: DNN[%d].output", i);
			return 1;
		}
#ifdef MEMORYOPT
		DNN[i].bias = (vbyte*) calloc(neurons_num[i], sizeof(vbyte));
#else
		DNN[i].bias = (DATA*) calloc(neurons_num[i], sizeof(DATA));
#endif
		if (DNN[i].bias == NULL){
			printf("Malloc failure: DNN[%d].bias", i);
			return 1;
		}
#ifdef MEMORYOPT
		DNN[i].weights = (vbyte*) calloc(neurons_num[i] * previous_output_size, sizeof(vbyte));
#else
		DNN[i].weights = (DATA*) calloc(neurons_num[i] * previous_output_size, sizeof(DATA));
#endif
		if (DNN[i].weights == NULL){
			printf("Malloc failure: DNN[%d].weights", i);
			return 1;
		}
		DNN[i].weights_size = previous_output_size * neurons_num[i];
		previous_output_size = neurons_num[i];
	}
#ifdef DEBUGSENDVALUES
	while(1){
#endif
#if !defined(DONTASKVALUES)
	int T1 = 0;
	int T2 = 0;
	int dTic = 0;
	float dT = 0;

	/* Load DNN */
	for(int i = 0; i < layers_num; i++){
		printf("Send bias for layer %d\n", i);
		while (!XUartPs_IsReceiveData(STDIN_BASEADDRESS)) {
			; // wait unitl first byte of the image is sent before starting the timer
		}
		T1 = Xil_In32(GLOBAL_TMR_BASEADDR);
#ifdef MEMORYOPT
		readVByte(DNN[i].bias, DNN[i].neurons);
#else
		readDATA(DNN[i].bias, DNN[i].neurons);
#endif
		T2 = Xil_In32(GLOBAL_TMR_BASEADDR);
		dTic = T2 - T1;
		dT = dTic / CLOCKFREQ,
		printf("Sending bias for layer %d took: %d tics or %.6f sec\n", i, dTic, dT);
		printf("Send weights for layer %d\n", i);
		while (!XUartPs_IsReceiveData(STDIN_BASEADDRESS)) {
			; // wait unitl first byte of the image is sent before starting the timer
		}
		T1 = Xil_In32(GLOBAL_TMR_BASEADDR);
#ifdef MEMORYOPT
		readVByte(DNN[i].weights, DNN[i].weights_size);
#else
		readDATA(DNN[i].weights, DNN[i].weights_size);
#endif
		T2 = Xil_In32(GLOBAL_TMR_BASEADDR);
		dTic = T2 - T1;
		dT = dTic / CLOCKFREQ,
		printf("Sending weights for layer %d took: %d tics or %.6f sec\n", i, dTic, dT);
	}
#endif
#ifdef DEBUGSENDVALUES
	}
#endif

	int result = -1;
#if !defined(MEMORYOPT)
	/* Test sent DNN on test images */
	int test_true = 0;
	int test_false = 0;
	for (int i=0; i<=9; i++){
		DNN[0].input = &test_images[i];
		result = processImage(DNN, layers_num);
		if (result == i) {
			printf("Test image %d correctly identified as %d\n", i, result);
			test_true += 1;
		}
		else {
			printf("Test image %d incorrect identified as %d\n", i, result);
			test_false += 1;
		}
	}

	printf("%d/%d test images correctly identified\n", test_true, (test_true + test_false));
#endif

	/* Variables for timing */
	int t1 = 0;
	int t2 = 0;
	int t3 = 0;
	int t4 = 0;
	int dtic_readImage = 0;
	int dtic_processImage = 0;
	int dtic_sendRespond = 0;
	int dtic_respondTime = 0; // without the print calls
	float dt_readImage = 0.0;
	float dt_processImage = 0.0;
	float dt_sendRespond = 0;
	float dt_respondTime = 0.0;

#ifdef MEMORYOPT
	vbyte image[28*28] = {0};
#else
	DATA image[28*28] = {0};
#endif
	while (1){
		printf("Waiting for the image...\n");
		while (!XUartPs_IsReceiveData(STDIN_BASEADDRESS)) {
			; // wait unitl first byte of the image is sent before starting the timer
		}
		t1 = Xil_In32(GLOBAL_TMR_BASEADDR);
#ifdef MEMORYOPT
		readImage2(DNN[0].input, input_image_size);
#else
		readImage(DNN[0].input, input_image_size);
#endif
		t2 = Xil_In32(GLOBAL_TMR_BASEADDR);
		result = processImage(DNN, layers_num);
		t3 = Xil_In32(GLOBAL_TMR_BASEADDR);
		xil_printf("Image shows the number %d\r\n", result);
		t4 = Xil_In32(GLOBAL_TMR_BASEADDR);

		dtic_readImage = t2 - t1;
		dtic_processImage = t3 - t2;
		dtic_sendRespond = t4 - t3;
		dtic_respondTime = t4 - t1;
		dt_readImage = dtic_readImage / CLOCKFREQ;
		dt_processImage = dtic_processImage / CLOCKFREQ;
		dt_sendRespond = dtic_sendRespond / CLOCKFREQ;
		dt_respondTime = dtic_respondTime / CLOCKFREQ;
		printf("Reading the image took: %d Tics or %.5f sec\n", dtic_readImage, dt_readImage); //printf needed because of printing floating point values
		printf("Processing the image took: %d Tics or %.6f sec\n", dtic_processImage, dt_processImage);
		printf("Sending the responds took: %d Tics or %.7f sec\n", dtic_sendRespond, dt_sendRespond);
		printf("The respond time was: %d Tics or %.5f sec\n", dtic_respondTime, dt_respondTime);
	}

	/* --- Cleanup --- */
	for (int i = 0; i < layers_num; i++) {
		free(DNN[i].input);
		free(DNN[i].output);
		free(DNN[i].bias);
		free(DNN[i].weights);
	}
	free(DNN);

    cleanup_platform();
    return 0;
}



void readImage(DATA * image, int size){
	for(int i = 0; i < size; i++){
#ifdef IMAGETRANOPT
		image[i] = readPixelfromUART_opt();
#else
		image[i] = readDATAfromUART();
#endif
	}
}

void readImage2(vbyte * image, int size){
	for(int i = 0; i < size; i++){
#if defined(IMAGETRANOPT) && defined(MEMORYOPT) && defined(VALUESTRANSOPT)
		image[i] = readPixelfromUART_opt2();
#else
		printf("Read Image does not work with this setup!");
#endif
	}
}

int processImage(layer * dnn, int dnn_depth) {
#ifdef MEMORYOPT
	for(int i = 0; i < dnn_depth - 1; i++){
		FC_forward2(dnn[i].input, dnn[i].output, dnn[i].input_size, dnn[i].neurons, dnn[i].weights, dnn[i].bias, 7);
		relu_forward2(dnn[i].output, dnn[i+1].input, dnn[i].neurons);
	}
	FC_forward2(dnn[dnn_depth-1].input, dnn[dnn_depth-1].output, dnn[dnn_depth-1].input_size, dnn[dnn_depth-1].neurons, dnn[dnn_depth-1].weights, dnn[dnn_depth-1].bias, 7);
	return resultsProcessing2(dnn[dnn_depth-1].output, dnn[dnn_depth-1].neurons);
#else
	for(int i = 0; i < dnn_depth - 1; i++){
		FC_forward(dnn[i].input, dnn[i].output, dnn[i].input_size, dnn[i].neurons, dnn[i].weights, dnn[i].bias, 8);
		relu_forward(dnn[i].output, dnn[i+1].input, dnn[i].neurons);
	}
	FC_forward(dnn[dnn_depth-1].input, dnn[dnn_depth-1].output, dnn[dnn_depth-1].input_size, dnn[dnn_depth-1].neurons, dnn[dnn_depth-1].weights, dnn[dnn_depth-1].bias, 8);
	return resultsProcessing(dnn[dnn_depth-1].output, dnn[dnn_depth-1].neurons);
#endif
}

void runTest(){
	int result = -1;
	int test_true = 0;
	int test_false = 0;
	for (int i=0; i<=9; i++){
		result = processTestImage(&test_images[i]);
		if (result == i) {
			printf("Test image %d correctly identified as %d\n", i, result);
			test_true += 1;
		}
		else {
			printf("Test image %d incorrect identified as %d\n", i, result);
			test_false += 1;
		}
	}

	printf("%d/%d test images correctly identified\n", test_true, (test_true + test_false));
}

int processTestImage(DATA * image){
	FC_forward(image, output_gemm0, 784, 64, gemm0_weights, gemm0_bias, 8);
	relu_forward(output_gemm0, input_gemm1, 64);
	FC_forward(input_gemm1, output_gemm1, 64, 32, gemm1_weights, gemm1_bias, 8);
	relu_forward(output_gemm1, input_gemm2, 32);
	FC_forward(input_gemm2, output_gemm2, 32, 10, gemm2_weights, gemm2_bias, 8);
	return resultsProcessing(output_gemm2, 10);
}


void FC_forward(DATA* input, DATA* output, int in_s, int out_s, DATA* weights, DATA* bias, int qf) {
	int hkern = 0;
	int wkern = 0;
	long long int mac = 0;
	DATA current = 0;

	/* foreach row in kernel */
//	#pragma omp parallel for private (hkern, wkern, mac, current)
	for (hkern = 0; hkern < out_s; hkern++) {
		mac = ((long long int)bias[hkern]) << qf;
		for (wkern = 0; wkern < in_s; wkern++) {
			current = input[wkern];
			mac += current * weights[hkern*in_s + wkern];
		}
		output[hkern] = (DATA)(mac >> qf);
	}
}


void FC_forward2(vbyte* input, vbyte* output, int in_s, int out_s, vbyte* weights, vbyte* bias, int qf) {

	int hkern = 0;
	int wkern = 0;
	long long int mac = 0;
	vbyte current = 0;

	/* foreach row in kernel */
//	#pragma omp parallel for private (hkern, wkern, mac, current)
	for (hkern = 0; hkern < out_s; hkern++) {
		mac = ((long long int)bias[hkern]) << qf;
		for (wkern = 0; wkern < in_s; wkern++) {
			current = input[wkern];
			mac += current * weights[hkern*in_s + wkern];
		}
		output[hkern] = (vbyte) saturate((mac >> qf));
	}
}

static inline long long int saturate(long long int mac)
{

	if(mac > _MAX_) {
		printf("[WARNING] Saturation.mac: %lld -> %llx _MAX_: %d  _MIN_: %d  res: %d\n",  mac, mac, _MAX_, _MIN_, _MAX_);
		return _MAX_;
	}

	if(mac < _MIN_){
		printf( "[WARNING] Saturation. mac: %lld -> %llx _MAX_: %d  _MIN_: %d  res: %d\n",  mac, mac, _MAX_, _MIN_, _MIN_);
		return _MIN_;
	}

	//printf("mac: %lld -> %llx _MAX_: %lld  _MIN_: %lld  res: %lld\n", mac, mac, _MAX_, _MIN_, mac);
    return mac;

}

static inline void relu_forward(DATA* input, DATA* output, int size) {
	int i = 0;
	for(i = 0; i < size; i++) {
		DATA v = input[i];
		v = v > 0 ? v : 0;
		output[i] = v;
	}
}

static inline void relu_forward2(vbyte* input, vbyte* output, int size) {
	int i = 0;
	for(i = 0; i < size; i++) {
		vbyte v = input[i];
		v = v > 0 ? v : 0;
		output[i] = v;
	}
}

#define SIZEWA 10
int resultsProcessing(DATA* results, int size){
  char *labels[10]={"digit 0", "digit 1", "digit 2", "digit 3", "digit 4", "digit 5", "digit 6", "digit 7", "digit 8", "digit 9"};


  int size_wa = SIZEWA;
  float  r[SIZEWA];
  int  c[SIZEWA];
  float results_float[SIZEWA];
  float sum=0.0;
  DATA max=0;
  int max_i;
  for (int i =0;i<size_wa;i++){
	  results_float[i] = FIXED2FLOAT(results[i],8);
	int n;
	if (results[i]>0)
	  n=results[i];
	else
	  n=-results[i];
	if (n>max){
	  max=n;
	  max_i=i;
	}
  }
  for (int i =0;i<size_wa;i++)
	sum+=exp(results_float[i]);

  for (int i =0;i<size_wa;i++){
	r[i]=exp(results_float[i]) / sum;
	c[i]=i;
  }
  for (int i =0;i<size_wa;i++){
	for (int j =i;j<size_wa;j++){
	  if (r[j]>r[i]){
		float t= r[j];
		r[j]=r[i];
		r[i]=t;
		int tc= c[j];
		c[j]=c[i];
		c[i]=tc;
	  }
	}
  }
  int top0=0;
  float topval=results_float[0];
  for (int i =1;i<size_wa;i++){
	if (results_float[i]>topval){
	  top0=i;
	  topval=results_float[i];
	}
  }
  return top0;
}

int resultsProcessing2(vbyte* results, int size){
  char *labels[10]={"digit 0", "digit 1", "digit 2", "digit 3", "digit 4", "digit 5", "digit 6", "digit 7", "digit 8", "digit 9"};


  int size_wa = SIZEWA;
  float  r[SIZEWA];
  int  c[SIZEWA];
  float results_float[SIZEWA];
  float sum=0.0;
  DATA max=0;
  int max_i;
  DATA result = 0;
  DATA sign = 0x0000;
  for (int i =0;i<size_wa;i++){
	  if (results[i] & 0x80){
		  sign = 0xFF00;
	  }
	  else {
		  sign = 0x0000;
	  }
	  result = (results[i] << 1) | sign;
	  results_float[i] = FIXED2FLOAT(result,7);
	int n;
	if (results[i]>0)
	  n=results[i];
	else
	  n=-results[i];
	if (n>max){
	  max=n;
	  max_i=i;
	}
  }
  for (int i =0;i<size_wa;i++)
	sum+=exp(results_float[i]);

  for (int i =0;i<size_wa;i++){
	r[i]=exp(results_float[i]) / sum;
	c[i]=i;
  }
  for (int i =0;i<size_wa;i++){
	for (int j =i;j<size_wa;j++){
	  if (r[j]>r[i]){
		float t= r[j];
		r[j]=r[i];
		r[i]=t;
		int tc= c[j];
		c[j]=c[i];
		c[i]=tc;
	  }
	}
  }
  int top0=0;
  float topval=results_float[0];
  for (int i =1;i<size_wa;i++){
	if (results_float[i]>topval){
	  top0=i;
	  topval=results_float[i];
	}
  }
  return top0;
}
