#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <memory>
#include <iostream>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cusparse.h>
#include <cstdlib>
#include "pcg.cuh"
double* x, * devx, * val, * gra, * r, * graMax;
double*hes_value;
int size;
int* pos_x, * pos_y;
int* csr;
thrust::pair<int, int> *device_pos;
__device__ __host__ inline double sqr(double x) {
	return x * x;
}
__device__ void wait() {
	for (int i = 1; i <= 10000000; i++);
}

__global__ void sum_val(double* val, double* r) {
	int index = threadIdx.x;
	for (int i = 1; i < blockDim.x; i <<= 1) {
		if (index % (i << 1) == i) {
			val[index - i] += val[index];
		}
		__syncthreads();
	}
	if (index == 0) {
		r[0] = val[0];
	}

}

__device__ __host__ inline double Max(double x, double y) {
	x = fabs(x);
	y = fabs(y);
	return x > y ? x : y;
}

__global__ void max_gra(double* gra, double* max) {
	int index = threadIdx.x;
	for (int i = 1; i < blockDim.x; i <<= 1) {
		if (index % (i << 1) == i) {
			gra[index - i] = Max(gra[index - i], gra[index]);
		}
		__syncthreads();
	}
	if (index == 0) {
		max[0] = gra[0];
	}

}

__global__ void calculate_val(double* devx, double* val, int size) {
	int index = threadIdx.x;
	int pre = index - 1;
	if (pre < 0) pre += size;
	int next = index + 1;
	if (next >= size) next -= size;
	val[index] = sqr(sin(devx[pre] * devx[index])) * sqr(sin(devx[next] * devx[index]));
	//	wait();
}


__global__ void calculate_gra(double* devx, double* gra,int size) {
	int index = threadIdx.x;
	int pre = index - 1;
	if (pre < 0) pre += size;
	int next = index + 1;
	if (next >= size) next -= size;
	gra[index] = devx[pre] * sin(2.0 * devx[index] * devx[pre]) + devx[next] * sin(2.0 * devx[index] * devx[next]);
	printf("gra %d %d %d %f %f %f\n", pre,index,next,sqr(devx[index]), devx[pre] * sin(2.0 * devx[index] * devx[pre]),gra[index]);
	//	wait();
}

__global__ void add_vec(double* sum, double* addA) {
	int index = threadIdx.x;
	printf("add %d %f\n", index, addA[index]);
	sum[index] += addA[index];
}

__global__ void minus_gra(double* gra) {
	gra[threadIdx.x]=0.0-gra[threadIdx.x];
	//printf("%f\n", gra[threadIdx.x]);
}



__global__ void calculate_pos(thrust::pair<int, int>* pos, double* devx, double* val, int N) {
	int index = threadIdx.x;
	printf("%d\n", index);
	int pre = index - 1 == -1 ? N - 1 : index - 1;
	int next = index + 1 == N ? 0 : index + 1;
	pos[3 * index] = thrust::make_pair<int, int>(index, pre);
	pos[3 * index + 1] = thrust::make_pair<int, int>(index, index);
	pos[3 * index + 2] = thrust::make_pair<int, int>(index, next);
	val[3 * index] = sin(2 * devx[index] * devx[pre]) + 2 * devx[index] * devx[pre] * cos(2 * devx[index] * devx[pre]);
	val[3 * index + 1] = 2 * sqr(devx[pre]) * cos(2 * devx[index] * devx[pre]) + 2 * sqr(devx[next]) * cos(2 * devx[index] * devx[next]);
	val[3 * index + 2] = sin(2 * devx[index] * devx[next]) + 2 * devx[index] * devx[next] * cos(2 * devx[index] * devx[next]);;

}

__global__ void create_tuple(double* devx, int* pos_x, int* pos_y, double* value, int N) {
	int index = threadIdx.x;
	if (index < N) {
		pos_x[index] = index;
		pos_y[index] = index;
		value[index] = 2 * cosf(2 * devx[index]);
	}
	else if(index == N){
		pos_x[index] = N; 

	}
}
PCG *one;
bool first = true;
double* init_pcg(int N, int NNZ, double* device_As, double* device_Bs, int* device_IAs, int* device_JAs) {
	if (first) {
		one = new PCG(N, NNZ, device_As, device_Bs, device_IAs, device_JAs, DeviceArray);
	}
	else {
		one->update_hes(device_As, device_Bs, DeviceArray);
	}
	return one->solve_pcg();
}

__global__ void print(thrust::pair<int, int>* pos,double*val, int size) {
	for (int i = 0; i < size; i++)
		printf("%d %d %d %f\n", i, pos[i].first, pos[i].second, val[i]);
}
__global__ void print(double *pos, int size) {
	for (int i = 0; i < size; i++)
		printf("%f\n",pos[i]);
}
//__global__ void decouple_pos(thrust::pair<int, int>* pos, int* pos_x, int* pos_y, double * value) {
//	int index = threadIdx.x;
//	pos_x[index] = pos[index].first;
//	pos_y[index] = pos[index].second;
//	printf("hes %d %d %d %f\n", index, pos_x[index], pos_y[index], value[index]);
//}

int main() {
	scanf("%d", &size);
	x = new double[size];
	for (int i = 0; i < size; i++) x[i] = i * 1.0 + 10.0;
	cudaMalloc((void**)&devx, size * sizeof(double));
	cudaMalloc((void**)&gra, size * sizeof(double));
	cudaMalloc((void**)&r, sizeof(double));
	cudaMalloc((void**)&val, size * sizeof(double));
	cudaMalloc((void**)& graMax, sizeof(double));
	cudaMalloc((void**)& pos_x, (3*size+1) * sizeof(int));
	cudaMalloc((void**)& pos_y, 3*size * sizeof(int));
	cudaMalloc((void**)& csr, (size+1) * sizeof(int));
	cudaMalloc((void**)& hes_value, 3*size * sizeof(double));
	cudaMemcpy(devx, x, size * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMalloc((void**)& device_pos, 3* size * sizeof(thrust::pair<int, int>));
	//calculate_pos << <1, size >> > (device_pos, devx, hes_value, size);
	//thrust::device_ptr<double> dev_data_ptr(hes_value);
	//thrust::device_ptr<thrust::pair<int, int>> dev_keys_ptr(device_pos);
	//thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr+size * 3, dev_data_ptr);
	//decouple_pos << <1, size * 3 >> > (device_pos, pos_x, pos_y);
	//print << <1, 1 >> > (device_pos,size*3);

	//cusparseHandle_t handle;
	//cusparseStatus_t status = cusparseCreate(&handle);
	//cusparseXcoo2csr(handle, pos_x, 3 * size, size, csr, CUSPARSE_INDEX_BASE_ZERO);
	//print << <1, 1 >> > (csr, size + 1);


	calculate_val <<<1, size>>> (devx, val, size);
	cudaThreadSynchronize();

	sum_val <<<1, size>>> (val, r);	
	cudaThreadSynchronize();
	
	calculate_gra << <1, size >> > (devx, gra, size);
	cudaThreadSynchronize();
	double hostVal;
	double host_maxGra;
	double* delta_x;
	max_gra << <1, size >> > (gra, graMax);
	cudaMemcpy(&host_maxGra, graMax, sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	calculate_gra << <1, size >> > (devx, gra, size);
	cudaThreadSynchronize();
	double eps = 1e-6;
	while (abs(host_maxGra) > eps) {
		calculate_gra << <1, size >> > (devx, gra, size);
		cudaThreadSynchronize();
		calculate_val << <1, size >> > (devx, val, size);

		sum_val << <1, size >> > (val, r);
		cudaThreadSynchronize();
		//system("pause");
		minus_gra << <1, size >> > (gra);

		calculate_pos << <1, size >> > (device_pos, devx, hes_value, size);
		thrust::device_ptr<double> dev_data_ptr(hes_value);
		thrust::device_ptr<thrust::pair<int, int>> dev_keys_ptr(device_pos);
		thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr+size * 3, dev_data_ptr);
		decouple_pos << <1, size*3 >> > (device_pos, pos_x, pos_y, hes_value);
	//	print << <1, 1 >> > (device_pos, hes_value, size * 3);
	//	print << <1, 1 >> > (device_pos,size*3);
		cusparseHandle_t handle;
		cusparseStatus_t status = cusparseCreate(&handle);
		cusparseXcoo2csr(handle, pos_x, 3 * size, size, csr, CUSPARSE_INDEX_BASE_ZERO);
		print << <1, 1 >> > (gra, size);
		delta_x=init_pcg(size,3*size,hes_value,gra,csr,pos_y);
		
		cudaThreadSynchronize();
		add_vec << <1, size >> > (devx, delta_x);
		cudaThreadSynchronize();
		calculate_gra << <1, size >> > (devx, gra, size);
		cudaThreadSynchronize();
		max_gra << <1, size >> > (gra, graMax);
		cudaThreadSynchronize();


		cudaMemcpy(&host_maxGra, graMax, sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		getchar();
	//--	printf("%f\n", host_maxGra);
	}


	calculate_val << <1, size >> > (devx, val, size);
	cudaMemcpy(&hostVal, val, sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	
	cudaMemcpy(x, devx, size* sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	printf("%f",val);
	for (int i = 0; i < size; i++) {
		int next = i + 1;
		if (next == size) next = 0;
		printf("x[%d]*x[%d]=%f %f\n", i, next, x[i] * x[next],x[i]);
	}

}