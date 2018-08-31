#ifndef __BASE_H__
#define __BASE_H__
#define USE_CUDA 0

#include <cmath>
#include "base_structs.h"

inline bool Equal(double a, double b)
{
	a = abs(a); b = abs(b);
	double bigger = (a> b)?a: b;
	if (bigger < 1e-8) return true;
	double diff = a-b;
	if (diff < 0) diff *=-1;
	if (diff < 0.0000001*bigger || diff < 0.0000001) return true;

	if (std::isnan((double) a) && std::isnan((double) b)) return true;
	if (std::isinf(a) && std::isinf(b)) return true;
	return false;
}

bool CheckResults(ArrayD &a, ArrayD &b)
{
	if (a.size() != b.size()) {printf("Array sizes don't match\n"); return false;}
	int count = 0;
	ArrayD aa, ba;
	ArrayBaseD ac=a, bc=b;
	if (a._onGPU) {aa.clone(a,false,false); ac=aa;}
	if (b._onGPU) {ba.clone(b,false,false); bc=ba;}
	for (int i = 0 ; i < ac.size(); i++)
		if (!Equal(ac[i], bc[i]))
		{
			printf("Chyba na %d mezi %e %e dela %f\n", i, ac[i], bc[i], abs(ac[i]-bc[i])/std::max(ac[i], bc[i]));
			if (count++ > 20) return false;
		}
	if (count==0) printf("Shodne\n");
	return count==0;
}

template <class MatrixType> void Mult(const MatrixType &A, const double * vec, double *out)
{
	//if (vec.size()< A.num_cols()) throw "CSRMatrix::mult - vec vector not big enough";

	int N = A.num_rows();
	#pragma omp parallel for schedule(static)
	for (int ri = 0; ri < N; ri++)
	{
		out[ri] = 0;
		for (int i = A.num_in_row(ri)-1; i>=0; i--)
		{
			double val=0; int col=0;
			A.get_el_in_row(ri,i,val,col);
			out[ri] += val * vec[col];
		}
	}
}

template <class MatrixType>
__cuda_call__ void JacobiIterKernel(const MatrixType &A, const ArrayBaseD &b, const ArrayBaseD & x, ArrayBaseD & out_x, const double damping, int r)
{
	double diag = 1;
	double nonDiag = 0;

	for (int i = 0; i < A.num_in_row(r); i++)
	{
		double aVal; int c;
		A.get_el_in_row(r,i,aVal,c);

		if (c==r) diag = aVal;
		else nonDiag += aVal * x[c];
	}
	out_x[r] = (1.0 - damping)*x[r] + damping*(b[r] - nonDiag)/diag;
}

#if USE_CUDA
template <class MatrixType>
__global__ void JacobiIterGPU(const MatrixType A, const ArrayBaseD b, const ArrayBaseD x, ArrayBaseD out_x, const double damping)
{
  int r = blockIdx.x*blockDim.x + threadIdx.x;
  if (r < b.size())
	JacobiIterKernel(A, b, x, out_x, damping, r);
}

template <class MatrixType> __global__ void ResiduumGPU(const MatrixType A, const ArrayBaseD b, const ArrayBaseD x, double * result)
{
	__shared__ float sdata[blockSize];
	const unsigned int tid = threadIdx.x;
	sdata[tid] = 0;
	int r = blockIdx.x*blockDim.x + threadIdx.x;
	if (r >= A.num_rows()) return;
	sdata[tid] = ResiduumKernel(A, b, x, r);
	__syncthreads();
	for( unsigned int s = blockDim.x/2 ; s > 0 ; s >>= 1 )
	{
		if( tid < s ) sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if( tid == 0 ) result[blockIdx.x] = sdata[0];
	//if( tid == 0 ) atomicAdd(result, sdata[0]); //Doesn't work
}
#endif

template <class MatrixType> void JacobiIter(const MatrixType &A, const ArrayD &b, const ArrayD & x, ArrayD & out_x, const double damping=1)
{
	const int n = A.num_rows();
	assert(A.num_cols() == n) ;
	assert(b.size() == n);
	assert(x.size() >= n);
	assert(out_x.size() >= n);
	assert(b._onGPU == x._onGPU && x._onGPU == out_x._onGPU);
#if USE_CUDA
	if (x._onGPU)
	{
		JacobiIterGPU <<<  gridSize(n), blockSize >>> (A.toKernel(), b, x, out_x, damping);
	}
	else
#endif
	{
#pragma omp parallel for schedule(static)
		for (int r = 0; r < n; r++)
			JacobiIterKernel(A, b, x, out_x, damping, r);
	}
}


template <class MatrixType> __cuda_call__ double ResiduumKernel(const MatrixType &A, const ArrayBaseD & b, const ArrayBaseD & x, int r)
{
	double res = 0;
	for (int i = 0; i < A.num_in_row(r); i++)
	{
		double aVal = 0; int c = r;
		A.get_el_in_row(r,i, aVal, c);
		res += aVal*x[c];
	}
	res = b[r] - res;
	return res*res;
}

template <class MatrixType> double Residuum(const MatrixType &A, const ArrayD & b, const ArrayD & x)
{
	const int n = A.num_rows();
	assert(A.num_cols() == n) ;
	assert(b.size() == n);
	assert(x.size() >= n);
	assert(b._onGPU == x._onGPU);
	double res = 0;
#if USE_CUDA
	if (x._onGPU)
	{
	#if 0
		static double *resGPU = 0;
		if (resGPU==0) cudaMalloc(&resGPU, sizeof(double));
		cudaMemset(resGPU, 0, sizeof(double));
		ResiduumGPU<MatrixType> <<<  gridSize(n), blockSize >>> (A, b, x, resGPU);
		cudaDeviceSynchronize();
		cudaMemcpy(&res, resGPU, sizeof(double), cudaMemcpyDeviceToHost);
	#else
		ArrayD resids(gridSize(n), true);
		resids.fill(0);
		ResiduumGPU <<<  gridSize(n), blockSize >>> (A.toKernel(), b, x, resids.data);
		resids.moveToCPU();
		for (int i = 0; i < resids.size(); i++) res+=resids[i];
	#endif
	}
	else
#endif //USE_CUDA
	{
#pragma omp parallel for reduction(+:res) schedule(static)
		for (int r = 0; r < n; r++)
			res+=ResiduumKernel(A, b, x, r);
	}
	return sqrt(res);
}
#endif
