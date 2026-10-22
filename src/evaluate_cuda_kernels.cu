#include "evaluate_cuda_kernels.h"
#include "conjugrad.h"
#include <math_functions.h> // CUDA math functions, used for log/__logf and exp/__expf

#define TILE_DIM 32

#define EDGE_CHUNK 512   // sequences per pass of d_compute_edge_gradients
#define EDGE_THREADS 128
#define PC_SEQS 8        // sequences per block of d_compute_pc (8 x 21 accumulators fill 255 registers)
#define PC_THREADS 128

#define DPC(n,i) d_precompiled[(n) * ncol + (i)]
#define DPCS(n,i) d_precompiled_sum[(n) * ncol + (i)]
#define DPCN(n,a,i) d_precompiled_norm[((n) * N_ALPHA_PAD + a) * ncol + (i)]

#define DPCNS(n,z) d_precompiled_norm[(n) * (ncol * N_ALPHA_PAD) +(z)]

// forward declaration of device functions
__device__ void sum_reduction_function2(volatile conjugrad_float_t *s_data, int tid);
__device__ void warp_reduction2(volatile conjugrad_float_t *s_data, int tid);

/** Kernel for the computation of the precompiled values
 * @param[in] d_msa The multiple sequence alignment (MSA)
 * @param[in] d_x1 The node parameters
 * @param[in] d_x2 The edge parameters
 * @param[in] d_precompiled[i,a,s] = V(a,s) + sum_k W(xik,k,a,s)
 * @param[in] d_precompiled_sum[i,s] = log(sum_a exp(d_precompiled[i,a,s]))
 * @param[in] d_precompiled_norm[i,a,s] = 
 * @param[in] d_weights The sequence weights
 * @param[in] ncol The number of columns of the MSA
 * @param[in] nrow The number of sequences in the MSA
 */
__global__ void d_compute_pc(
	const unsigned char *d_msa,
	const conjugrad_float_t *d_x1,
	const conjugrad_float_t *d_x2,
	conjugrad_float_t *d_precompiled,
	conjugrad_float_t *d_precompiled_sum,
	conjugrad_float_t *d_precompiled_norm,
	const conjugrad_float_t *d_weights,
	const int ncol,
	const int nrow) {

	// a block holds PC_SEQS sequences and blockDim.x columns s; each thread keeps
	// PC(i,a,s) of all PC_SEQS sequences in registers, so a coupling slice
	// W(b,k,.,s) shared by several of them (same residue b at column k) is read
	// from L1 after the first sequence rather than from L2 for each one
	extern __shared__ unsigned char s_msa[];   // [PC_SEQS][ncol]
	const int i0 = blockIdx.x * PC_SEQS;
	const int s = blockIdx.y * blockDim.x + threadIdx.x;

	for (int z = threadIdx.x; z < PC_SEQS * ncol; z += blockDim.x) {
		int q = z / ncol;
		s_msa[z] = i0 + q < nrow ? d_msa[(i0 + q) * ncol + z % ncol] : 0;
	}
	__syncthreads();
	if (s >= ncol) {
		return;
	}

	// initialize with single potentials, gap at 0
	conjugrad_float_t pc[PC_SEQS][N_ALPHA];
	#pragma unroll
	for (int a = 0; a < N_ALPHA - 1; a++) {
		conjugrad_float_t v = d_x1[a * ncol + s];
		#pragma unroll
		for (int q = 0; q < PC_SEQS; q++) {
			pc[q][a] = v;
		}
	}
	#pragma unroll
	for (int q = 0; q < PC_SEQS; q++) {
		pc[q][N_ALPHA - 1] = F0; // 0.0
	}

	// add sum_k w_sk(a, x_ik)
	for (int k = 0; k < ncol; k++) {
		#pragma unroll
		for (int q = 0; q < PC_SEQS; q++) {
			const conjugrad_float_t *w = &d_x2[(s_msa[q * ncol + k] * ncol + k) * N_ALPHA_PAD * ncol + s];
			#pragma unroll
			for (int a = 0; a < N_ALPHA; a++) {
				pc[q][a] += w[a * ncol];
			}
		}
	}

	#pragma unroll
	for (int q = 0; q < PC_SEQS; q++) {
		const int i = i0 + q;
		if (i >= nrow) {
			break;
		}
		const conjugrad_float_t weight = d_weights[i];

		// log(sum_a exp(PC(i,a,s)))
		conjugrad_float_t acc = F0; // 0.0
		#pragma unroll
		for (int a = 0; a < N_ALPHA; a++) {
			acc += fdexp(pc[q][a]);
		}
		acc = fdlog(acc);
		DPCS(i,s) = weight * acc;

		#pragma unroll
		for (int a = 0; a < N_ALPHA; a++) {
			DPCN(i,a,s) = weight * fdexp(pc[q][a] - acc);
		}

		// select rather than pc[q][x]: a runtime index would move pc out of registers
		const int xis = s_msa[q * ncol + s];
		conjugrad_float_t pcx = F0;
		#pragma unroll
		for (int a = 0; a < N_ALPHA; a++) {
			pcx = a == xis ? pc[q][a] : pcx;
		}
		DPC(i,s) = weight * pcx;
	}
}


/** Kernel for the function value. Kept completely independent from the length/height of the MSA etc.
  * blockDim.x has to be a power of two
  * Try with 256 threads, 128 blocks, 256 * sizeof(conjugrad_float_t) Bytes shared mem
  */
__global__ void d_compute_fx_inter(
	const conjugrad_float_t *d_precompiled,
	const conjugrad_float_t *d_precompiled_sum,
	const int nvar,
	conjugrad_float_t *d_inter_fx
) {
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ conjugrad_float_t s_inter_fx[];

	s_inter_fx[tid] = F0; // 0.0

	// load variables into shared mem
	while (i < nvar) {
		s_inter_fx[tid] += d_precompiled_sum[i] - d_precompiled[i];
		i += blockDim.x * gridDim.x;
	}
	__syncthreads();

	// start reduction
	sum_reduction_function2(s_inter_fx, tid);

	// write intermediate result back to global mem
	if (0 == tid) {
		d_inter_fx[blockIdx.x] = s_inter_fx[0];
	}
}

/** Kernel for the intermediate result for REG 
 * @param[in] d_x
 * @param[in] d_g
 * @param[in] d_lambda
 * @param[in] d_inter_reg
 * @param[in] nvar
 */
__global__ void d_compute_reg_inter_nodes(
	conjugrad_float_t *d_g,
	conjugrad_float_t lambda,
	conjugrad_float_t *d_inter_reg,
	conjugrad_float_t *d_histograms,
	int nvar
) {
	int ti = threadIdx.x;
	int i = blockIdx.x * blockDim.x + ti;

	extern __shared__ conjugrad_float_t s_inter_reg[];

	s_inter_reg[ti] = F0; // 0.0

	while (i < nvar) {
		conjugrad_float_t x_i = d_g[i];
		conjugrad_float_t hist_i = - (conjugrad_float_t)d_histograms[i];

		// add to shared mem
		s_inter_reg[ti] += lambda * x_i * x_i; 

		// update g
		d_g[i] = hist_i + F2 * lambda * x_i; // F2 is 2.0

		i += blockDim.x * gridDim.x;
	}
	__syncthreads();

	sum_reduction_function2(s_inter_reg, ti);

	if (0 == ti) {
		d_inter_reg[blockIdx.x] = s_inter_reg[0];
	}
}

__global__ void d_compute_reg_inter_edges(
	conjugrad_float_t *d_g,
	int nsingle,
	conjugrad_float_t lambda,
	conjugrad_float_t *d_inter_reg,
	conjugrad_float_t *d_histograms,
	int nvar
) {
	int ti = threadIdx.x;
	int i = blockIdx.x * blockDim.x + ti;

	extern __shared__ conjugrad_float_t s_inter_reg[];

	s_inter_reg[ti] = F0; // 0.0

	while (i < nvar) {
		conjugrad_float_t x_i = d_g[i];
		conjugrad_float_t hist_i = - (conjugrad_float_t)d_histograms[i];

		// add to shared mem
		s_inter_reg[ti] += F05 * lambda * x_i * x_i; // F05 is 0.5

		// update g
		d_g[i] = hist_i + lambda * x_i;

		i += blockDim.x * gridDim.x;
	}
	__syncthreads();

	sum_reduction_function2(s_inter_reg, ti);

	if (0 == ti) {
		d_inter_reg[blockIdx.x] = s_inter_reg[0];
	}
}


__global__ void sum_reduction2(
	conjugrad_float_t *d_in,
	conjugrad_float_t *d_out,
	int nvar
) {

	int ti = threadIdx.x;
	int i = threadIdx.x; // equals threadIdx.x since we only have one block

	extern __shared__ conjugrad_float_t s_tmp[];

	s_tmp[ti] = F0; // 0.0

	while (i < nvar) {
		s_tmp[ti] += d_in[i];
		i += blockDim.x; // we will only start one block
	}
	__syncthreads();

	sum_reduction_function2(s_tmp, ti);

	if (0 == ti) {
		d_out[0] = s_tmp[0];
	}
}



/* Kernel for node gradients
 * @param[in] d_g1
 * @param[in] d_precompiled_norm
 * @param[in] ncol
 * @param[in] nrow
 * @param[in] nvar = (N_ALPHA - 1) * ncol
 */
__global__ void d_compute_node_gradients(
	conjugrad_float_t *d_g1,
	conjugrad_float_t *d_precompiled_norm,
	int ncol,
	int nrow,
	const unsigned int nvar //ncol * (N_ALPHA - 1)
) {
	unsigned int z = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int offset = blockDim.x * gridDim.x;

	while (z < nvar) {
		conjugrad_float_t local_g = F0; // 0.0

		for (int i = 0; i < nrow; i++) {
			local_g += DPCNS(i,z);
		}
		d_g1[z] += local_g;

		z += offset;
	}
}

__global__ void d_compute_edge_gradients(
	unsigned char *d_msa_transposed,
	conjugrad_float_t *d_precompiled_norm,
	conjugrad_float_t *d_g2,
	int nvar, // N_ALPHA * ncol
	int ncol,
	int nrow
) {
	// d_g2[b,k,aj] += sum over sequences i with x_ik = b of PCN(i,aj).  A block
	// covers column k and blockDim.x consecutive aj; each thread sums its own aj
	// in shared memory (s_acc[b][tid]: no atomics, no bank conflicts) and writes
	// d_g2 once, instead of reading and writing d_g2 once per sequence
	extern __shared__ conjugrad_float_t s_acc[];   // [N_ALPHA][blockDim.x]
	unsigned char *s_xk = (unsigned char *)&s_acc[N_ALPHA * blockDim.x]; // [EDGE_CHUNK]

	const int tid = threadIdx.x;
	const int k = blockIdx.y;
	const int aj = blockIdx.x * blockDim.x + tid;

	for (int b = 0; b < N_ALPHA; b++) {
		s_acc[b * blockDim.x + tid] = F0; // 0.0
	}

	for (int i0 = 0; i0 < nrow; i0 += EDGE_CHUNK) {
		const int n = nrow - i0 < EDGE_CHUNK ? nrow - i0 : EDGE_CHUNK;
		__syncthreads();
		for (int z = tid; z < n; z += blockDim.x) {
			s_xk[z] = d_msa_transposed[k * nrow + i0 + z];
		}
		__syncthreads();
		if (aj < nvar) {
			for (int z = 0; z < n; z++) {
				s_acc[s_xk[z] * blockDim.x + tid] += DPCNS(i0 + z, aj);
			}
		}
	}

	if (aj < nvar) {
		for (int b = 0; b < N_ALPHA; b++) {
			d_g2[(b * ncol + k) * N_ALPHA_PAD * ncol + aj] += s_acc[b * blockDim.x + tid];
		}
	}
}


__global__ void d_node_gradients_histogram_weighted(
	unsigned char *d_msa,
	conjugrad_float_t *d_node_histogram,
	const conjugrad_float_t *d_weights,
	int ncol,
	int nrow
) {

	int col = blockIdx.x * blockDim.x + threadIdx.x; 
	
	extern __shared__ conjugrad_float_t s_node_hist_weighted[];
	
	while (col < ncol) {
		// initialize shared mem
		#pragma unroll 21
		for (int a = 0; a < N_ALPHA; a++) {
			s_node_hist_weighted[a * blockDim.x + threadIdx.x] = F0; // 0.0
		}

		// count
		for (int i = 0; i < nrow; i++) {
			//s_node_hist[d_msa[i * ncol + col] * blockDim.x + threadIdx.x]++;
			s_node_hist_weighted[d_msa[i * ncol + col] * blockDim.x + threadIdx.x] += d_weights[i];
		}

		// write to histogram, ignore gaps
		#pragma unroll 20
		for (int a = 0; a < (N_ALPHA - 1); a++) {
			d_node_histogram[a * ncol + col] = s_node_hist_weighted[a * blockDim.x + threadIdx.x]; 
		}
		col += blockDim.x * gridDim.x;
	}
}


__global__ void d_edge_gradients_histogram_weighted(
	unsigned char *d_msa,
	conjugrad_float_t *d_edge_histogram,
	const conjugrad_float_t *d_weights,
	int ncol,
	int nrow
) {

	int k = blockIdx.x;
	int b = blockIdx.y;
	int j = threadIdx.x;

	extern __shared__ conjugrad_float_t s_edge_hist_weighted[];

	while (j < ncol) {
		// initialize shared mem
		#pragma unroll 21
		for (int a = 0; a < N_ALPHA; a++) {
			s_edge_hist_weighted[a * blockDim.x + threadIdx.x] = F0; // 0.0
		}

		for (int i = 0; i < nrow; i++) {
			unsigned char xik = d_msa[i * ncol + k]; //bad, cache this somehow
			unsigned char xij = d_msa[i * ncol + j];

			// TODO: can we get around this somehow???
			if (xik == b) {
				s_edge_hist_weighted[xij * blockDim.x + threadIdx.x] += d_weights[i];
			}
		}
		
		#pragma unroll 21
		for (int a = 0; a < N_ALPHA; a++) {
			d_edge_histogram[((b * ncol + k) * N_ALPHA_PAD + a) * ncol + j] = s_edge_hist_weighted[a * blockDim.x + threadIdx.x];
		}
		j += blockDim.x;
	}
}


__global__ void d_add_transposed_g2(
	conjugrad_float_t *d_g2,
	int dim,
	int realDim) {

	extern __shared__ conjugrad_float_t s_g2[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// ignore diagonal, will be set to 0 at the end of evaluate anyway
	if (x < dim && y < x) {
		s_g2[threadIdx.y * TILE_DIM + threadIdx.x] = d_g2[y * realDim + x];
	}
	__syncthreads();

	y = blockIdx.x * blockDim.x + threadIdx.y;
	x = blockIdx.y * blockDim.y + threadIdx.x;

	if (y < dim && x < y) {
		conjugrad_float_t tmp = d_g2[y * realDim + x];
		d_g2[y * realDim + x] = tmp + s_g2[threadIdx.x * TILE_DIM + threadIdx.y];
		s_g2[threadIdx.x * TILE_DIM + threadIdx.y] = tmp;
	}
	__syncthreads();

	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < dim && y <= x) {
		d_g2[y * realDim + x] += s_g2[threadIdx.y * TILE_DIM + threadIdx.x];
	}
}



__global__ void d_transpose_msa(
	unsigned char *d_msa,
	unsigned char *d_msa_transposed,
	int ncol,
	int nrow
) {

	extern __shared__ unsigned char s_msa_tile[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < ncol && y < nrow) {
		s_msa_tile[threadIdx.y * blockDim.y + threadIdx.x] = d_msa[y * ncol + x];
	}

	__syncthreads();

	y = blockIdx.x * blockDim.x + threadIdx.y;
	x = blockIdx.y * blockDim.y + threadIdx.x;

	if (y < ncol && x < nrow) {
		d_msa_transposed[y * nrow + x] = s_msa_tile[threadIdx.x * blockDim.y + threadIdx.y]; 
	}
}


__global__ void d_reset_edge_gradients(
	conjugrad_float_t *d_g2,
	int ncol
) {
	int j = blockIdx.x;
	int b = threadIdx.x;
	int a = threadIdx.y;
	d_g2[((a * ncol + j) * N_ALPHA_PAD + b) * ncol + j] = F0; // 0.0
}


__global__ void d_initialize_w(
	conjugrad_float_t *d_weights,
	int idthres,
	const unsigned char *d_msa,
	const int nrow,
	const int ncol
) {
	int n = blockIdx.x; // aka first sequence
	extern __shared__ conjugrad_float_t s_ids[];
	while (n < nrow) {
		int i = threadIdx.x; // aka column
		for (int m = 0; m < nrow; m++) { // aka second sequence
			s_ids[threadIdx.x] = F0; // 0.0

			// count identities for n and m
			while (i < ncol) {
				if (d_msa[n * ncol + i] == d_msa[m * ncol + i]) {
					s_ids[threadIdx.x] += F1; // 1.0
				}
				i += blockDim.x;
			}

			// reduce shared array
			__syncthreads();
			sum_reduction_function2(s_ids, threadIdx.x);

			// increase weight
			if (0 == threadIdx.x) {
				if (s_ids[0] > idthres) {
					d_weights[n] += F1; // 1.0
				}
			}
			__syncthreads();
			
			i = threadIdx.x;
		}
		// ignore n = m
		if (0 == threadIdx.x) {
			d_weights[n] -= F1; // 1.0
		}
		__syncthreads();
		n += gridDim.x;
	}
}


__global__ void d_update_w(
	conjugrad_float_t *d_weights,
	const int nrow
) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int offset = blockDim.x * gridDim.x;
	while (i < nrow) {
		d_weights[i] = F1 / (F1 + d_weights[i]); // F1 is 1.0
		i += offset;
	}
}


// Device functions
__device__ void sum_reduction_function2(
	volatile conjugrad_float_t *s_data,
	int tid
) {

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			s_data[tid] += s_data[tid + s];
		}
		__syncthreads();
	}

	if (tid < 32) {
		warp_reduction2(s_data, tid);
	}
}

__device__ void warp_reduction2(
	volatile conjugrad_float_t *s_data,
	int tid
) {
	// explicit warp sync: required with independent thread scheduling (Volta+)
	conjugrad_float_t v = s_data[tid];
	v += s_data[tid + 32]; __syncwarp(); s_data[tid] = v; __syncwarp();
	v += s_data[tid + 16]; __syncwarp(); s_data[tid] = v; __syncwarp();
	v += s_data[tid + 8]; __syncwarp(); s_data[tid] = v; __syncwarp();
	v += s_data[tid + 4]; __syncwarp(); s_data[tid] = v; __syncwarp();
	v += s_data[tid + 2]; __syncwarp(); s_data[tid] = v; __syncwarp();
	v += s_data[tid + 1]; __syncwarp(); s_data[tid] = v; __syncwarp();
}


// wrapper functions
#ifdef __cplusplus
extern "C" {
#endif

void gpu_pc(
	const unsigned char *d_msa,
	const conjugrad_float_t *d_x1,
	const conjugrad_float_t *d_x2,
	conjugrad_float_t *d_precompiled,
	conjugrad_float_t *d_precompiled_sum,
	conjugrad_float_t *d_precompiled_norm,
	const conjugrad_float_t *d_weights,
	const int ncol,
	const int nrow
) {

	dim3 nblocks((nrow + PC_SEQS - 1) / PC_SEQS, (ncol + PC_THREADS - 1) / PC_THREADS, 1);
	size_t nbytes = PC_SEQS * ncol;

	d_compute_pc<<<nblocks, PC_THREADS, nbytes>>>(
		d_msa,
		d_x1,
		d_x2,
		d_precompiled,
		d_precompiled_sum,
		d_precompiled_norm,
		d_weights,
		ncol,
		nrow
	);

	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());
}

void gpu_compute_fx(
	const conjugrad_float_t *d_precompiled,
	const conjugrad_float_t *d_precompiled_sum,
	const int ncol,
	const int nrow,
	conjugrad_float_t *d_fx
) {

	const unsigned int nblocks = 128;
	const unsigned int nthreads = 256;
	size_t nbytes = nthreads * sizeof(conjugrad_float_t);

	const int nvar = ncol * nrow;

	// allocate memory for intermediate results
	conjugrad_float_t *d_fx_inter;
	CHECK_ERR(cudaMalloc((void **) &d_fx_inter, sizeof(conjugrad_float_t) * nblocks));


	d_compute_fx_inter<<<nblocks, nthreads, nbytes>>>(d_precompiled, d_precompiled_sum, nvar, d_fx_inter);

	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());


	sum_reduction2<<<1, nblocks, sizeof(conjugrad_float_t) * nblocks>>>(d_fx_inter, d_fx, nblocks);

	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());

	CHECK_ERR(cudaFree(d_fx_inter));
}

void gpu_compute_node_gradients(
	conjugrad_float_t *d_g1,
	conjugrad_float_t *d_precompiled_norm,
	int ncol,
	int nrow
) {

	const unsigned int nvar = ncol * (N_ALPHA - 1);

	const unsigned int nthreads = 256;
	const unsigned int nblocks = ceil((float)nvar / (float)nthreads);

	d_compute_node_gradients<<<nblocks, nthreads>>>(d_g1, d_precompiled_norm, ncol, nrow, nvar);

	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());
}

void gpu_compute_edge_gradients(
	unsigned char *d_msa_transposed,
	conjugrad_float_t *d_g2,
	conjugrad_float_t *d_precompiled_norm,
	int ncol,
	int nrow
) {

	dim3 nblocks((N_ALPHA * ncol + EDGE_THREADS - 1) / EDGE_THREADS, ncol, 1);
	size_t nbytes = sizeof(conjugrad_float_t) * N_ALPHA * EDGE_THREADS + EDGE_CHUNK;

	d_compute_edge_gradients<<<nblocks, EDGE_THREADS, nbytes>>>(
		d_msa_transposed,
		d_precompiled_norm,
		d_g2,
		(N_ALPHA * ncol),
		ncol,
		nrow
	);

	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());


	int dim = ncol * N_ALPHA;
	int realDim = ncol * N_ALPHA_PAD;
	int border = ceil((float)dim / 32.0);
	dim3 blocks(border,border,1);
	dim3 threads(32,32,1);

	d_add_transposed_g2<<<blocks, threads, 1056 * sizeof(conjugrad_float_t)>>>(
		d_g2,
		dim,
		realDim
	);

	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());


	dim3 nthreads(N_ALPHA,N_ALPHA,1);
	d_reset_edge_gradients<<<ncol, nthreads>>>(d_g2, ncol);

	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());
}

void gpu_compute_regularization(
	conjugrad_float_t *d_g,
	conjugrad_float_t lambda_nodes,
	conjugrad_float_t lambda_edges,
	conjugrad_float_t *d_reg,
	conjugrad_float_t *d_histograms,
	int nvar,
	int nsingle
) {

	const unsigned int nblocks = 128;
	const unsigned int nthreads = 256;
	const unsigned int nbytes = nthreads * sizeof(conjugrad_float_t);

	// allocate mem for intermediate results
	conjugrad_float_t *d_reg_inter;
	CHECK_ERR(cudaMalloc((void **) &d_reg_inter, sizeof(conjugrad_float_t) * nblocks));


	conjugrad_float_t *d_g1 = d_g;
	conjugrad_float_t *d_g2 = &d_g[nsingle];

	conjugrad_float_t *d_node_histogram = d_histograms;
	conjugrad_float_t *d_edge_histogram = &d_histograms[nsingle];

	conjugrad_float_t *d_reg_inter_nodes = d_reg_inter;
	conjugrad_float_t *d_reg_inter_edges = &d_reg_inter[1];


	d_compute_reg_inter_nodes<<<1, nthreads, nbytes>>>(
		d_g1,
		lambda_nodes,
		d_reg_inter_nodes,
		d_node_histogram,
		nsingle
	);

	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());

	// reg edge gradients
	d_compute_reg_inter_edges<<<(nblocks - 1), nthreads, nbytes>>>(
		d_g2,
		nsingle,
		lambda_edges,
		d_reg_inter_edges,
		d_edge_histogram,
		(nvar - nsingle)
	);

	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());


	sum_reduction2<<<1, nblocks, sizeof(conjugrad_float_t) * nblocks>>>(d_reg_inter, d_reg, nblocks);

	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());

	CHECK_ERR(cudaFree(d_reg_inter));
}


void gpu_tranpose_msa(
	unsigned char *d_msa,
	unsigned char *d_msa_transposed,
	int ncol,
	int nrow
) {

	const unsigned int blocks_col = ceil((float)ncol / 32.0f);
	const unsigned int blocks_row = ceil((float)nrow / 32.0f);
	const dim3 nblocks(blocks_col, blocks_row, 1);
	const dim3 nthreads(32,32,1);
	const size_t nbytes = sizeof(unsigned char) * 1024;

	d_transpose_msa<<<nblocks, nthreads, nbytes>>>(d_msa, d_msa_transposed, ncol, nrow);

	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());
}


void gpu_initialize_histograms_weighted(
	unsigned char *d_msa,
	conjugrad_float_t *d_histograms,
	const conjugrad_float_t *d_weights,
	int ncol,
	int nrow
) {

	int nsingle = ncol * (N_ALPHA - 1);
	int new_nsingle = nsingle + N_ALPHA_PAD - (nsingle % N_ALPHA_PAD);
	conjugrad_float_t *d_node_histogram = d_histograms;
	conjugrad_float_t *d_edge_histogram = &d_histograms[new_nsingle];

	int nthreads = 128;
	int nblocks = ceil((float)ncol / (float)nthreads);
	int nbytes = nthreads * N_ALPHA * sizeof(conjugrad_float_t);

	d_node_gradients_histogram_weighted<<<nblocks, nthreads, nbytes>>>(d_msa, d_node_histogram, d_weights, ncol, nrow);

	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());

	dim3 nblocks_dim3(ncol, N_ALPHA, 1);
	d_edge_gradients_histogram_weighted<<<nblocks_dim3, nthreads, nbytes>>>(d_msa, d_edge_histogram, d_weights, ncol, nrow);

	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());
}


void gpu_compute_weights_simple(
	conjugrad_float_t *d_weights,
	conjugrad_float_t threshold,
	unsigned char *d_msa,
	int nrow,
	int ncol
) {

	int idthres = (int)ceil(threshold * (conjugrad_float_t)ncol);
	unsigned int nblocks = nrow < 65535 ? nrow : 65535;
	unsigned int nthreads = 256;
	size_t nbytes = sizeof(conjugrad_float_t) * nthreads;

	d_initialize_w<<<nblocks, nthreads, nbytes>>>(d_weights, idthres, d_msa, nrow, ncol);
	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());

	nblocks = 128;
	d_update_w<<<nblocks, nthreads>>>(d_weights, nrow);
	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());
}


#ifdef __cplusplus
}
#endif
