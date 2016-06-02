#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <x86intrin.h>

#include "timer.c"

#define N_ 4096
#define K_ 4096
#define M_ 4096
const int CBBLOCK = 256;
const int SVBLOCK = 256;

typedef double dtype;

void verify(dtype *C, dtype *C_ans, int N, int M)
{
  int i, cnt;
  cnt = 0;
  for(i = 0; i < N * M; i++) {
    if(abs (C[i] - C_ans[i]) > 1e-6) cnt++;
  }
  if(cnt != 0) printf("ERROR:%d\n",cnt); else printf("SUCCESS\n");
}

void mm_serial (dtype *C, dtype *A, dtype *B, int N, int K, int M)
{
  int i, j, k;
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < M; j++) {
      for(int k = 0; k < K; k++) {
        C[i * M + j] += A[i * K + k] * B[k * M + j];
      }
    }
  }
}

void mm_cb (dtype *C, dtype *A, dtype *B, int N, int K, int M, int Block)
{
  /* =======================================================+ */
  /* Implement your own cache-blocked matrix-matrix multiply  */
  /* =======================================================+*/
  for(int kk = 0; kk < K; kk += Block)
	  for(int ii = 0; ii < N; ii += Block)
		  for(int jj = 0; jj < M; jj += Block)
			  for(int k = kk; k < kk + Block && k < K; k++)
				  for(int i = ii; i < ii + Block && i < N; i++) {
					  double r = A[i * K + k];
					  for(int j=jj; j < jj + Block && j < M; j++)
						  C[i * M + j] += r * B[k * M + j];
				  }
}

void mm_sv (dtype *C, dtype *A, dtype *B, int N, int K, int M, int Block)
{
  /* =======================================================+ */
  /* Implement your own SIMD-vectorized matrix-matrix multiply  */
  /* =======================================================+ */
    for(int kk = 0; kk < K; kk += Block)
	  for(int ii = 0; ii < N; ii += Block)
		  for(int jj = 0; jj < M; jj += Block)
			  for(int k = kk; k < kk + Block && k < K; k++)
				  for(int i = ii; i < ii + Block && i < N; i++) {
					  __m128d a2 = _mm_set1_pd(A[i * K + k]);
					  for(int j=jj; j < jj + Block && j < M; j= j + 2){
						  __m128d c2 = _mm_loadu_pd(&C[i * M + j]);
						  __m128d b2 = _mm_loadu_pd(&B[k * M + j]);
						  c2 = _mm_add_pd(_mm_mul_pd(a2,b2),c2);
						  _mm_storeu_pd(&C[i * M + j], c2);
					  }
				  }
}

int main(int argc, char** argv)
{
  int i, j, k;
  int N, K, M;

  if(argc == 4) {
    N = atoi (argv[1]);		
    K = atoi (argv[2]);		
    M = atoi (argv[3]);		
    printf("N: %d K: %d M: %d\n", N, K, M);
  } else {
    N = N_;
    K = K_;
    M = M_;
    printf("N: %d K: %d M: %d\n", N, K, M);	
  }

  dtype *A = (dtype*) malloc (N * K * sizeof (dtype));
  dtype *B = (dtype*) malloc (K * M * sizeof (dtype));
  dtype *C = (dtype*) malloc (N * M * sizeof (dtype));
  dtype *C_cb = (dtype*) malloc (N * M * sizeof (dtype));
  dtype *C_sv = (dtype*) malloc (N * M * sizeof (dtype));
  assert (A && B && C);

  /* initialize A, B, C */
  srand48 (time (NULL));
  for(i = 0; i < N; i++) {
    for(j = 0; j < K; j++) {
      A[i * K + j] = drand48 ();
    }
  }
  for(i = 0; i < K; i++) {
    for(j = 0; j < M; j++) {
      B[i * M + j] = drand48 ();
    }
  }
  bzero(C, N * M * sizeof (dtype));
  bzero(C_cb, N * M * sizeof (dtype));
  bzero(C_sv, N * M * sizeof (dtype));
  long double work1 = M*N / (1e6);
  long double work2 = (2*K-1)/(1e3);
  long double GFLOP = work1*work2;
  printf("Work:%Lf GB\n",GFLOP);

  stopwatch_init ();
  struct stopwatch_t* timer = stopwatch_create ();
  assert (timer);
  long double t;

  printf("Naive matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  mm_serial (C, A, B, N, K, M);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for naive implementation: %Lg seconds => %Lf GFLOPs\n\n", t,GFLOP/t);


  printf("Cache-blocked matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  mm_cb (C_cb, A, B, N, K, M, CBBLOCK);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for cache-blocked implementation: %Lg seconds => %Lf GFLOPs\n", t, GFLOP/t);

  /* verify answer */
  verify (C_cb, C, N, M);

  printf("\nSIMD-vectorized Cache-blocked matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  mm_sv (C_sv, A, B, N, K, M, SVBLOCK);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for SIMD-vectorized cache-blocked implementation: %Lf seconds => %Lg GFLOPs\n", t, GFLOP/t);

  /* verify answer */
  verify (C_sv, C, N, M);

  return 0;
}
