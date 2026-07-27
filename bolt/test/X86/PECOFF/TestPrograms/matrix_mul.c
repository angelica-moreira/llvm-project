// matrix_mul.c -- Matrix multiplication for profiling.
//
// Build:  clang-cl /O2 matrix_mul.c -o matrix_mul.exe

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 256

static int A[N][N];
static int B[N][N];
static int C[N][N];

static void fill_matrix(int m[N][N], unsigned seed) {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      seed = seed * 1103515245 + 12345;
      m[i][j] = (int)((seed >> 16) & 0xff);
    }
}

static void multiply(void) {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < N; i++)
    for (int k = 0; k < N; k++)
      for (int j = 0; j < N; j++)
        C[i][j] += A[i][k] * B[k][j];
}

static int verify(void) {
  if (C[0][0] == 0) {
    fprintf(stderr, "verification failed: C[0][0] is zero\n");
    return 1;
  }
  return 0;
}

int main(void) {
  fill_matrix(A, 42);
  fill_matrix(B, 97);

  // Run multiply several times to generate enough samples.
  for (int iter = 0; iter < 5; iter++)
    multiply();

  if (verify()) return 1;

  printf("matrix multiply %dx%d done\n", N, N);
  return 0;
}
