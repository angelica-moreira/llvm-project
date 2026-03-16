// bubble_sort.c -- Bubble sort on a large array for profiling.
//
// The hot inner loop generates many branch events that show up clearly
// in ETW SampledProfile data.  The comparison branch is taken roughly
// half the time on random data, making it a good target for block
// reordering by BOLT.
//
// Build:  clang-cl /O2 bubble_sort.c -o bubble_sort.exe
// Profile: xperf -on PROC_THREAD+LOADER+PROFILE && bubble_sort.exe && xperf -d trace.etl

#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 30000

static void bubble_sort(int *arr, int n) {
  for (int i = 0; i < n - 1; i++) {
    int swapped = 0;
    for (int j = 0; j < n - i - 1; j++) {
      if (arr[j] > arr[j + 1]) {
        int tmp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = tmp;
        swapped = 1;
      }
    }
    if (!swapped)
      break;
  }
}

int main(void) {
  int *arr = (int *)malloc(ARRAY_SIZE * sizeof(int));
  if (!arr) {
    fprintf(stderr, "out of memory\n");
    return 1;
  }

  // Fill with pseudo-random data so the sort actually does work.
  unsigned seed = 12345;
  for (int i = 0; i < ARRAY_SIZE; i++) {
    seed = seed * 1103515245 + 12345;
    arr[i] = (int)(seed >> 16) & 0x7fff;
  }

  bubble_sort(arr, ARRAY_SIZE);

  // Quick sanity check.
  for (int i = 1; i < ARRAY_SIZE; i++) {
    if (arr[i] < arr[i - 1]) {
      fprintf(stderr, "sort failed at index %d\n", i);
      free(arr);
      return 1;
    }
  }

  printf("sorted %d elements\n", ARRAY_SIZE);
  free(arr);
  return 0;
}
