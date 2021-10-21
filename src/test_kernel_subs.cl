#define __kernel
#define __global
#define __global_id loop_index

__kernel void add_buffer(int loop_index, __global float *a,__global float *b,__global float *c){
    int i = __global_id;
    a[i] = b[i] + c[i];
}