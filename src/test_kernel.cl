__kernel void add_buffer(__global float *a,__global float *b,__global float *c){
    size_t i = get_global_id(0);
    a[i] = b[i] + c[i];
}