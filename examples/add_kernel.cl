__kernel void simple_add(__global *v){
    int i = get_global_id(0);
    v[i] += 12;
}