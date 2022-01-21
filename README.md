# minicl
Minimal compute library for accessing OpenCL from Rust.

With normal use, this library should prevent memory leaks
 and most unsafety related to OpenCL.

 All the OpenCL things (device, context, buffers, etc.)
are packed into a single Accelerator struct.

Tested (January 2022) on:

- Linux, with NVIDIA drivers (GPU), Intel drivers (CPU), POCL drivers (CPU) and Oclgrind debugger;

- Apple M1 Silicon with Apple drivers (GPU);

- Windows 11 in WSL2 with POCL drivers (CPU).

Simple example:

 ```rust
let source = "__kernel  void simple_add(__global int *v, int x){
int i = get_global_id(0);
v[i] += x;
}"
.to_string();
let num_platform = 0;
let mut cldev = minicl::Accel::new(source, num_platform);

// the used kernel has to be registered
let kname = "simple_add".to_string();
cldev.register_kernel(&kname);

let n = 64;

// the memory buffer shared with the
// accelerator has to be registered
let v: Vec<i32> = vec![12; n];
let v = cldev.register_buffer(v);

let x: i32 = 1000;
let globsize = n;
let locsize = 16;

// first kernel call
minicl::kernel_set_args_and_run!(cldev, kname, globsize, locsize, v, x);

// map the buffer for access from the host
let v: Vec<i32> = cldev.map_buffer(v);
println!("First kernel run v={:?}", v);

// unmap for giving it back to the device
let v = cldev.unmap_buffer(v);

// next call
minicl::kernel_set_args_and_run!(cldev, kname, globsize, locsize, v, x);

let v: Vec<i32> = cldev.map_buffer(v);
println!("Next kernel run v={:?}", v);
 ```