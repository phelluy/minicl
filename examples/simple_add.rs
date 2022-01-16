#[macro_use]
extern crate minicl;

fn main() {


    let source = "__kernel  void simple_add(__global int *v, int x){
        int i = get_global_id(0);
        v[i] += x;
    }".to_string();

    let mut cldev = minicl::Accel::new(source);

    // the used kernels has to be registered
    let kname = "simple_add".to_string();
    cldev.register_kernel(&kname);

    let n = 64;

    // the memory buffer shared with the 
    // accelerator has to be registered
    let v : Vec<i32> = vec![12; n];
    let v = cldev.register_buffer(v);

    let x: i32 = 1000;
    
    let globsize = n;
    let locsize= 16;

    // invoke this macro for the first kernel call
    kernel_set_args_and_run!(cldev, kname, globsize, locsize, v, x);

    // map the buffer for access from the host 
    let v: Vec<i32> = cldev.map_buffer(v);
    println!("First kernel run v={:?}",v);

    // unmap for giving it back to the device
    let v = cldev.unmap_buffer(v);

    // next call: no need to redefine the kernel args
    // if they are the same
    cldev.run_kernel(&kname, globsize, locsize);

    let v: Vec<i32> = cldev.map_buffer(v);
    println!("Next kernel run v={:?}",v);

}




