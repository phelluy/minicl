#[macro_use]
extern crate minicl;

fn main() {


    let source = "__kernel  void simple_add(__global int *v, int x){
        int i = get_global_id(0);
        v[i] += x;
    }".to_string();

    let mut cldev = minicl::Accel::new(source);

    let n = 64;

    let v : Vec<i32> = vec![12; n];

    let kname = "simple_add".to_string();
    cldev.register_kernel(&kname);

    let v = cldev.register_buffer(v);

    let x: i32 = 1000;
    
    let globsize = n;
    let locsize= 16;

    kernel_set_args_and_run!(cldev, kname, globsize, locsize, v, x);

    let v: Vec<i32> = cldev.map_buffer(v);
    println!("First kernel run v={:?}",v);

    let v = cldev.unmap_buffer(v);
    cldev.run_kernel(&kname, globsize, locsize);
    let v: Vec<i32> = cldev.map_buffer(v);
    println!("Next kernel run v={:?}",v);

}




