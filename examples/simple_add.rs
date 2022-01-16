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
    
    cldev.set_kernel_arg(&kname, 0, &v);
    cldev.set_kernel_arg(&kname, 1, &x);

    let globsize = n;
    let locsize= 16;
 
    cldev.run_kernel(&kname,globsize,locsize);
    let v: Vec<i32> = cldev.map_buffer(v);
    println!("v={:?}",v);
    let v = cldev.unmap_buffer(v);
    cldev.run_kernel(&kname,globsize,locsize);
    let v: Vec<i32> = cldev.map_buffer(v);
    println!("v={:?}",v);


}




