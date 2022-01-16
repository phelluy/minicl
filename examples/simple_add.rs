#[macro_use]
extern crate minicl;

fn main() {


    let source = "__kernel  void simple_add(__global int *v, int x){
        int i = get_global_id(0);
        v[i] += x;
    }".to_string();

    let mut cldev = minicl::Accel::new(source);

    let v : Vec<i32> = vec![12; 10];

    let kname = "simple_add".to_string();
    cldev.register_kernel(&kname);
    let v = cldev.register_buffer(v);
    cldev.run_kernel(&kname,v);
    let v: Vec<i32> = cldev.map_buffer(v);
    println!("v={:?}",v);

    let v = cldev.unmap_buffer(v);
    cldev.run_kernel(&kname,v);
    let v: Vec<i32> = cldev.map_buffer(v);
    println!("v={:?}",v);

    let w : Vec<i32> = vec![11; 10];
    let w = cldev.register_buffer(w);
    //let wt: Vec<i32> = cldev.map_buffer(w);
    cldev.run_kernel(&kname,w);
    //let w: Vec<i32> = cldev.map_buffer(w);
    println!("w={:?}",w);
    let w: Vec<i32> = cldev.map_buffer(w);
    println!("w={:?}",w);
    let w = cldev.unmap_buffer(w);
    println!("w={:?}",w);

    run_kernel!(cldev,1,2,3);

    //std::mem::forget(v); // moche moche moche !

}




