extern crate minicl;

fn main() {


    println!("Simple kernel add example");

    let source = "__kernel  void simple_add(__global int *v){
        int i = get_global_id(0);
        v[i] += 12;
    }".to_string();



    let mut cldev = minicl::Accel::new(source);

    println!("Accel={:?}", cldev);

    let v : Vec<i32> = vec![12; 10];

    let kname = "simple_add".to_string();
    cldev.register_kernel(&kname);

    let v = cldev.register_buffer(v);

    cldev.run_kernel(&kname,v);

    let v: Vec<i32> = cldev.take_buffer(v);

    let v = cldev.register_buffer(v);
    cldev.run_kernel(&kname,v);

    let v: Vec<i32> = cldev.take_buffer(v);

    println!("v={:?}",v);

    let w : Vec<i32> = vec![11; 10];
    let w = cldev.register_buffer(w);
    cldev.run_kernel(&kname,w);
    let w: Vec<i32> = cldev.take_buffer(w);
    println!("w={:?}",w);

    //std::mem::forget(v); // moche moche moche !

}

