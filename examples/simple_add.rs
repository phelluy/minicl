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

    let vname = "v".to_string();
    cldev.register_buffer(vname.clone(),v);

    let kname = "simple_add".to_string();
    cldev.register_kernel(kname.clone());

    //cldev.run_kernel(kname,vname.clone());

    let v: Vec<i32> = cldev.take_buffer(vname);

    println!("v={:?}",v);

    //std::mem::forget(v); // moche moche moche !

}

