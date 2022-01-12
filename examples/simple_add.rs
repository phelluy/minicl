extern crate minicl;

fn main() {


    println!("Simple kernel add example");

    let source = "__kernel  void simple_add(__global int *v){
        int i = get_global_id(0);
        v[i] += 12;
    }".to_string();



    let mut cldev = minicl::Accel::new(source);

    println!("Accel={:?}", cldev);

    let mut v = vec![12; 10];

    let v = cldev.run_kernel(v);

    println!("v={:?}",v);

    std::mem::forget(v); // moche moche moche !

}

