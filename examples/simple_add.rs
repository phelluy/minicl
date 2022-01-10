extern crate minicl;

fn main() {


    println!("Simple kernel add example");

    let source = "__kernel void simple_add(__global int *v){
        int i = get_global_id(0);
        v[i] += 12;
    }".to_string();



    let mut cldev = minicl::Accel::new(source);

    println!("Accel={:?}", cldev);

}

