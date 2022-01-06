extern crate minicl;

fn main() {


    println!("Simple kernel add example");

    let source = "__kernel simple_add(__global *v){
        int i = get_global_id(0);
        v[i] += 12;
    }".to_string();



    let mut cldev = minicl::Accel::new(source);

    println!("Accel={:?}", cldev);

}

