//#[macro_use]
//extern crate minicl;

fn main() {
    use std::fs;

    let nx = 128;
    let ny = 128;
    let tmax: f32 = 2.;
    let lx: f32 = 2.;
    let ly: f32 = 1.;

    let dx = lx / (nx-1) as f32;
    let dy = ly / (ny-1) as f32;

    let cson: f32 = (2 as f32).sqrt();

    let cfl: f32 = 0.4;

    let dt: f32 = cfl * (dx * dx + dy * dy).sqrt() / cson;

    let mut source = fs::read_to_string("examples/wave2d_kernels.cl").unwrap();

    source = source.replace("real",&"float");
    source = source.replace("_float_",&"float");
    source = source.replace("_F",&"f");
    source = source.replace("_nx_",&nx.to_string());
    source = source.replace("_ny_",&ny.to_string());
    source = source.replace("_dx_",&dx.to_string());
    source = source.replace("_dy_",&dy.to_string());
    source = source.replace("_cson_",&cson.to_string());
    source = source.replace("_dt_",&dt.to_string());


    use std::io::stdin;
    let mut s = String::new();
    println!("Enter platform num.");
    stdin()
        .read_line(&mut s)
        .expect("Did not enter a correct string");
    let input: usize = s.trim().parse().expect("Wanted a number");
    let numplat = input;

    let mut cldev = minicl::Accel::new(source, numplat);

    // the used kernels has to be registered
    let kname = "init_sol".to_string();
    cldev.register_kernel(&kname);

    let n = nx*ny;

    // the memory buffer shared with the
    // accelerator has to be registered
    let un: Vec<f32> = vec![0.; n];
    let unm1: Vec<f32> = vec![0.; n];
    let un = cldev.register_buffer(un);
    let unm1 = cldev.register_buffer(unm1);


    let globsize = n;
    let locsize = 32;

    use std::time::Instant;
    let start = Instant::now();
    minicl::kernel_set_args_and_run!(cldev, kname, globsize, locsize, un, unm1);
    let duration = start.elapsed();
    println!("Computing time: {:?}", duration);

    // map the buffer for access from the host
    let un: Vec<f32> = cldev.map_buffer(un);
    //println!("First kernel run v={:?}", v);

}
