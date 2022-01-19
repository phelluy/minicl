//#[macro_use]
//extern crate minicl;

fn main() {
    use std::fs;

    let nx = 1024;
    let ny = 1024;
    let tmax: f32 = 2.;
    let lx: f32 = 1.;
    let ly: f32 = 1.;

    let dx = lx / (nx - 1) as f32;
    let dy = ly / (ny - 1) as f32;

    let cson: f32 = (2 as f32).sqrt();

    let cfl: f32 = 0.4;

    let dt: f32 = cfl * (dx * dx + dy * dy).sqrt() / cson;

    let mut source = fs::read_to_string("examples/wave2d_kernels.cl").unwrap();

    source = source.replace("real", &"float");
    source = source.replace("_float_", &"float");
    source = source.replace("_F", &"f");
    source = source.replace("_nx_", &nx.to_string());
    source = source.replace("_ny_", &ny.to_string());
    source = source.replace("_dx_", &dx.to_string());
    source = source.replace("_dy_", &dy.to_string());
    source = source.replace("_cson_", &cson.to_string());
    let source = source.replace("_dt_", &dt.to_string());

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
    let init_sol = "init_sol".to_string();
    cldev.register_kernel(&init_sol);
    let time_step = "time_step".to_string();
    cldev.register_kernel(&time_step);

    let n = nx * ny;

    // the memory buffer shared with the
    // accelerator has to be registered
    let un: Vec<f32> = vec![0.; n];
    let unm1: Vec<f32> = vec![0.; n];
    let unp1: Vec<f32> = vec![0.; n];
    let mut un = cldev.register_buffer(un);
    let mut unm1 = cldev.register_buffer(unm1);
    let mut unp1 = cldev.register_buffer(unp1);

    let globsize = n;
    let locsize = 32;

    use std::time::Instant;
    let start = Instant::now();
    minicl::kernel_set_args_and_run!(cldev, init_sol, globsize, locsize, un, unm1);

    let mut t = 0.;
    while t < tmax {
        t += dt;
        println!("t={}", t);
        minicl::kernel_set_args_and_run!(cldev, time_step, globsize, locsize, t, unm1, un, unp1);
        // let unm1: Vec<f32> = cldev.map_buffer(unm1);
        // let unp1: Vec<f32> = cldev.map_buffer(unp1);
        // let un: Vec<f32> = cldev.map_buffer(un);
        // let unm1= cldev.unmap_buffer(unm1);
        // let unp1= cldev.unmap_buffer(unp1);
        // let un= cldev.unmap_buffer(un);

        let temp = unm1;
        unm1 = un;
        un = unp1;
        unp1 = temp;
    }

    let duration = start.elapsed();
    println!("Computing time: {:?}", duration);
    let unp1: Vec<f32> = cldev.map_buffer(unp1);

    let xp: Vec<f32> = (0..nx).map(|i| i as f32 * dx).collect();
    let yp: Vec<f32> = (0..ny).map(|i| i as f32 * dy).collect();
    plotpy(xp.clone(), yp.clone(), unp1);
}

/// Plots a 2D data set using matplotlib.
fn plotpy(xp: Vec<f32>, yp: Vec<f32>, zp: Vec<f32>) {
    use std::fs::File;
    use std::io::BufWriter;
    use std::io::Write;
    {
        let meshfile = File::create("plotpy.dat").unwrap();
        let mut meshfile = BufWriter::new(meshfile); // create a buffer for faster writes...
        xp.iter().for_each(|x| {
            writeln!(meshfile, "{}", x).unwrap();
        });
        writeln!(meshfile).unwrap();
        yp.iter().for_each(|y| {
            writeln!(meshfile, "{}", y).unwrap();
        });
        writeln!(meshfile).unwrap();
        zp.iter().for_each(|z| {
            writeln!(meshfile, "{}", z).unwrap();
        });
    } // ensures that the file is closed

    use std::process::Command;
    Command::new("python3")
        .arg("examples/plot.py")
        .status()
        .expect("Plot failed: you need Python3 and Matplotlib in your PATH.");
}
