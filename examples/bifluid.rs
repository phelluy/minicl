// resolution of the wave equation on a square
// with the leapfrog method and minicl
// you need python and matplotlib for seeing the results
fn main() -> Result<(), minicl::MCLError> {
    use std::fs;

    // numerical parameters
    let nx = 512;
    let ny = 256;
    let mw = 4;
    let nk = mw * 4;
    let tmax: f32 = 0.0;
    let lx: f32 = 0.146;
    let ly: f32 = 0.146 / 2.;

    let dx = lx / nx as f32;
    let dy = ly / ny as f32;

    let vmax: f32 = 20.;

    let cfl: f32 = 1.;

    let dt: f32 = cfl * dx / vmax;

    println!("grid size {}x{}", nx, ny);

    // tuning of the OpenCL sources
    let mut source = fs::read_to_string("examples/bifluid_kernels.cl").unwrap();

    source = source.replace("real", &"float");
    source = source.replace("_float_", &"float");
    source = source.replace("_F", &"f");
    source = source.replace("_nx_", &nx.to_string());
    source = source.replace("_ny_", &ny.to_string());
    source = source.replace("_dx_", &dx.to_string());
    source = source.replace("_dy_", &dy.to_string());
    source = source.replace("_lambda_", &vmax.to_string());
    source = source.replace("_dt_", &dt.to_string());
    source = source.replace("_m_", &mw.to_string());
    source = source.replace("_n_", &nk.to_string());

    // ask the platform id to the user
    use std::io::stdin;
    let mut s = String::new();
    println!("Platform id ? (0 is a safe choice)");
    stdin().read_line(&mut s).unwrap();
    let input: usize = s.trim().parse().unwrap();
    let numplat = input;

    let mut cldev = minicl::Accel::new(source, numplat)?;

    // registration of the kernels
    let init_sol = "init_sol".to_string();
    cldev.register_kernel(&init_sol)?;
    let time_step = "time_step".to_string();
    cldev.register_kernel(&time_step)?;

    let n = nx * ny;

    // memory buffers needed for the leapfrog computations
    let fnow: Vec<f32> = vec![0.; nk * n];
    let fnext: Vec<f32> = vec![0.; nk * n];
    let mut fnow = cldev.register_buffer(fnow)?;
    let mut fnext = cldev.register_buffer(fnext)?;

    let globsize = n;
    let locsize = 32;

    use std::time::Instant;
    let start = Instant::now();
    // initial data
    minicl::kernel_set_args_and_run!(cldev, init_sol, globsize, locsize, fnow)?;
    // time loop
    let mut t = 0.;
    let mut count = 0;
    while t < tmax {
        t += dt;
        count += 1;
        minicl::kernel_set_args_and_run!(cldev, time_step, globsize, locsize, fnow, fnext)?;
        println!("tmax={} tend={}", tmax, t);
        let temp = fnow;
        fnow = fnext;
        fnext = temp;
    }

    let duration = start.elapsed();
    println!("{} iters in {:?}", count, duration);

    println!("Plotting...");
    // get back the buffer on the host
    // for plotting
    let fnow: Vec<f32> = cldev.map_buffer(fnow)?;
    let mut wnow: Vec<f32> = vec![0.; n];

    let iplot = 3;
    for ij in 0..n {
        for d in 0..4 {
            let ik = mw * d + iplot;
            wnow[ij] += fnow[ij + ik * n];
        }
    }

    let xp: Vec<f32> = (0..nx).map(|i| i as f32 * dx).collect();
    let yp: Vec<f32> = (0..ny).map(|i| i as f32 * dy).collect();
    plotpy(xp.clone(), yp.clone(), wnow);
    Ok(())
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
