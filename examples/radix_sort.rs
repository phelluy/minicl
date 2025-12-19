// Radix sort algorithm in Rust and OpenCL
// based on:
// A portable implementation of the radix sort algorithm in OpenCL, 2011.
// http://hal.archives-ouvertes.fr/hal-00596730

use minicl::Accel;
use minicl::MCLError;
use std::fs;
use std::io::stdin;
use std::time::Instant;

// CONSTANTS
const ITEMS: usize = 64;
const GROUPS: usize = 16;
const HISTOSPLIT: usize = 512;
const TOTALBITS: usize = 30; // 32
const BITS: usize = 5;
const RADIX: usize = 1 << BITS; // 32
const PASS: usize = TOTALBITS / BITS; // 6
const N: usize = 32 * (1 << 20); // 32M
const MAXINT: u32 = 1 << (TOTALBITS - 1);
// const VERBOSE: bool = true;

// Simple LCG for random numbers to avoid 'rand' dependency
struct Lcg {
    state: u32,
}
impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }
    fn next(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        self.state
    }
}

// Helper for random data
fn rand_data(n: usize) -> Vec<u32> {
    let mut v = Vec::with_capacity(n);
    let mut rng = Lcg::new(12345);
    for _ in 0..n {
        v.push(rng.next() % MAXINT);
    }
    v
}

fn main() -> Result<(), MCLError> {
    println!("Radix Sort OpenCL Example in Rust");
    println!("Reading OpenCL source...");

    let mut source = fs::read_to_string("examples/radix_sort_kernel.cl")
        .expect("Could not read examples/radix_sort_kernel.cl");

    // Substitute parameters
    source = source.replace("<ITEMS>", &ITEMS.to_string());
    source = source.replace("<GROUPS>", &GROUPS.to_string());
    source = source.replace("<HISTOSPLIT>", &HISTOSPLIT.to_string());
    source = source.replace("<TOTALBITS>", &TOTALBITS.to_string());
    source = source.replace("<BITS>", &BITS.to_string());
    source = source.replace("<BITS>", &BITS.to_string());
    source = source.replace("<N>", &N.to_string());

    // Calculate max local memory for scan
    let maxmemcache = std::cmp::max(HISTOSPLIT, (ITEMS * GROUPS * RADIX) / HISTOSPLIT);
    let max_loc_scan = maxmemcache; // This is the count of items
    source = source.replace("<MAX_LOC_SCAN>", &max_loc_scan.to_string());

    println!("Enter platform num:");
    let mut s = String::new();
    stdin()
        .read_line(&mut s)
        .expect("Did not enter a correct string");
    let input: usize = s.trim().parse().unwrap_or(0);
    let numplat = input;

    let mut cldev = Accel::new(source, numplat)?;

    // Register kernels
    let k_histogram = "histogram";
    cldev.register_kernel(k_histogram)?;
    let k_scan_histogram = "scanhistograms";
    cldev.register_kernel(k_scan_histogram)?;
    let k_paste_histogram = "pastehistograms";
    cldev.register_kernel(k_paste_histogram)?;
    let k_reorder = "reorder";
    cldev.register_kernel(k_reorder)?;

    // Data generation
    println!("Generating {} random keys...", N);
    let h_keys = rand_data(N);
    // let h_check_keys = h_keys.clone(); // For verification if needed

    // Buffers
    // d_inKeys
    let mut d_in_keys = cldev.register_buffer(h_keys)?;
    // d_outKeys
    let d_out_keys_vec = vec![0u32; N];
    let mut d_out_keys = cldev.register_buffer(d_out_keys_vec)?;

    // d_inPermut (0..N) - Used if PERMUT defined or in reorder args
    // Even if we don't verify permutation, the kernel signature might require it?
    // reorder signature: (inKeys, outKeys, Histograms, pass, inPermut, outPermut, local_histo, n)
    // It requires them.
    let h_permut: Vec<u32> = (0..N as u32).collect();
    let mut d_in_permut = cldev.register_buffer(h_permut)?;
    // d_outPermut
    let d_out_permut_vec = vec![0u32; N];
    let mut d_out_permut = cldev.register_buffer(d_out_permut_vec)?;

    // Histograms
    let h_histograms = vec![0u32; RADIX * GROUPS * ITEMS];
    let d_histograms = cldev.register_buffer(h_histograms)?;

    // Global Sum
    let h_globsum = vec![0u32; HISTOSPLIT];
    let d_globsum = cldev.register_buffer(h_globsum)?;

    // Temp for scan
    let d_temp_vec = vec![0u32; HISTOSPLIT];
    let d_temp = cldev.register_buffer(d_temp_vec)?;

    // Resize / Padding Logic (simplified: assuming N is valid as per C++ default)
    assert_eq!(N % (GROUPS * ITEMS), 0);
    let nkeys_rounded = N;

    println!("Starting Sort...");
    let start_time = Instant::now();

    // Loop
    for pass in 0..PASS {
        // --- 1. HISTOGRAM ---
        // __kernel void histogram(d_Keys, d_Histograms, pass, __local loc_histo, n)
        // Global: GROUPS * ITEMS
        // Local: ITEMS
        let g_size_histo = GROUPS * ITEMS;
        let l_size_histo = ITEMS;

        let pass_i32 = pass as i32;
        let nkeys_i32 = nkeys_rounded as i32;
        // __kernel void histogram(d_Keys, d_Histograms, pass, n)
        // Local memory internalized

        // Remove loc_histo_struct from args
        minicl::kernel_set_args_and_run!(
            cldev,
            k_histogram,
            g_size_histo,
            l_size_histo,
            d_in_keys,
            d_histograms,
            pass_i32,
            nkeys_i32
        )?;

        // --- 2. SCAN HISTOGRAMS ---
        let nbitems_scan1 = (RADIX * GROUPS * ITEMS) / 2;
        let nblocitems_scan1 = nbitems_scan1 / HISTOSPLIT;

        // Max local memory for scan kernel
        let _maxmemcache = std::cmp::max(HISTOSPLIT, (ITEMS * GROUPS * RADIX) / HISTOSPLIT);

        // Part 1: Scan locally
        // scanhistograms(histo, globsum) - temp is internal
        minicl::kernel_set_args_and_run!(
            cldev,
            k_scan_histogram,
            nbitems_scan1,
            nblocitems_scan1,
            d_histograms,
            d_globsum
        )?;

        // Part 2: Scan global sum
        let nbitems_scan2 = HISTOSPLIT / 2;
        let nblocitems_scan2 = nbitems_scan2;

        // Scan2: input=d_globsum. Output=d_globsum (scanned). Aux Output=d_temp (unused).
        // scanhistograms(globsum, temp)

        minicl::kernel_set_args_and_run!(
            cldev,
            k_scan_histogram,
            nbitems_scan2,
            nblocitems_scan2,
            d_globsum,
            d_temp
        )?;

        // --- 3. PASTE HISTOGRAMS ---
        let nbitems_paste = (RADIX * GROUPS * ITEMS) / 2;
        let nblocitems_paste = nbitems_paste / HISTOSPLIT;

        // pastehistograms(histo, globsum)

        let nbitems_paste = (RADIX * GROUPS * ITEMS) / 2;
        let nblocitems_paste = nbitems_paste / HISTOSPLIT;

        minicl::kernel_set_args_and_run!(
            cldev,
            k_paste_histogram,
            nbitems_paste,
            nblocitems_paste,
            d_histograms,
            d_globsum
        )?;

        // --- 4. REORDER ---
        let g_size_reorder = GROUPS * ITEMS;
        let l_size_reorder = ITEMS;

        // reorder(d_inKeys, d_outKeys, d_Histograms, pass, d_inPermut, d_outPermut, n)

        minicl::kernel_set_args_and_run!(
            cldev,
            k_reorder,
            g_size_reorder,
            l_size_reorder,
            d_in_keys,
            d_out_keys,
            d_histograms,
            pass_i32,
            d_in_permut,
            d_out_permut,
            nkeys_i32
        )?;

        // Swap buffers for next pass
        std::mem::swap(&mut d_in_keys, &mut d_out_keys);
        std::mem::swap(&mut d_in_permut, &mut d_out_permut);
    } // End of loop

    let duration = start_time.elapsed();
    println!("Sorting took: {:?}", duration);

    // Verify
    // The final result is in d_in_keys (because we swapped at end of loop)
    // Map back
    let final_keys: Vec<u32> = cldev.map_buffer(d_in_keys)?;
    // let final_permut = cldev.map_buffer(d_in_permut)?;

    println!("Verifying order...");
    let mut ok = true;
    for i in 0..N - 1 {
        if final_keys[i] > final_keys[i + 1] {
            println!(
                "Error at index {}: {} > {}",
                i,
                final_keys[i],
                final_keys[i + 1]
            );
            ok = false;
            if i > 10 {
                break;
            } // Don't spam
        }
    }

    if ok {
        println!("Test OK!");
    } else {
        println!("Test FAILED!");
    }

    Ok(())
}
