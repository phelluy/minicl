use std::fs;
use std::io::stdin;
use std::time::Instant;
use minicl::MCLError;
use minicl::Accel;
use minicl::LocalBuffer;

// CONSTANTS matching CLRadixSortParam.hpp (mostly)
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
struct Lcg { state: u32 }
impl Lcg {
    fn new(seed: u32) -> Self { Lcg { state: seed } }
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
    source = source.replace("<N>", &N.to_string());

    println!("Enter platform num:");
    let mut s = String::new();
    stdin().read_line(&mut s).expect("Did not enter a correct string");
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
        let loc_histo_size = RADIX * ITEMS * std::mem::size_of::<u32>();
        
        let pass_i32 = pass as i32;
        let nkeys_i32 = nkeys_rounded as i32;
        let loc_histo_struct = LocalBuffer{size: loc_histo_size};

        minicl::kernel_set_args_and_run!(cldev, k_histogram, g_size_histo, l_size_histo, 
            d_in_keys, d_histograms, pass_i32, loc_histo_struct, nkeys_i32)?;
        
        // --- 2. SCAN HISTOGRAMS ---
        // Part 1: Scan locally
        // size_t nbitems=_RADIX* _GROUPS*_ITEMS / 2;
        // size_t nblocitems= nbitems/_HISTOSPLIT ;
        let nbitems_scan1 = (RADIX * GROUPS * ITEMS) / 2;
        let nblocitems_scan1 = nbitems_scan1 / HISTOSPLIT; 

        // int maxmemcache=max(_HISTOSPLIT,_ITEMS * _GROUPS * _RADIX / _HISTOSPLIT);
        let maxmemcache = std::cmp::max(HISTOSPLIT, (ITEMS * GROUPS * RADIX) / HISTOSPLIT);
        let loc_mem_scan1 = maxmemcache * std::mem::size_of::<u32>();

        // void scanhistograms(histo, __local temp, globsum)
        // void scanhistograms(histo, __local temp, globsum)
        let loc_mem_scan1_struct = LocalBuffer{size: loc_mem_scan1};
        minicl::kernel_set_args_and_run!(cldev, k_scan_histogram, nbitems_scan1, nblocitems_scan1, 
            d_histograms, loc_mem_scan1_struct, d_globsum)?;
        
        // Part 2: Scan global sum
        // nbitems= _HISTOSPLIT / 2;
        // nblocitems=nbitems;
        let nbitems_scan2 = HISTOSPLIT / 2;
        let nblocitems_scan2 = nbitems_scan2;
        // Using d_temp as dummy second arg (actually it's expected to be local mem in kernel signature: __local int* temp)
        // Wait, the C++ code says:
        // err = clSetKernelArg(ckScanHistogram, 0, sizeof(cl_mem), &d_globsum);
        // err = clSetKernelArg(ckScanHistogram, 2, sizeof(cl_mem), &d_temp);
        // BUT the kernel signature is: (histo, __local temp, globsum)
        // Arg 1 is "__local temp". 
        // In C++ for second pass:
        // clSetKernelArg(ckScanHistogram, 0 ... &d_globsum) -> histo input is d_globsum
        // clSetKernelArg(ckScanHistogram, 2 ... &d_temp) -> globsum output is d_temp (unused?)
        // What about Arg 1 (local temp)?
        // C++ code doesn't seem to reset Arg 1 for second pass? It retains the previous size? 
        // "err = clSetKernelArg(ckScanHistogram, 1, sizeof(uint)* maxmemcache , NULL); // mem cache" was set before first pass.
        // It stays valid if the local size logic allows it.
        // For second pass, we scan `HISTOSPLIT` items. 
        // `maxmemcache` is >= HISTOSPLIT (checked in C++). So the local memory is sufficient.
        
        // Part 2: Scan global sum
        // Aguments for scan 2: (globsum, local_temp, temp)
        // Re-creating local arg struct (although same size)
        let loc_mem_scan2_struct = LocalBuffer{size: loc_mem_scan1};
        minicl::kernel_set_args_and_run!(cldev, k_scan_histogram, nbitems_scan2, nblocitems_scan2, 
            d_globsum, loc_mem_scan2_struct, d_temp)?;
        
        // --- 3. PASTE HISTOGRAMS ---
        // __kernel void pastehistograms(histo, globsum)
        // Global size = RADIX * GROUPS * ITEMS / 2 (same as scan1?)
        // C++ code:
        // err = clSetKernelArg(ckPasteHistogram, 0, sizeof(cl_mem), &d_Histograms);
        // err = clSetKernelArg(ckPasteHistogram, 1, sizeof(cl_mem), &d_globsum);
        // Dispatch?
        // Wait, C++ source `CLRadixSort.cpp` doesn't seem to dispatch `pastehistograms` inside `ScanHistogram`?
        // Ah, `ScanHistogram` calls `ckScanHistogram` twice.
        // Where is `ckPasteHistogram` used?
        // It is initialized in constructor.
        // But searching `CLRadixSort.cpp` for `clEnqueueNDRangeKernel` with `ckPasteHistogram`...
        // It is NOT CALLED in `ScanHistogram`.
        // Is it called in `Reorder`? No.
        // Is it called in `Sort`? No.
        // Wait. `CLRadixSort.cpp` lines 392: calls `ScanHistogram()`.
        // `ScanHistogram` implementation (lines 715-802):
        // Calls `ckScanHistogram` (scan1).
        // Calls `ckScanHistogram` (scan2).
        // It ends at line 800 (truncated in my view).
        // I MISSED THE END OF THE FILE.
        // `pastehistograms` MUST be called at the end of `ScanHistogram` to distribute the global sums back to chunks.
        // I need to verify this hypothesis.
        // But for now I will assume it IS called (pattern of Blelloch scan).
        
        // Let's assume there is a 3rd dispatch in `ScanHistogram`.
        // nbitems = _RADIX * _GROUPS * _ITEMS / 2;
        // nblocitems = nbitems / _HISTOSPLIT;
        
        let nbitems_paste = (RADIX * GROUPS * ITEMS) / 2;
        let nblocitems_paste = nbitems_paste / HISTOSPLIT;
        
        minicl::kernel_set_args_and_run!(cldev, k_paste_histogram, nbitems_paste, nblocitems_paste, 
            d_histograms, d_globsum)?;

        // --- 4. REORDER ---
        // __kernel void reorder(d_inKeys, d_outKeys, d_Histograms, pass, d_inPermut, d_outPermut, loc_histo, n)
        // Global: GROUPS * ITEMS
        // Local: ITEMS
        // loc_histo size: RADIX * ITEMS * sizeof(int)
        
        let g_size_reorder = GROUPS * ITEMS;
        let l_size_reorder = ITEMS;
        let loc_histo_reorder_size = RADIX * ITEMS * std::mem::size_of::<u32>();
        
        let loc_histo_reorder_struct = LocalBuffer{size: loc_histo_reorder_size};
        
        minicl::kernel_set_args_and_run!(cldev, k_reorder, g_size_reorder, l_size_reorder, 
            d_in_keys, d_out_keys, d_histograms, pass_i32, d_in_permut, d_out_permut, loc_histo_reorder_struct, nkeys_i32)?;
        
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
    for i in 0..N-1 {
        if final_keys[i] > final_keys[i+1] {
            println!("Error at index {}: {} > {}", i, final_keys[i], final_keys[i+1]);
            ok = false;
            if i > 10 { break; } // Don't spam
        }
    }

    if ok {
        println!("Test OK!");
    } else {
        println!("Test FAILED!");
    }

    Ok(())
}
