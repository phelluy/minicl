/// Minimal GPU compute library, based on OpenCL.
///  With normal use, this library should prevent memory leaks
/// and most unsafety related to OpenCL.
use std::collections::HashMap;
/// All the OpenCL things (device, context, buffers, etc.)
///  are packed into a single Accelerator struct.
/// # Examples
/// ```
///let source = "__kernel  void simple_add(__global int *v, int x){
///int i = get_global_id(0);
///v[i] += x;
///}"
///.to_string();
///let num_platform = 0;
///let mut cldev = minicl::Accel::new(source, num_platform);
///
/// // the used kernel has to be registered
///let kname = "simple_add".to_string();
///cldev.register_kernel(&kname);
///
///let n = 64;
///
/// // the memory buffer shared with the
/// // accelerator has to be registered
///let v: Vec<i32> = vec![12; n];
///let v = cldev.register_buffer(v);
///
///let x: i32 = 1000;
///let globsize = n;
///let locsize = 16;
///
/// // invoke this macro for the first kernel call
///minicl::kernel_set_args_and_run!(cldev, kname, globsize, locsize, v, x);
///
/// // map the buffer for access from the host
///let v: Vec<i32> = cldev.map_buffer(v);
///println!("First kernel run v={:?}", v);
///
/// // unmap for giving it back to the device
///let v = cldev.unmap_buffer(v);
///
/// // next call: no need to redefine the kernel args
/// // if they are the same
///cldev.run_kernel(&kname, globsize, locsize);
///
///let v: Vec<i32> = cldev.map_buffer(v);
///println!("Next kernel run v={:?}", v);
///
/// ```
#[derive(Debug)]
pub struct Accel {
    // source: String,
    // platform: cl_sys::cl_platform_id,
    context: cl_sys::cl_context,
    device: cl_sys::cl_device_id,
    program: cl_sys::cl_program,
    queue: cl_sys::cl_command_queue,
    kernels: HashMap<String, cl_sys::cl_kernel>,
    buffers: HashMap<*mut cl_sys::c_void, (cl_sys::cl_mem, usize, usize, bool)>,
}

impl Accel {
    /// Generate a minicl environment
    /// from an OpenCL source code and a platform id
    pub fn new(oclsource: String, numplat: usize) -> Accel {
        let platform: cl_sys::cl_platform_id = std::ptr::null_mut();
        let mut nb_platforms: u32 = 0;
        let mut device: cl_sys::cl_device_id = std::ptr::null_mut();
        let err = unsafe { cl_sys::clGetPlatformIDs(0, std::ptr::null_mut(), &mut nb_platforms) };
        assert_eq!(err, cl_sys::CL_SUCCESS);
        println!("Found {} platform(s)", nb_platforms);
        assert!(10 > nb_platforms);
        let mut platform = [platform; 10];
        let err =
            unsafe { cl_sys::clGetPlatformIDs(nb_platforms, &mut platform[0], &mut nb_platforms) };
        assert_eq!(err, cl_sys::CL_SUCCESS);
        // use std::io::stdin;
        // let mut s = String::new();
        // println!("Enter platform num.");
        // stdin()
        //     .read_line(&mut s)
        //     .expect("Did not enter a correct string");
        // let input: usize = s.trim().parse().expect("Wanted a number");
        // let numplat = input;
        assert!(numplat < nb_platforms as usize);

        let mut size: usize = 0;
        let err = unsafe {
            cl_sys::clGetPlatformInfo(
                platform[numplat],
                cl_sys::CL_PLATFORM_VENDOR,
                0,
                std::ptr::null_mut(),
                &mut size,
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);
        assert!(size > 0);

        let platform_name = vec![1; size];
        let platform_name = String::from_utf8(platform_name).unwrap();
        let platform_name = std::ffi::CString::new(platform_name).unwrap();
        let err = unsafe {
            cl_sys::clGetPlatformInfo(
                platform[numplat],
                cl_sys::CL_PLATFORM_VENDOR,
                size,
                platform_name.as_ptr() as *mut cl_sys::c_void,
                &mut size,
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);
        let platform_name = unsafe {
            std::ffi::CStr::from_ptr(platform_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        println!("Platform: {}", platform_name);

        let mut temp: u32 = 0;
        let err = unsafe {
            cl_sys::clGetDeviceIDs(
                platform[numplat],
                cl_sys::CL_DEVICE_TYPE_GPU | cl_sys::CL_DEVICE_TYPE_CPU,
                1,
                &mut device,
                &mut temp,
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);

        let mut err: i32 = 0;
        let context = unsafe {
            cl_sys::clCreateContext(
                std::ptr::null(),
                1,
                &device,
                None,
                std::ptr::null_mut(),
                &mut err,
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);

        let mut err: i32 = 0;
        let queue = unsafe {
            cl_sys::clCreateCommandQueue(
                context,
                device,
                cl_sys::CL_QUEUE_PROFILING_ENABLE,
                &mut err,
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);

        let mut err: i32 = 0;
        let oclsources = [oclsource.as_str()];
        let program = unsafe {
            cl_sys::clCreateProgramWithSource(
                context,
                1,
                oclsources.as_ptr() as *const *const cl_sys::libc::c_char,
                std::ptr::null(),
                &mut err,
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);

        let opt = std::ffi::CString::new("-w").unwrap();
        let log: *mut cl_sys::c_void = std::ptr::null_mut();
        let errb = unsafe { cl_sys::clBuildProgram(program, 1, &device, opt.as_ptr(), None, log) };

        // first get the size of the build log
        let mut size = 0;
        let err = unsafe {
            cl_sys::clGetProgramBuildInfo(
                program,
                device,
                cl_sys::CL_PROGRAM_BUILD_LOG,
                0,
                std::ptr::null_mut(),
                &mut size,
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);
        println!("Size of build log: {}", size);
        // then get the build log
        let log = vec![1; size];
        let log = String::from_utf8(log).unwrap();
        //let log: *mut cl_sys::c_void = std::ptr::null_mut();
        let log = std::ffi::CString::new(log).unwrap();

        let err = unsafe {
            cl_sys::clGetProgramBuildInfo(
                program,
                device,
                cl_sys::CL_PROGRAM_BUILD_LOG,
                size,
                log.as_ptr() as *mut cl_sys::c_void,
                &mut size,
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);
        let log = unsafe {
            std::ffi::CStr::from_ptr(log.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        println!("Build messages:\n-------------------------------------");
        println!("{}", log);
        println!("-------------------------------------");
        assert_eq!(errb, cl_sys::CL_SUCCESS, "Build failure");

        Accel {
            // source: oclsource,
            // platform: platform[numplat],
            context,
            device,
            program,
            queue,
            kernels: HashMap::new(),
            buffers: HashMap::new(),
        }
    }

    pub fn register_kernel(&mut self, name: &String) {
        let mut err: i32 = 0;
        let cname = std::ffi::CString::new(name.clone()).unwrap();
        let kernel: cl_sys::cl_kernel =
            unsafe { cl_sys::clCreateKernel(self.program, cname.as_ptr(), &mut err) };
        assert_eq!(err, cl_sys::CL_SUCCESS);
        println!("kernel={:?}", kernel);
        self.kernels.insert(name.clone(), kernel);
    }

    pub fn register_buffer<T>(&mut self, mut v: Vec<T>) -> *mut cl_sys::c_void {
        v.shrink_to_fit();
        assert!(v.len() == v.capacity());
        let ptr0 = v.as_mut_ptr() as *mut cl_sys::c_void;
        assert!(
            !self.buffers.contains_key(&ptr0),
            "Buffer already registered."
        );
        let n = v.len();
        // leave deallocation duty to MiniCL
        std::mem::forget(v);
        let szf = std::mem::size_of::<T>();
        let mut err: i32 = 0;
        let buffer: cl_sys::cl_mem = unsafe {
            cl_sys::clCreateBuffer(
                self.context,
                cl_sys::CL_MEM_READ_WRITE | cl_sys::CL_MEM_USE_HOST_PTR,
                n * szf,
                ptr0,
                &mut err,
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);
        let is_map = false;
        self.buffers.insert(ptr0, (buffer, n * szf, szf, is_map));
        ptr0
    }

    pub fn unmap_buffer<T>(&mut self, mut v: Vec<T>) -> *mut cl_sys::c_void {
        v.shrink_to_fit();
        assert!(v.len() == v.capacity());
        let ptr0 = v.as_mut_ptr() as *mut cl_sys::c_void;
        assert!(
            self.buffers.contains_key(&ptr0),
            "Buffer not yet registered."
        );
        let tup = self.buffers.get(&ptr0).unwrap();
        let buffer = tup.0;
        let size = tup.1;
        let szf = tup.2;
        let is_map = tup.3;
        assert!(is_map);
        //let (mut buffer, mut size, mut szf, mut is_map) = self.buffers.get(&ptr0).unwrap();
        self.buffers.remove(&ptr0).unwrap();
        println!("ptr0 before cl buffer création: {:?}", ptr0);
        //let n = v.len();
        std::mem::forget(v);
        // let szf = std::mem::size_of::<T>();
        // println!("size={}", n * szf);
        let err = unsafe {
            cl_sys::clEnqueueUnmapMemObject(
                self.queue,
                buffer,
                ptr0,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);
        let is_map = false;
        self.buffers.insert(ptr0, (buffer, size, szf, is_map));
        //println!("buffer={:?}", buffer);
        //self.buffers.insert(ptr0, (buffer, n * szf, szf));
        ptr0
    }

    pub fn map_buffer<T>(&mut self, ptr0: *mut cl_sys::c_void) -> Vec<T> {
        let mut err = 0;
        let blocking = cl_sys::CL_TRUE;
        //let szf = std::mem::size_of::<T>();
        //let toto = self.buffers.get(&name).unwrap();
        let tup = self.buffers.get(&ptr0).unwrap();
        let buffer = tup.0;
        let size = tup.1;
        let szf = tup.2;
        let is_map = tup.3;
        assert!(!is_map);
        //let (mut buffer, mut size, mut szf, mut is_map) = self.buffers.get(&ptr0).unwrap();
        println!("buffer={:?} size={} szf={}", buffer, size, szf);
        let ptr = unsafe {
            cl_sys::clEnqueueMapBuffer(
                self.queue,
                buffer,
                blocking,
                cl_sys::CL_MAP_READ,
                0,
                size,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut err,
            )
        } as *mut T;
        assert_eq!(err, cl_sys::CL_SUCCESS);
        self.buffers.remove(&ptr0);
        let is_map = true;
        self.buffers.insert(ptr0, (buffer, size, szf, is_map));
        let n = size / szf;
        println!("size={} szf={}", size, szf);
        assert!(size % szf == 0);
        println!("ptr0 before cl map: {:?}", ptr0);
        println!("ptr after cl map: {:?}", ptr);
        assert_eq!(ptr, ptr0 as *mut T);
        let v: Vec<T> = unsafe { Vec::from_raw_parts(ptr, n, n) };
        // take possession of the memory
        //let err = unsafe{ cl_sys::clRetainMemObject(buffer)};
        v
    }

    pub fn set_kernel_arg<T: TrueArg>(&mut self, kname: &String, index: usize, arg: &T) {
        let kernel = self.kernels.get(kname).unwrap();
        let smem = std::mem::size_of::<T>();
        let targ = arg.true_arg(self);
        let err = unsafe { cl_sys::clSetKernelArg(*kernel, index as u32, smem, targ) };
        assert_eq!(err, cl_sys::CL_SUCCESS);
    }

    pub fn run_kernel(&mut self, kname: &String, globsize: usize, locsize: usize) {

        let kernel = self.kernels.get(kname).unwrap();

        assert!(globsize % locsize == 0);

        let offset = 0;
        let err = unsafe {
            cl_sys::clEnqueueNDRangeKernel(
                self.queue,
                *kernel,
                1,
                &offset,
                &globsize,
                &locsize,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);

        let err = unsafe { cl_sys::clFinish(self.queue) };
        assert_eq!(err, cl_sys::CL_SUCCESS);
    }
}

impl Drop for Accel {
    fn drop(&mut self) {
        println!("MiniCL memory drop");
        for (ptr, (buffer, size, szf, is_map)) in self.buffers.iter() {
            if !is_map {
                let n = size / szf;
                assert!(size % szf == 0);
                println!("Free buffer {:?}", ptr);
                // give back the vector to Rust
                // who will free the memory at the end
                assert_eq!(std::mem::size_of::<u8>(), 1);
                let _v: Vec<u8> = unsafe { Vec::from_raw_parts(*ptr as *mut u8, n * szf, n * szf) };
            }
            let err = unsafe { cl_sys::clReleaseMemObject(*buffer) };
            assert!(err == cl_sys::CL_SUCCESS);
        }
        for (s, kernel) in self.kernels.iter() {
            println!("Free kernel {}", s);
            let err = unsafe { cl_sys::clReleaseKernel(*kernel) };
            assert!(err == cl_sys::CL_SUCCESS);
        }

        println!("Free MiniCL env.");
        let err = unsafe {
            cl_sys::clReleaseCommandQueue(self.queue)
                | cl_sys::clReleaseProgram(self.program)
                | cl_sys::clReleaseDevice(self.device)
                | cl_sys::clReleaseContext(self.context)
        };
        assert!(err == cl_sys::CL_SUCCESS);
    }
}

pub trait TrueArg {
    fn true_arg(&self, _dev: &Accel) -> *const cl_sys::c_void {
        self as *const _ as *const cl_sys::c_void
    }
}

impl TrueArg for *mut cl_sys::c_void {
    fn true_arg(&self, dev: &Accel) -> *const cl_sys::c_void {
        let (buffer, _size, _szf, is_map) = dev.buffers.get(self).unwrap();
        assert!(!is_map, "Buffer is mapped on the host");
        buffer as *const _ as *const cl_sys::c_void
    }
}

impl TrueArg for i32 {

}


#[macro_export]
macro_rules! kernel_set_args_and_run {
    ($dev: expr, $kname: expr, $globsize: expr, $locsize:expr, $($arg:expr),*) => {{
        println!("Device={:?}", $dev);
        println!("Kernel={:?}", $kname);
        println!("Glob. size={:?}", $globsize);
        println!("Loc. size={:?}", $locsize);
        let mut count = 0;
        $(
            println!("Arg {} = {:?}", count,$arg);
            $dev.set_kernel_arg(& $kname, count, & $arg);
            count += 1;
        )*
        $dev.run_kernel(& $kname, $globsize, $locsize);
    }}
}
