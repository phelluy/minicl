//! Minimal compute library for accessing OpenCL from Rust.
//!  With normal use, this library should prevent memory leaks
//! and most unsafety related to OpenCL.
//!
//! All the OpenCL things (device, context, buffers, etc.)
//!  are packed into a single Accelerator struct.
//!
//! # Examples
//! ```
//!let source = "__kernel  void simple_add(__global int *v, int x){
//!int i = get_global_id(0);
//!v[i] += x;
//!}"
//!.to_string();
//!let num_platform = 0;
//!let mut cldev = minicl::Accel::new(source, num_platform);
//!
//! // the used kernel has to be registered
//!let kname = "simple_add".to_string();
//!cldev.register_kernel(&kname);
//!
//!let n = 64;
//!
//! // the memory buffer shared with the
//! // accelerator has to be registered
//!let v: Vec<i32> = vec![12; n];
//!let v = cldev.register_buffer(v);
//!
//!let x: i32 = 1000;
//!let globsize = n;
//!let locsize = 16;
//!
//! // first kernel call
//!minicl::kernel_set_args_and_run!(cldev, kname, globsize, locsize, v, x);
//!
//! // map the buffer for access from the host
//!let v: Vec<i32> = cldev.map_buffer(v);
//!println!("First kernel run v={:?}", v);
//!
//! // unmap for giving it back to the device
//!let v = cldev.unmap_buffer(v);
//!
//! // next call
//!minicl::kernel_set_args_and_run!(cldev, kname, globsize, locsize, v, x);
//!
//!let v: Vec<i32> = cldev.map_buffer(v);
//!println!("Next kernel run v={:?}", v);
//!
//! ```
use std::collections::HashMap;
/// All the OpenCL things (device, context, buffers, etc.)
///  are packed into a single Accelerator struct.
#[derive(Debug)]
pub struct Accel {
    context: cl_sys::cl_context,
    device: cl_sys::cl_device_id,
    program: cl_sys::cl_program,
    queue: cl_sys::cl_command_queue,
    kernels: HashMap<String, cl_sys::cl_kernel>,
    buffers: HashMap<*mut cl_sys::c_void, (cl_sys::cl_mem, usize, usize, bool)>,
}

impl Accel {
    /// Generates a minicl environment
    /// from an OpenCL source code and a platform id.
    pub fn new(oclsource: String, numplat: usize) -> Accel {
        let platform: cl_sys::cl_platform_id = std::ptr::null_mut();
        let mut nb_platforms: u32 = 0;
        let mut device: cl_sys::cl_device_id = std::ptr::null_mut();
        let err = unsafe { cl_sys::clGetPlatformIDs(0, std::ptr::null_mut(), &mut nb_platforms) };
        assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));
        println!("Found {} platform(s)", nb_platforms);
        assert!(10 > nb_platforms);
        let mut platform = [platform; 10];
        let err =
            unsafe { cl_sys::clGetPlatformIDs(nb_platforms, &mut platform[0], &mut nb_platforms) };
        assert_eq!(err, cl_sys::CL_SUCCESS, "OpenCL error: {}", error_text(err));

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
        assert_eq!(err, cl_sys::CL_SUCCESS, "OpenCL error: {}", error_text(err));
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
        assert_eq!(err, cl_sys::CL_SUCCESS, "OpenCL error: {}", error_text(err));
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
        assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));

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
        assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));

        let mut err: i32 = 0;
        let queue = unsafe {
            cl_sys::clCreateCommandQueue(
                context,
                device,
                cl_sys::CL_QUEUE_PROFILING_ENABLE,
                &mut err,
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));

        let mut err: i32 = 0;
        let oclsource = std::ffi::CString::new(oclsource).unwrap();
        //let oclsources = [oclsource.as_ptr()];
        let program = unsafe {
            cl_sys::clCreateProgramWithSource(
                context,
                1,
                &(oclsource.as_ptr()) as *const *const cl_sys::libc::c_char,
                std::ptr::null(),
                &mut err,
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));

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
        assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));
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
        assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));
        let log = unsafe {
            std::ffi::CStr::from_ptr(log.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        println!("Build messages:\n-------------------------------------");
        println!("{}", log);
        println!("-------------------------------------");
        assert_eq!(errb, cl_sys::CL_SUCCESS, "{}", error_text(errb));

        Accel {
            context,
            device,
            program,
            queue,
            kernels: HashMap::new(),
            buffers: HashMap::new(),
        }
    }

    /// Registers a kernel, before it can be called.
    pub fn register_kernel(&mut self, name: &str) {
        assert!(
            !self.kernels.contains_key(name),
            "Kernel already registered."
        );
        let mut err: i32 = 0;
        let cname = std::ffi::CString::new(name.to_string()).unwrap();
        let kernel: cl_sys::cl_kernel =
            unsafe { cl_sys::clCreateKernel(self.program, cname.as_ptr(), &mut err) };
        assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));
        //println!("kernel={:?}", kernel);
        self.kernels.insert(name.to_string(), kernel);
    }

    /// Registers a buffer before it can be passed to a kernel.
    /// The buffer is automatically copied on the device. The memory is
    /// managed by OpenCL until the next map.
    /// It can not be accessed by the host until the next map.
    /// Returns a pointer to the memory zone managed by OpenCL.
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
        assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));
        let is_map = false;
        self.buffers.insert(ptr0, (buffer, n * szf, szf, is_map));
        ptr0
    }

    /// Unmaps back to the device a buffer mapped on the host.
    /// The buffer must have been registered before.
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
        assert!(is_map, "Buffer is already unmap.");
        //let (mut buffer, mut size, mut szf, mut is_map) = self.buffers.get(&ptr0).unwrap();
        self.buffers.remove(&ptr0).unwrap();
        //println!("ptr0 before cl buffer création: {:?}", ptr0);
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
        assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));
        let is_map = false;
        self.buffers.insert(ptr0, (buffer, size, szf, is_map));
        //println!("buffer={:?}", buffer);
        //self.buffers.insert(ptr0, (buffer, n * szf, szf));
        ptr0
    }

    /// Maps a buffer from the device to the host.
    /// Must be called before any access to the buffer
    /// from the host side.
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
        assert!(!is_map, "Buffer already mapped.");
        //let (mut buffer, mut size, mut szf, mut is_map) = self.buffers.get(&ptr0).unwrap();
        //println!("buffer={:?} size={} szf={}", buffer, size, szf);
        let ptr = unsafe {
            cl_sys::clEnqueueMapBuffer(
                self.queue,
                buffer,
                blocking,
                cl_sys::CL_MAP_READ | cl_sys::CL_MAP_WRITE,  
                0,
                size,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut err,
            )
        } as *mut T;
        assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));
        self.buffers.remove(&ptr0);
        let is_map = true;
        self.buffers.insert(ptr0, (buffer, size, szf, is_map));
        let n = size / szf;
        //println!("size={} szf={}", size, szf);
        assert!(size % szf == 0, "Possible type mistmatch.");
        //println!("ptr0 before cl map: {:?}", ptr0);
        //println!("ptr after cl map: {:?}", ptr);
        assert_eq!(ptr, ptr0 as *mut T);
        let v: Vec<T> = unsafe { Vec::from_raw_parts(ptr, n, n) };
        // take possession of the memory
        //let err = unsafe{ cl_sys::clRetainMemObject(buffer)};
        v
    }

    /// Defines kernel args value/location before kernel call.
    /// The args must implement the TrueArg traits, which converts
    /// the Rust arg type to the corresponding OpenCL type, with the
    /// same size.
    pub fn set_kernel_arg<T: TrueArg>(&mut self, kname: &str, index: usize, arg: &T) {
        let kernel = self.kernels.get(kname).unwrap();
        let smem = std::mem::size_of::<T>();
        let targ = arg.true_arg(self);
        let err = unsafe { cl_sys::clSetKernelArg(*kernel, index as u32, smem, targ) };
        assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));
    }

    /// Runs a kernel with given global size and local size.
    /// Before calling this function, it is necessay to set the kernel args.
    /// This can be achieved with the function [set_kernel_arg](Accel::set_kernel_arg).
    /// # Safety
    /// This function is not safe, because if mem buffers are mapped to the host
    /// and used by the kernel, this can produce undefined behavior.
    /// It is better to use the macro [kernel_set_args_and_run!](kernel_set_args_and_run!), which recheck all args.
    /// The measured overhead is generally very very small.
    pub unsafe fn run_kernel(&mut self, kname: &str, globsize: usize, locsize: usize) {
        let kernel = self.kernels.get(kname).unwrap();

        assert!(globsize % locsize == 0);

        let offset = 0;
        #[allow(unused_unsafe)]
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
        assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));

        #[allow(unused_unsafe)]
        let err = unsafe { cl_sys::clFinish(self.queue) };
        assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));
    }
}

/// OpenCL memory is managed in a C-like fashion.
/// All the buffers which have not yet been mapped back to the
/// host must be carefully given back to Rust.
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
            assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));
        }
        for (s, kernel) in self.kernels.iter() {
            println!("Free kernel {}", s);
            let err = unsafe { cl_sys::clReleaseKernel(*kernel) };
            assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));
        }

        println!("Free MiniCL env.");
        let err = unsafe {
            cl_sys::clReleaseCommandQueue(self.queue)
                | cl_sys::clReleaseProgram(self.program)
                | cl_sys::clReleaseDevice(self.device)
                | cl_sys::clReleaseContext(self.context)
        };
        assert_eq!(err, cl_sys::CL_SUCCESS, "{}", error_text(err));
    }
}

/// Pointer type conversion from Rust type to C type.
/// For most cases, the default conversion is OK: simply
/// converts the ref to a C void pointer.
pub trait TrueArg {
    fn true_arg(&self, _dev: &Accel) -> *const cl_sys::c_void {
        self as *const _ as *const cl_sys::c_void
    }
}

/// Nothing to do for basic types.
impl TrueArg for i32 {}
impl TrueArg for u32 {}
impl TrueArg for f32 {}
impl TrueArg for f64 {}


/// Pointer conversion for buffer. We provide additional
/// checks for better safety: the buffer must be registered
/// and not currently mapped to the host.
impl TrueArg for *mut cl_sys::c_void {
    fn true_arg(&self, dev: &Accel) -> *const cl_sys::c_void {
        let (buffer, _size, _szf, is_map) = dev.buffers.get(self).unwrap();
        assert!(!is_map, "Buffer is mapped on the host.");
        buffer as *const _ as *const cl_sys::c_void
    }
}


/// This macro helps to run a kernel the first time, by simplifying
/// the definition of the kernel args.
/// For the next calls, it is possible to use [run_kernel](Accel::run_kernel)
/// if the args are not changed. But it is better to use this macro which recheck all args.
/// The measured overhead is generally very very small.
/// # Safety
/// Calling an OpenCL kernel is not safe. A bug in the C code of the kernel 
/// can lead to a segfault for instance.
#[macro_export]
macro_rules! kernel_set_args_and_run {
    ($dev: expr, $kname: expr, $globsize: expr, $locsize:expr, $($arg:expr),*) => {{
        // println!("Device={:?}", $dev);
        // println!("Kernel={:?}", $kname);
        // println!("Glob. size={:?}", $globsize);
        // println!("Loc. size={:?}", $locsize);
        let mut count = -1;
        $(
            // println!("Arg {} = {:?}", count,$arg);
            count +=1;
            $dev.set_kernel_arg(& $kname, count as usize, & $arg);
        )*
        unsafe { $dev.run_kernel(& $kname, $globsize, $locsize)};
    }}
}

pub fn error_text(error_code: cl_sys::cl_int) -> &'static str {
    match error_code {
        cl_sys::CL_SUCCESS => "CL_SUCCESS",
        cl_sys::CL_DEVICE_NOT_FOUND => "CL_DEVICE_NOT_FOUND",
        cl_sys::CL_DEVICE_NOT_AVAILABLE => "CL_DEVICE_NOT_AVAILABLE",
        cl_sys::CL_COMPILER_NOT_AVAILABLE => "CL_COMPILER_NOT_AVAILABLE",
        cl_sys::CL_MEM_OBJECT_ALLOCATION_FAILURE => "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        cl_sys::CL_OUT_OF_RESOURCES => "CL_OUT_OF_RESOURCES",
        cl_sys::CL_OUT_OF_HOST_MEMORY => "CL_OUT_OF_HOST_MEMORY",
        cl_sys::CL_PROFILING_INFO_NOT_AVAILABLE => "CL_PROFILING_INFO_NOT_AVAILABLE",
        cl_sys::CL_MEM_COPY_OVERLAP => "CL_MEM_COPY_OVERLAP",
        cl_sys::CL_IMAGE_FORMAT_MISMATCH => "CL_IMAGE_FORMAT_MISMATCH",
        cl_sys::CL_IMAGE_FORMAT_NOT_SUPPORTED => "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        cl_sys::CL_BUILD_PROGRAM_FAILURE => "CL_BUILD_PROGRAM_FAILURE",
        cl_sys::CL_MAP_FAILURE => "CL_MAP_FAILURE",
        cl_sys::CL_MISALIGNED_SUB_BUFFER_OFFSET => "CL_MISALIGNED_SUB_BUFFER_OFFSET",
        cl_sys::CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST => {
            "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"
        }
        cl_sys::CL_COMPILE_PROGRAM_FAILURE => "CL_COMPILE_PROGRAM_FAILURE",
        cl_sys::CL_LINKER_NOT_AVAILABLE => "CL_LINKER_NOT_AVAILABLE",
        cl_sys::CL_LINK_PROGRAM_FAILURE => "CL_LINK_PROGRAM_FAILURE",
        cl_sys::CL_DEVICE_PARTITION_FAILED => "CL_DEVICE_PARTITION_FAILED",
        cl_sys::CL_KERNEL_ARG_INFO_NOT_AVAILABLE => "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",
        cl_sys::CL_INVALID_VALUE => "CL_INVALID_VALUE",
        cl_sys::CL_INVALID_DEVICE_TYPE => "CL_INVALID_DEVICE_TYPE",
        cl_sys::CL_INVALID_PLATFORM => "CL_INVALID_PLATFORM",
        cl_sys::CL_INVALID_DEVICE => "CL_INVALID_DEVICE",
        cl_sys::CL_INVALID_CONTEXT => "CL_INVALID_CONTEXT",
        cl_sys::CL_INVALID_QUEUE_PROPERTIES => "CL_INVALID_QUEUE_PROPERTIES",
        cl_sys::CL_INVALID_COMMAND_QUEUE => "CL_INVALID_COMMAND_QUEUE",
        cl_sys::CL_INVALID_HOST_PTR => "CL_INVALID_HOST_PTR",
        cl_sys::CL_INVALID_MEM_OBJECT => "CL_INVALID_MEM_OBJECT",
        cl_sys::CL_INVALID_IMAGE_FORMAT_DESCRIPTOR => "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        cl_sys::CL_INVALID_IMAGE_SIZE => "CL_INVALID_IMAGE_SIZE",
        cl_sys::CL_INVALID_SAMPLER => "CL_INVALID_SAMPLER",
        cl_sys::CL_INVALID_BINARY => "CL_INVALID_BINARY",
        cl_sys::CL_INVALID_BUILD_OPTIONS => "CL_INVALID_BUILD_OPTIONS",
        cl_sys::CL_INVALID_PROGRAM => "CL_INVALID_PROGRAM",
        cl_sys::CL_INVALID_PROGRAM_EXECUTABLE => "CL_INVALID_PROGRAM_EXECUTABLE",
        cl_sys::CL_INVALID_KERNEL_NAME => "CL_INVALID_KERNEL_NAME",
        cl_sys::CL_INVALID_KERNEL_DEFINITION => "CL_INVALID_KERNEL_DEFINITION",
        cl_sys::CL_INVALID_KERNEL => "CL_INVALID_KERNEL",
        cl_sys::CL_INVALID_ARG_INDEX => "CL_INVALID_ARG_INDEX",
        cl_sys::CL_INVALID_ARG_VALUE => "CL_INVALID_ARG_VALUE",
        cl_sys::CL_INVALID_ARG_SIZE => "CL_INVALID_ARG_SIZE",
        cl_sys::CL_INVALID_KERNEL_ARGS => "CL_INVALID_KERNEL_ARGS",
        cl_sys::CL_INVALID_WORK_DIMENSION => "CL_INVALID_WORK_DIMENSION",
        cl_sys::CL_INVALID_WORK_GROUP_SIZE => "CL_INVALID_WORK_GROUP_SIZE",
        cl_sys::CL_INVALID_WORK_ITEM_SIZE => "CL_INVALID_WORK_ITEM_SIZE",
        cl_sys::CL_INVALID_GLOBAL_OFFSET => "CL_INVALID_GLOBAL_OFFSET",
        cl_sys::CL_INVALID_EVENT_WAIT_LIST => "CL_INVALID_EVENT_WAIT_LIST",
        cl_sys::CL_INVALID_EVENT => "CL_INVALID_EVENT",
        cl_sys::CL_INVALID_OPERATION => "CL_INVALID_OPERATION",
        cl_sys::CL_INVALID_GL_OBJECT => "CL_INVALID_GL_OBJECT",
        cl_sys::CL_INVALID_BUFFER_SIZE => "CL_INVALID_BUFFER_SIZE",
        cl_sys::CL_INVALID_MIP_LEVEL => "CL_INVALID_MIP_LEVEL",
        cl_sys::CL_INVALID_GLOBAL_WORK_SIZE => "CL_INVALID_GLOBAL_WORK_SIZE",
        cl_sys::CL_INVALID_PROPERTY => "CL_INVALID_PROPERTY",
        cl_sys::CL_INVALID_IMAGE_DESCRIPTOR => "CL_INVALID_IMAGE_DESCRIPTOR",
        cl_sys::CL_INVALID_COMPILER_OPTIONS => "CL_INVALID_COMPILER_OPTIONS",
        cl_sys::CL_INVALID_LINKER_OPTIONS => "CL_INVALID_LINKER_OPTIONS",
        cl_sys::CL_INVALID_DEVICE_PARTITION_COUNT => "CL_INVALID_DEVICE_PARTITION_COUNT",
        cl_sys::CL_INVALID_PIPE_SIZE => "CL_INVALID_PIPE_SIZE",
        cl_sys::CL_INVALID_DEVICE_QUEUE => "CL_INVALID_DEVICE_QUEUE",
        cl_sys::CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR => "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR",
        cl_sys::CL_PLATFORM_NOT_FOUND_KHR => "CL_PLATFORM_NOT_FOUND_KHR",
        _ => "UNKNOWN_ERROR",
    }
}

// some unit tests
#[test]
fn test_init() {
    let source = "__kernel  void simple_kernel(){
        int i = get_global_id(0);
    }"
    .to_string();

    let dev = Accel::new(source, 0);
    println!("{:?}", dev);
}

#[test]
fn test_buffer() {
    let source = "__kernel  void simple_kernel(void){
        int i = get_global_id(0);
    }"
    .to_string();

    let mut dev = Accel::new(source, 0);
    let v: Vec<i32> = vec![3; 16];
    let v0 = v.clone();
    let v = dev.register_buffer(v);
    let v = dev.map_buffer(v);
    assert_eq!(v0, v);
}

#[test]
fn test_kernel() {
    let source = "__kernel  void simple_add(__global int *v, int x){
        int i = get_global_id(0);
        v[i] += x;
    }"
    .to_string();

    let kernel_name = "simple_add".to_string();
    let mut dev = Accel::new(source, 0);
    dev.register_kernel(&kernel_name);
    let v: Vec<i32> = vec![3; 16];
    let vp: Vec<i32> = vec![6; 16];
    let v = dev.register_buffer(v);
    let x = 3;
    kernel_set_args_and_run!(dev, kernel_name, 16, 4, v, x);
    let v = dev.map_buffer(v);
    assert_eq!(vp, v);
}
