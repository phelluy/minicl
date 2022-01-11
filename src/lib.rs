/// Minimal GPU compute library, based on OpenCL
#[derive(Debug)]
pub struct Accel {
    source: String,
    platform: cl_sys::cl_platform_id,
    context: cl_sys::cl_context,
    device: cl_sys::cl_device_id,
    program: cl_sys::cl_program,
    queue: cl_sys::cl_command_queue,
    kernel: cl_sys::cl_kernel,
}

impl Accel {
    /// Generate a minicl environment
    /// from an OpenCL source code
    pub fn new(oclsource: String) -> Accel {
        let mut platform: cl_sys::cl_platform_id = std::ptr::null_mut();
        let mut nb_platforms: u32 = 0;
        let mut device: cl_sys::cl_device_id = std::ptr::null_mut();
        let kernel: cl_sys::cl_kernel = std::ptr::null_mut();
        let err = unsafe { cl_sys::clGetPlatformIDs(1, &mut platform, &mut nb_platforms) };
        assert_eq!(err, cl_sys::CL_SUCCESS);

        println!("Found {} platform(s)", nb_platforms);
        let mut temp: u32 = 0;
        let err = unsafe {
            cl_sys::clGetDeviceIDs(
                platform,
                cl_sys::CL_DEVICE_TYPE_GPU,
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

        let mut err: i32 = 0;
        let kernel = unsafe {
            let name = std::ffi::CString::new("simple_add").unwrap();
            cl_sys::clCreateKernel(program, name.as_ptr(), &mut err)
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);

        Accel {
            source: oclsource,
            platform,
            context,
            device,
            program,
            queue,
            kernel,
        }
    }

    pub fn run_kernel(&mut self, mut v: Vec<i32>) -> Vec<i32> {
        v.shrink_to_fit();
        assert!(v.len() == v.capacity());
        let ptr = v.as_mut_ptr();
        let n = v.len();
        std::mem::forget(v);
        let szf = std::mem::size_of::<i32>();
        let mut err: i32 = 0;
        let buffer = unsafe {
            cl_sys::clCreateBuffer(
                self.context,
                cl_sys::CL_MEM_READ_WRITE | cl_sys::CL_MEM_USE_HOST_PTR,
                n * szf,
                ptr as *mut cl_sys::c_void,
                &mut err,
            )
        };

        let szf = std::mem::size_of::<cl_sys::cl_mem>();
        let err = unsafe {
            cl_sys::clSetKernelArg(self.kernel, 0, szf, buffer)
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);

        let global_size = n;
        let local_size = n;
        let offset = 0;
        let err = unsafe{
            cl_sys::clEnqueueNDRangeKernel(self.queue, self.kernel, 1, &offset, &global_size, 
            &local_size, 0, std::ptr::null(), std::ptr::null_mut())
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);
        let v = unsafe {Vec::from_raw_parts(ptr, n, n) };
        v
    }
}
