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
}
