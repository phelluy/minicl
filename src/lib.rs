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
        let z = unsafe { std::mem::uninitialized() };
        let mut platform: cl_sys::cl_platform_id = z;
        let mut nb_platforms: u32 = 0;
        let mut device: cl_sys::cl_device_id = z;
        let kernel: cl_sys::cl_kernel = z;
        unsafe {
            let err = cl_sys::clGetPlatformIDs(1, &mut platform, &mut nb_platforms);
            assert_eq!(err, cl_sys::CL_SUCCESS);
        }

        println!("Found {} platform(s)", nb_platforms);
        unsafe {
            let mut temp: u32 = 0;
            let err = cl_sys::clGetDeviceIDs(
                platform,
                cl_sys::CL_DEVICE_TYPE_GPU,
                1,
                &mut device,
                &mut temp,
            );
            assert_eq!(err, cl_sys::CL_SUCCESS);
        }

        let context = unsafe {
            let mut err: i32 = 0;
            let context = cl_sys::clCreateContext(
                std::ptr::null(),
                1,
                &device,
                None,
                std::ptr::null_mut(),
                &mut err,
            );
            assert_eq!(err, cl_sys::CL_SUCCESS);
            context
        };

        let queue = unsafe {
            let mut err = 0;
            let queue = cl_sys::clCreateCommandQueue(
                context,
                device,
                cl_sys::CL_QUEUE_PROFILING_ENABLE,
                &mut err,
            );
            assert_eq!(err, cl_sys::CL_SUCCESS);
            queue
        };

        let program = unsafe {
            let mut err: i32 = 0;
            let oclsources = [oclsource.as_str()];
            let program = cl_sys::clCreateProgramWithSource(
                context,
                1,
                oclsources.as_ptr() as *const *const cl_sys::libc::c_char,
                std::ptr::null(),
                &mut err,
            );
            assert_eq!(err, cl_sys::CL_SUCCESS);
            program
        };

        unsafe {
            let opt = "";
            let mut log = [" "; 10000]; 
            let err = cl_sys::clBuildProgram(
                program,
                1,
                &device,
                opt.as_ptr() as *const cl_sys::libc::c_char,
                None,
                log.as_ptr() as *mut cl_sys::c_void,
            );
            //println!("{:?}",log);
            assert_eq!(err, cl_sys::CL_SUCCESS);
        }

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
