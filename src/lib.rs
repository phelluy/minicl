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
        let mut z = unsafe { std::mem::uninitialized()};
        let mut platform: cl_sys::cl_platform_id = z;
        let mut nb_platforms: u32 = 0;
        let mut context: cl_sys::cl_context = z;
        let mut device: cl_sys::cl_device_id =z;
        let mut program: cl_sys::cl_program = z;
        let mut queue: cl_sys::cl_command_queue = z;
        let mut kernel: cl_sys::cl_kernel= z;
        unsafe {
            cl_sys::clGetPlatformIDs(1, &mut platform, &mut nb_platforms);
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
