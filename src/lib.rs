/// Minimal GPU compute library, based on OpenCL
use std::collections::HashMap;
#[derive(Debug)]
pub struct Accel {
    source: String,
    platform: cl_sys::cl_platform_id,
    context: cl_sys::cl_context,
    device: cl_sys::cl_device_id,
    program: cl_sys::cl_program,
    queue: cl_sys::cl_command_queue,
    kernels: HashMap<String, cl_sys::cl_kernel>,
    buffers: HashMap<String,(cl_sys::cl_mem,usize,usize)>,
}

impl Accel {
    /// Generate a minicl environment
    /// from an OpenCL source code
    pub fn new(oclsource: String) -> Accel {
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
        use std::io::{stdin, stdout, Write};
        let mut s = String::new();
        println!("Enter platform num.");
        stdin()
            .read_line(&mut s)
            .expect("Did not enter a correct string");
        let input: usize = s.trim().parse().expect("Wanted a number");
        let numplat = input;
        assert!(numplat < nb_platforms as usize);

        let mut size = 1000;
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
            source: oclsource,
            platform: platform[numplat],
            context,
            device,
            program,
            queue,
            kernels: HashMap::new(),
            buffers: HashMap::new(),
        }
    }

    pub fn register_kernel(&mut self, name: String) {
        let mut err: i32 = 0;
        let cname = std::ffi::CString::new(name.clone()).unwrap();
        let kernel:cl_sys::cl_kernel = unsafe {
            cl_sys::clCreateKernel(self.program, cname.as_ptr(), &mut err)
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);
        println!("kernel={:?}",kernel);
        self.kernels.insert(name,kernel);        
    }

    pub fn register_buffer<T>(&mut self, name: String, mut v:Vec<T>) {
        v.shrink_to_fit();
        assert!(v.len() == v.capacity());
        let ptr0 = v.as_mut_ptr();
        println!("ptr0 before cl buffer création: {:?}", ptr0);
        let n = v.len();
        std::mem::forget(v);
        let szf = std::mem::size_of::<T>();
        println!("size={}",n*szf);
        let mut err: i32 = 0;
        let buffer : cl_sys::cl_mem = unsafe {
            cl_sys::clCreateBuffer(
                self.context,
                cl_sys::CL_MEM_READ_WRITE | cl_sys::CL_MEM_USE_HOST_PTR,
                n * szf,
                ptr0 as *mut cl_sys::c_void,
                &mut err,
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);
        println!("buffer={:?}",buffer);
        self.buffers.insert(name,(buffer,n*szf,szf));        
    }

    pub fn take_buffer<T>(&mut self, name: String) -> Vec<T> {
        let mut err = 0;
        let blocking = cl_sys::CL_TRUE;
        let szf = std::mem::size_of::<T>();
        //let toto = self.buffers.get(&name).unwrap();
        let (buffer, size,_) = self.buffers.get(&name).unwrap();
        println!("buffer={:?} size={} szf={}",*buffer,size,szf);
        let ptr = unsafe {
            cl_sys::clEnqueueMapBuffer(
                self.queue,
                *buffer,
                blocking,
                cl_sys::CL_MAP_READ,
                0,
                *size,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut err,
            )
        } as *mut T;
        assert_eq!(err, cl_sys::CL_SUCCESS);
        let n = size / szf;
        println!("size={} szf={}",size,szf);
        assert!(size%szf == 0);
        println!("ptr after cl map: {:?}", ptr);
        let v : Vec<T> = unsafe { Vec::from_raw_parts(ptr, n, n) };
        // take possession of the memory
        //let err = unsafe{ cl_sys::clRetainMemObject(buffer)};
        v
    }


    pub fn run_kernel(&mut self, kname: String, vname: String) {

        let smem = std::mem::size_of::<cl_sys::cl_mem>();
        let kernel = self.kernels.get(&kname).unwrap();
        let (buffer,size,szf) = self.buffers.get(&vname).unwrap();
        let szf = *szf;
        println!("buffer={:?} szf={}",*buffer,szf);
        let err = unsafe {
            cl_sys::clSetKernelArg(
                *kernel,
                0,
                smem,
                buffer as *const _ as *const cl_sys::c_void,
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);
        let n = size/szf;
        assert!(size%szf == 0);

        let global_size = n;
        let local_size = n;
        let offset = 0;
        let err = unsafe {
            cl_sys::clEnqueueNDRangeKernel(
                self.queue,
                *kernel,
                1,
                &offset,
                &global_size,
                &local_size,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);

        let err = unsafe {
            cl_sys::clFinish(self.queue)
        };
        assert_eq!(err, cl_sys::CL_SUCCESS);

    }
}
