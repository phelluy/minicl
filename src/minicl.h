// minimal implementation of a compute library
// in the spirit of OpenCL, with essential features
#include <stdbool.h>

#ifndef MINICL_H
#define MINICL_H

enum minicl_device_type
{
    CPU,
    OPENCL,
    METAL
};

typedef struct minicl_device
{

    // type of accelerator
    enum minicl_device_type accel_type;

    // an opaque pointer to
    // the boilerplate things that
    // any compute library needs...
    void *config;

    // a string containing the source code
    // of the computing kernels
    char *source;

} minicl_device;

typedef struct minicl_buffer
{

    // a pointer to the data
    void *buf_data;

    // number of data
    size_t buf_len;

    // size of the type
    // the occupoied memory is buf_len * type_size
    size_t type_size;

    // the device on which the buffer is created
    minicl_device *dev;

    // opaque pointer to the copied data
    // inside the accelerator
    // the exact type depend on the underlying API (opencl, metal, etc.)
    void *buf_data_copy;

} minicl_buffer;

// utility for reading a file in a string
char* minicl_get_string(char* filename);

// unless stated, all functions return 0 if success or an error code.

// initialization of the device of a given type with a source code
int minicl_device_init(minicl_device *dev, minicl_device_type accel_type, char *program);

// cleanly release the device
int minicl_device_release(minicl_device *dev);

// create a buffer on the accelerator from given data
int minicl_buffer_init(minicl_buffer *buf, minicl_device *dev,
                       void *data, size_t len, size_t type_size);

// push/pull the buffer to/from accelerator
int minicl_buffer_push(minicl_buffer *buf);
int minicl_buffer_pull(minicl_buffer *buf);

// cleanly release the buffer
int minicl_buffer_release(minicl_buffer *buf);

// launch the kernel with name kernel_name
// with work_size work items
// and group_size work groups
// passing a list of nargs buffers to the kernel
int minicl_device_call(minicl_device *dev, char *kernel_name,
                       size_t group_size, size_t work_size,
                       minicl_buffer **args, size_t nargs);

#endif