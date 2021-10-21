#include "minicl.h"

#include <stdio.h>
#include <string.h>


int main()
{

    char *program = minicl_get_string("src/test_kernel_subs.cl");

    printf("%s\n", program);

    minicl_device dev = {0};

    minicl_device_init(&dev, CPU, program);

    size_t n = 100;

    float *a = calloc(n, sizeof(float));
    float *b = calloc(n, sizeof(float));
    float *c = calloc(n, sizeof(float));

    for(size_t i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = n - i;
    }

    minicl_buffer buf_a = {0};
    minicl_buffer buf_b = {0};
    minicl_buffer buf_c = {0};

    minicl_buffer_init(&buf_a, &dev, a, n, sizeof(float));
    minicl_buffer_init(&buf_b, &dev, b, n, sizeof(float));
    minicl_buffer_init(&buf_c, &dev, c, n, sizeof(float));

    size_t work_size = n;
    size_t group_size = 1;

    size_t nargs = 3;

    minicl_buffer *args = {&buf_a, &buf_b, &buf_c};

    minicl_device_call(&dev, "add_buffer", group_size, work_size, args, nargs);



    minicl_device_release(&dev);

    return 0;
}