#include "minicl.h"

#include <stdio.h>
#include <string.h>

int main()
{

    char *program = minicl_get_string("src/test_kernel_subs.cl");

    char *mystring = "coucou";

    printf("%s\n %s\n", program, mystring);

    minicl_device dev = {0};

    minicl_device_init(&dev, CPU, program);

    size_t n = 100;

    float *a = calloc(n, sizeof(float));
    float *b = calloc(n, sizeof(float));
    float *c = calloc(n, sizeof(float));

    for (size_t i = 0; i < n; ++i)
    {
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

    minicl_buffer *args[3];
    args[0] = &buf_a;
    args[1] = &buf_b;
    args[2] = &buf_c;

    minicl_device_call(&dev, "add_buffer", group_size, work_size, args, nargs);

    minicl_buffer_pull(&buf_c);

    for (size_t i = 0; i < n; ++i)
    {
    }

    minicl_device_release(&dev);

    return 0;
}