#include "minicl.h"

#include <stdio.h>
#include <stdlib.h>

char* minicl_get_string(char* filename)
{
   char *buffer = NULL;
   int string_size;
   int read_size;
   FILE *handler = fopen(filename, "r");

   if (handler)
   {
       // Seek the last byte of the file
       fseek(handler, 0, SEEK_END);
       // Offset from the first to the last byte, or in other words, filesize
       string_size = ftell(handler);
       // go back to the start of the file
       rewind(handler);

       // Allocate a string that can hold it all
       buffer = (char*) malloc(sizeof(char) * (string_size + 1) );

       // Read it all in one operation
       read_size = fread(buffer, sizeof(char), string_size, handler);

       // fread doesn't set it so put a \0 in the last position
       // and buffer is now officially a string
       buffer[string_size] = '\0';

       if (string_size != read_size)
       {
           // Something went wrong, throw away the memory and set
           // the buffer to NULL
           free(buffer);
           buffer = NULL;
       }

       // Always remember to close the file.
       fclose(handler);
    }

    return buffer;
}

int minicl_device_init(minicl_device *dev, enum minicl_device_type accel_type, char *program){

// load a shared library created with
// gcc -c toto.c
// gcc -dynamiclib -fPIC -o toto.dylib toto.o
// then dlopen, dlsym and dlclose

}