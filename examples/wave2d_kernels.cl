
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define real _real_

#define _NX _nx_
#define _NY _ny_
#define _DX _dx_
#define _DY _dy_
#define _DT _dt_

#define _C _cson_

#define M_PI (3.14159265358979323846264338328_F)


#define _VOL (_DX * _DY)

__constant int dir[4][2] = { {-1, 0}, {1, 0},
			     {0, -1}, {0, 1}};

__constant real ds[4] = { _DY, _DY, _DX, _DX };

#define _R (0._F)
#define _G ((1 - _R) /(1 + _R))



void exact_sol(real* x, real t, real* u){

  
  //*u = sin(10 * 2 * M_PI * (x[1] + x[0] - sqrt((real) 2) * _C * t));
  //*u=1;
  *u = 0;
  
}

void source(real *x, real t, real *s){

  real xm = 1;
  xm /= 4;
  
  real ym = 1;
  ym /= 4;

  real r = 1;
  r /= 10;

  real d = sqrt((x[0] - xm) * (x[0] - xm)
		+ (x[1] - ym) * (x[1] - ym));

  *s = 0;
  
  if (d < r){
    *s =  sin(10 * 2 * M_PI * (sqrt((real) 2) * _C * t));

  }

}


// initial condition on the macro data
__kernel void init_sol(__global real *un,__global real *unm1){

  int id = get_global_id(0);
  
  int i = id % _NX;
  int j = id / _NX;

  real unow;

  real t = 0;
  real xy[2] = {i * _DX, j * _DY};
  
  exact_sol(xy, t, &unow);
  
  int imem = i + j * _NX;

  un[imem] =  unow;
    //fn[imem] = j;

  t += _DT;
  
  exact_sol(xy, t, &unow);

  unm1[imem] = unow;
  
}



// one time step of the LBM scheme
__kernel void time_step(real tnow,
			__global const real *unm1,
			__global const real *un,
			__global real *unp1
			){

  real bx = _C * _DT / _DX; 
  real by = _C * _DT / _DY; 

  int id = get_global_id(0);
  
  int i = id % _NX;
  int j = id / _NX;

  real a = 1;

  if (i == 0 || i == _NX - 1)
    a = 1 / (1 + bx * _G);

  if (j == 0 || j == _NY - 1)
    a = 1 / (1 + by * _G);

  real u[4];
  
  for(int d = 0; d < 4; d++){
    //int iR = (i + dir[d][0] + _NX) % _NX;
    //int jR = (j + dir[d][1] + _NY) % _NY;
    int iR = i + dir[d][0];
    if (iR == -1) iR = 1;
    if (iR == _NX) iR = _NX - 2;
    
    int jR = j + dir[d][1];
    if (jR == -1) jR = 1;
    if (jR == _NY) jR = _NY - 2;

    int imem = iR + jR * _NX;
    u[d] = un[imem];
  }

  int imem = i + j * _NX;

  real xy[2] = {i * _DX, j * _DY};

  real s;
  source(xy, tnow, &s);
  
  unp1[imem] = (1 - 2 * a) * unm1[imem] +
    2 * a * (1 - bx * bx - by * by) * un[imem]
    + a * bx * bx * (u[0] + u[1]) + a * by * by * (u[2] + u[3])
    - _DT * _DT * a * s;

}


