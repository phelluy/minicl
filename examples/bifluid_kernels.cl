//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define real _real_
#define _NX _nx_
#define _NY _ny_
#define _DX _dx_
#define _DY _dy_
#define _DT _dt_
#define _M _m_
#define _N _n_
#define _VX _vx_
#define _VY _vy_
#define _NCPR _ncpr_

#define _LAMBDA _lambda_

#ifndef M_PI
#define M_PI (3.14159265358979323846264338328)
#endif

#define _VOL (_DX * _DY)

#define _C 10
#define _RHO0 1.
#define _LX (_nx_*_dx_)
#define _LY (_ny_*_dy_)

#define GRAVITY 9.81
#define rhoAir 1
#define rhoWater 1000

//#define dirichlet
//#define dirichlet_updown
//#define dirichlet_leftright

__constant real cylinderCenter[2] = {0.073, 0.05};
__constant real cylinderRadius = 0.01;

// this function swaps two integers


void flux_phy(const real *W, const real *vn, real *flux);

__constant int dir[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

__constant real ds[4] = {_DY, _DY, _DX, _DX};


void flux_phy(const real *w, const real *vn, real *flux){

  real un = (w[0] * vn[0] + w[1] * vn[1])/w[0];

  real theta = (rhoWater-rhoAir)/w[0];

  real phi = w[3]/w[0];

  real A = theta;
  real B = 1-theta;
  real C = -phi;
  real alpha;
  if(B > 0)
  {
    real q = -0.5*(B + sqrt(B*B - 4*A*C));
    alpha = C/q;
  }
  else
  {
    real q = -0.5*(B - sqrt(B*B - 4*A*C));
    alpha = q/A;
  }

  real p = _C*_C*(w[0] - (phi * rhoAir + (1-phi) * rhoWater));

  flux[0] = w[0]*un;
  flux[1] = w[1]*un + p*vn[0];
  flux[2] = w[2]*un + p*vn[1];
  flux[3] = w[3]*un;

}

// equilibrium "maxwellian" from macro data w
void w2f(const real *w, real *f) {
  for (int d = 0; d < 4; d++) {
    real flux[_M];
    real vnorm[3] = {(real)dir[d][0], (real)dir[d][1], (real)0};
    flux_phy(w, vnorm, flux);
    for (int iv = 0; iv < _M; iv++) {
      f[d * _M + iv] =  w[iv] / 4 + flux[iv] / 2 / _LAMBDA;
    }
  }
}

// macro data w from micro data f
void f2w(const real *f, real *w) {
  for (int iv = 0; iv < _M; iv++)
    w[iv] = 0;
  for (int d = 0; d < 4; d++) {
    for (int iv = 0; iv < _M; iv++) {
      w[iv] += f[d * _M + iv];
    }
  }
}

real distanceToCylinderCenter(real Rcoord[2])
{
  real distance = 0.;

  for(int d=0;d<2;++d)
  {
    distance += pow(Rcoord[d]-cylinderCenter[d], 2);
  }
  distance = sqrt(distance);

  return distance;
}

int isInLimit(int Icoord[2], int gridSize[2])
{
  for(int d=0;d<2;++d)
  {
    if(Icoord[d] == 0 || Icoord[d] == gridSize[d]-1)
    {
      return 1;
    }
  }

  return 0;
}

bool isInCylinder(real Rcoord[2])
{
  return distanceToCylinderCenter(Rcoord)<cylinderRadius;
}

int mask(real Rcoord[2], int Icoord[2], int gridSize[2])
{
  return isInLimit(Icoord, gridSize) || isInCylinder(Rcoord);
}

real sigmoid(real v)
{
  return 1/(1+exp(v));
}

// init_data
void exact_sol(real *w, real t, const real *x) {

  real rhoInWater = rhoWater;//*exp(-GRAVITY/_C/_C*x[1]);
  real rhoInAir = rhoAir;//*exp(-GRAVITY/_C/_C*x[1]);
  real phiInWater = 0;
  real phiInAir = 1;//*rhoInAir;

  real waterProportion = 0;
  if(isInCylinder(x))
  {
    waterProportion = 1;
  }
  waterProportion = sigmoid((distanceToCylinderCenter(x)-cylinderRadius)*500);

  w[0] = (1-waterProportion)*rhoInAir + waterProportion*rhoInWater;
  w[1] = 0.0;
  w[2] = 0.0;
  w[3] = (1-waterProportion)*phiInAir + waterProportion*phiInWater;

}

void applySource(real *w, int *Icoord, real *Rcoord, int *gridSize)
{
/*  real mu=isInCylinder(Rcoord);
  real wBar[_M];
  exact_sol(wBar, 0, Rcoord);
  real U[2] = {w[1]/w[0], w[2]/w[0]};
  real Ubar[2] = {wBar[1]/wBar[0], wBar[2]/wBar[0]};
  real Ustar[2] = {mu*Ubar[0] + (1-mu)*U[0], mu*Ubar[1] + (1-mu)*U[1]};
  w[1] = w[0]*Ustar[0];
  w[2] = w[0]*Ustar[1];*/

  real dt = _DT;
  //w[2] -= dt*GRAVITY*w[0];
}

// initial condition on the macro data
__kernel void init_sol(__global real *fn) {

  int id = get_global_id(0);

  int i = id % _NX;
  int j = id / _NX;

  int ngrid = _NX * _NY;

  real wnow[_M];

  real t = 0;
  real xy[2] = {i * _DX + _DX / 2, j * _DY + _DY / 2};

  // exact_sol(xy, t, wnow);
  exact_sol(wnow, t, xy);

  real fnow[_N];
  w2f(wnow, fnow);

  // printf("x=%f, y=%f \n",xy[0],xy[1]);
  // load middle value
  for (int ik = 0; ik < _N; ik++) {
    int imem = i + j * _NX + ik * ngrid;
    fn[imem] = fnow[ik];
    // fn[imem] = j;
  }
}

// one time step of the LBM scheme
__kernel void time_step(__global const real *fn, __global real *fnp1) {

  int id = get_global_id(0);

  int i = id % _NX;
  int j = id / _NX;

  int ngrid = _NX * _NY;

  real fnow[_N];

  // shift of values in domain
  for (int d = 0; d < 4; d++) {
    int iR = (i - dir[d][0] + _NX) % _NX;
    int jR = (j - dir[d][1] + _NY) % _NY;

    for (int iv = 0; iv < _M; iv++) {
      int ik = d * _M + iv;
      int imem = iR + jR * _NX + ik * ngrid;
      // #ifdef dirichlet_updown
      //       // dirichlet condition on up and down borders
      //       // (values of border cells are unchanged)
      //       if ((j == 0) || (j == _NY - 1)) {
      //         imem = i + j * _NX + ik * ngrid;
      //       }
      // #elif defined dirichlet_leftright
      //       // dirichlet condition on left and right borders
      //       // (values of border cells are unchanged)
      //       if ((i == 0) || (i == _NX - 1)) {
      //         imem = i + j * _NX + ik * ngrid;
      //       }
      // #elif defined dirichlet
      //       // dirichlet condition on all borders
      //       // (values of border cells are unchanged)
      //       if ((i == 0) || (i == _NX - 1) || (j == 0) || (j == _NY - 1)) {
      //         imem = i + j * _NX + ik * ngrid;
      //       }
      // #endif
      fnow[ik] = fn[imem];
    }
  }

  real xy[2] = {i * _DX + _DX / 2, j * _DY + _DY / 2};
  int iCoord[2] = {i,j};
  int gridSize[2] = {_NX, _NY};

  real wnow[_M];
  f2w(fnow + 0, wnow + 0);

  applySource(wnow, iCoord, xy, gridSize);

  real fnext[_N];
  //first order relaxation
  w2f(wnow + 0, fnext + 0);

  real om = 1.0;
  // second order relaxation
  for (int iv = 0; iv < _M; iv++) {
    for (int d = 0; d < 4; d++) {
      int ik = d * _M + iv;
      // if (iv == ePsi) {
      // fnext[ik] = 1.0 * fnext[ik];
      fnext[ik] = om * fnext[ik] - (om - 1) * fnow[ik];
      //fnext[ik] = fnow[ik];
      //} else {
      // fnext[ik] = 1.9 * fnext[ik] - 0.9 * fnow[ik];
      // fnext[ik] = 2.0 * fnext[ik] - 1.0 * fnow[ik];
      // fnext[ik] = 1.0 * fnext[ik] - 0.0 * fnow[ik];
      //}
    }
  }

  // save
  for (int ik = 0; ik < _N; ik++) {
    int imem = i + j * _NX + ik * ngrid;
    fnp1[imem] = fnext[ik];
    //fnp1[imem] = fnow[ik];
  }
}


// one time step of the LBM scheme
__kernel void time_shift(__global const real *fn, __global real *fnp1) {

  int id = get_global_id(0);

  int i = id % _NX;
  int j = id / _NX;

  int ngrid = _NX * _NY;

  real fnow[_N];

  // shift of values in domain
  for (int d = 0; d < 4; d++) {
    int iR = (i - dir[d][0] + _NX) % _NX;
    int jR = (j - dir[d][1] + _NY) % _NY;

    for (int iv = 0; iv < _M; iv++) {
      int ik = d * _M + iv;
      int imem = iR + jR * _NX + ik * ngrid;
      // #ifdef dirichlet_updown
      //       // dirichlet condition on up and down borders
      //       // (values of border cells are unchanged)
      //       if ((j == 0) || (j == _NY - 1)) {
      //         imem = i + j * _NX + ik * ngrid;
      //       }
      // #elif defined dirichlet_leftright
      //       // dirichlet condition on left and right borders
      //       // (values of border cells are unchanged)
      //       if ((i == 0) || (i == _NX - 1)) {
      //         imem = i + j * _NX + ik * ngrid;
      //       }
      // #elif defined dirichlet
      //       // dirichlet condition on all borders
      //       // (values of border cells are unchanged)
      //       if ((i == 0) || (i == _NX - 1) || (j == 0) || (j == _NY - 1)) {
      //         imem = i + j * _NX + ik * ngrid;
      //       }
      // #endif
      fnow[ik] = fn[imem];
    }
  }

  real fnext[_N];

  // transfer
  for (int iv = 0; iv < _M; iv++) {
    for (int d = 0; d < 4; d++) {
      int ik = d * _M + iv;
      // if (iv == ePsi) {
      // fnext[ik] = 1.0 * fnext[ik];
      fnext[ik] = fnow[ik];
      //fnext[ik] = fnow[ik];
      //} else {
      // fnext[ik] = 1.9 * fnext[ik] - 0.9 * fnow[ik];
      // fnext[ik] = 2.0 * fnext[ik] - 1.0 * fnow[ik];
      // fnext[ik] = 1.0 * fnext[ik] - 0.0 * fnow[ik];
      //}
    }
  }

  // save
  for (int ik = 0; ik < _N; ik++) {
    int imem = i + j * _NX + ik * ngrid;
    fnp1[imem] = fnext[ik];
    //fnp1[imem] = fnow[ik];
  }
}



