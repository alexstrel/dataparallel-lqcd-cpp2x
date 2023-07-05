#pragma once

#define TMP  gauge(xm1,y,0)
#define TMP2 gauge(xm1,y,0,other_parity)

template<typename Float>
void Dpsi(auto &out_spinor, const auto &in_spinor, const auto &gauge_field, const Float mass, const Float r, const int nx, const int ny) {
  
  Float constant = (mass + 2*r);
  
  Float fwd_bc = 1.0;
  Float bwd_bc = 1.0;  
  
  //Sum over 0,1 directions at each lattice site.
  int Nx = nx;
  int Ny = ny;
  int xp1, xm1, yp1, ym1;
  
  std::complex<Float> tmp = std::complex<Float>(0.);
  
  std::complex<Float> I   = std::complex<Float>(0., 1.0);
  std::complex<Float> H   = std::complex<Float>(0.5, 0.);  
  
  auto out          = out_spinor.ParityAccessor();
  const auto in     = in_spinor.ParityAccessor();
  const auto gauge  = gauge_field.ParityAccessor();  
  
  //#pragma omp parallel for collapse(2)
  for(int y=0; y<Ny; y++) {
    yp1 = (y+1) == Ny ? 0    : (y+1);
    ym1 = (y-1) == -1 ? Ny-1 : (y-1);

    if(yp1 == 0) fwd_bc = -1.0; 
    if(y   == 0) bwd_bc = -1.0;  

    for(int x=0; x<Nx; x++) {
      xp1 = (x+1) == Nx ? 0    : (x+1);
      xm1 = (x-1) == -1 ? Nx-1 : (x-1);
      
      //upper
      //(m0 + 2) * I_{X}
      tmp = constant * in(x,y,0)
	// mu = 1
	- H*(gauge(x,y,0) * (in(xp1,y,0) - in(xp1,y,1)) + conj(gauge(xm1,y,0)) * (in(xm1,y,0) + in(xm1,y,1)))
	
	// mu = 2
	- H*(gauge(x,y,1) * fwd_bc*(in(x,yp1,0) + I*in(x,yp1,1)) + conj(gauge(x,ym1,1)) * bwd_bc*(in(x,ym1,0) - I*in(x,ym1,1)));
      out(x,y,0) = tmp;
      
      //lower
      //(m0 + 2) * I_{X}
      tmp = constant * in(x,y,1)
	- H*(gauge(x,y,0) * (in(xp1,y,1) - in(xp1,y,0)) + conj(gauge(xm1,y,0)) * (in(xm1,y,1) + in(xm1,y,0)))
	
	// mu = 2
	- H*(gauge(x,y,1) * fwd_bc*(in(x,yp1,1) - I*in(x,yp1,0)) + conj(gauge(x,ym1,1)) * bwd_bc*(in(x,ym1,1) + I*in(x,ym1,0)));
      out(x,y,1) = tmp;
    }
    fwd_bc = 1.0;
    bwd_bc = 1.0;  
  }
}



template<typename Float>
void Dpsi_parity(auto &out_spinor, const auto &in_spinor, const auto &in_spinor_par, const auto &gauge_field, const Float mass, const Float r, const int nxh, const int ny, const int parity) {
  
  Float constant = (mass + 2*r);
  
  Float fwd_bc = 1.0;
  Float bwd_bc = 1.0;  
  
  //Sum over 0,1 directions at each lattice site.
  int Nxh = nxh;
  int Ny  = ny;
  int xp1, xm1, yp1, ym1;
  
  std::complex<Float> tmp = std::complex<Float>(0.);
  
  std::complex<Float> I   = std::complex<Float>(0., 1.0);
  std::complex<Float> H   = std::complex<Float>(0.5, 0.);  
  
  auto out         = out_spinor.ParityAccessor();
  const auto in    = in_spinor.ParityAccessor();
  //
  const auto in_par= in_spinor_par.ParityAccessor();  
  //
  const auto gauge = gauge_field.Accessor(); 
  
  const int other_parity = 1 - parity; 
  
  //#pragma omp parallel for collapse(2)
  for(int y=0; y<Ny; y++) {
    yp1 = (y+1) == Ny ? 0    : (y+1);
    ym1 = (y-1) == -1 ? Ny-1 : (y-1);

    if(yp1 == 0) fwd_bc = -1.0; 
    if(y   == 0) bwd_bc = -1.0;  
  
    const int parity_bit = parity ? (1 - y & 1) : y & 1;// (1 - y % 1) e1 o0
  
    const int fwd_stride = parity_bit ? +1 :  0; 
    const int bwd_stride = parity_bit ?  0 : +1;       
  
    for(int x=0; x<Nxh; x++) {
      //      
      xp1 = (x+fwd_stride) == Nxh ? 0     : (x+fwd_stride);
      xm1 = (x-bwd_stride) == -1  ? Nxh-1 : (x-bwd_stride);
      //upper
      //(m0 + 2) * I_{X}
      tmp = constant * in_par(x,y,0)
	// mu = 1
	- H*(gauge(x,y,0, parity) * (in(xp1,y,0) - in(xp1,y,1)) + conj(gauge(xm1,y,0,other_parity)) * (in(xm1,y,0) + in(xm1,y,1)))
	
	// mu = 2
	- H*(gauge(x,y,1, parity) * fwd_bc*(in(x,yp1,0) + I*in(x,yp1,1)) + conj(gauge(x,ym1,1,other_parity)) * bwd_bc*(in(x,ym1,0) - I*in(x,ym1,1)));
      out(x,y,0) = tmp;
      
      //lower
      //(m0 + 2) * I_{X}
      tmp = constant * in_par(x,y,1)
	// mu = 1
	- H*(gauge(x,y,0, parity) * (in(xp1,y,1) - in(xp1,y,0)) + conj(gauge(xm1,y,0, other_parity)) * (in(xm1,y,1) + in(xm1,y,0)))
	
	// mu = 2
	- H*(gauge(x,y,1, parity) * fwd_bc*(in(x,yp1,1) - I*in(x,yp1,0)) + conj(gauge(x,ym1,1, other_parity)) * bwd_bc*(in(x,ym1,1) + I*in(x,ym1,0)));
      out(x,y,1) = tmp;
    }
    fwd_bc = 1.0;
    bwd_bc = 1.0;    
  }
}








