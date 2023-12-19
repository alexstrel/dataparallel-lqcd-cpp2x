#pragma once

template<typename Float_, int nColors = 3>
void StaggeredDslashRef(auto &out_spinor, const auto &in_spinor, const auto &accum_spinor, const auto &gauge_field, const Float mass, const std::array<int, 4> n, const int parity) {//const int nx, const int ny
  
  constexpr bool is_constant = true;
  
  const Float_ constant = (2.0*mass);
  
  const int nxh = n[0];
  const int ny  = n[1];
  const int nz  = n[2];  
  const int nt  = n[3];  
  //    
  auto I = [](auto x){ return std::complex<Float_>(-x.imag(), x.real());};  
  
  MDViewTp auto out          = out_spinor.Accessor();
  const MDViewTp auto in     = in_spinor.template Accessor<is_constant>();
  const MDViewTp auto accum  = accum_spinor.template Accessor<is_constant>();  
  //
  const MDViewTp auto gauge  = gauge_field.template Accessor<is_constant>(); 

  const int other_parity = 1 - parity;
  
  std::array<std::complex<Float_>, nColors> tmp;

  for(int t = 0; t < nt; t++) {
    const int tp1 = (t+1) == nt ? 0    : (t+1);
    const int tm1 = (t-1) == -1 ? nt-1 : (t-1);  
    
    for(int z = 0; z < nz; z++) {
      const int zp1 = (z+1) == nz ? 0    : (z+1);
      const int zm1 = (z-1) == -1 ? nz-1 : (z-1);      
      
      for(int y = 0; y < ny; y++) {
        const int yp1 = (y+1) == ny ? 0    : (y+1);
        const int ym1 = (y-1) == -1 ? ny-1 : (y-1);

        const int mask = (t*z*y) & 1; 
             
        const int parity_bit = parity ? (1 - mask) : mask; 
  
        const int fwd_stride = parity_bit ? +1 :  0; 
        const int bwd_stride = parity_bit ?  0 : +1;       
  
        for(int x = 0; x < nxh; x++) {
          //      
          const int xp1 = (x+fwd_stride) == nxh ? 0     : (x+fwd_stride);
          const int xm1 = (x-bwd_stride) == -1  ? nxh-1 : (x-bwd_stride);      
          //
          for (int c = 0; c < nColors; c++) {
            tmp[c] = constant * accum(x,y,z,t,c);          
          }
          for (int c1 = 0; c1 < nColors; c1++) {
            std::array<std::complex<Float_>, nColors> tmp2 = {std::complex<Float_>(0.), std::complex<Float_>(0.), std::complex<Float_>(0.)};
            //fwd gather:
            {
              for (int c2 = 0; c2 < nColors; c2++) {         
                tmp2[c1] += gauge(x,y,z,t,c1,c2,0,parity)*in(xp1,y,z,t,c2);
                tmp2[c1] += gauge(x,y,z,t,c1,c2,1,parity)*in(x,yp1,z,t,c2);
                tmp2[c1] += gauge(x,y,z,t,c1,c2,2,parity)*in(x,y,zp1,t,c2);                                
                tmp2[c1] += gauge(x,y,z,t,c1,c2,3,parity)*in(x,y,z,tp1,c2);                
              } 
            }
            //bwd gather:
            {
              for (int c2 = 0; c2 < nColors; c2++) {         
                tmp2[c1] -= conj(gauge(xm1,y,z,t,c2,c1,0,other_parity))*in(xm1,y,z,t,c2);
                tmp2[c1] -= conj(gauge(x,ym1,z,t,c2,c1,1,other_parity))*in(x,ym1,z,t,c2);
                tmp2[c1] -= conj(gauge(x,y,zm1,t,c2,c1,2,other_parity))*in(x,y,zm1,t,c2);                                
                tmp2[c1] -= conj(gauge(x,y,z,tm1,c2,c1,3,other_parity))*in(x,y,z,tm1,c2);                
              } 
            }
            out(x,y,z,t,c1) = tmp[c1] -  tmp2[c1];  
          }
        }
      }
    }
  }
}



