#pragma once

using vector_tp         = std::vector<std::complex<Float>>;
using sloppy_vector_tp  = std::vector<std::complex<float>>;

using pmr_vector_tp         = impl::pmr::vector<std::complex<Float>>;
using sloppy_pmr_vector_tp  = impl::pmr::vector<std::complex<float>>;


template<typename Float_, int nColors = 3>
void ImprovedDslashRef(auto &out_spinor, const auto &in_spinor, const auto &accum_spinor, const auto &fat_field, const auto &long_field, const Float mass, const std::array<int, 4> n, const int parity) {//const int nx, const int ny
  
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
  const MDViewTp auto fat_links   = fat_field.template Accessor<is_constant>(); 
  const MDViewTp auto long_links  = long_field.template Accessor<is_constant>();   

  const int other_parity = 1 - parity;
  
  std::array<std::complex<Float_>, nColors> tmp;

  for(int t = 0; t < nt; t++) {
    const int tp1 = (t+1) == nt ? 0    : (t+1);
    const int tm1 = (t-1) == -1 ? nt-1 : (t-1);  

    const int tp3 = (t+3) >= nt ? ((t+3) - nt) : (t+3);
    const int tm3 = (t-3) <= -1 ? (nt - (t+3)) : (t-3);  

    
    for(int z = 0; z < nz; z++) {
      const int zp1 = (z+1) == nz ? 0    : (z+1);
      const int zm1 = (z-1) == -1 ? nz-1 : (z-1);      
      //
      const int zp3 = (z+3) >= nz ? ((z+3) - nz) : (z+3);
      const int zm3 = (z-3) <= -1 ? (nz - (z+3)) : (z-3);            
      
      for(int y = 0; y < ny; y++) {
        const int yp1 = (y+1) == ny ? 0    : (y+1);
        const int ym1 = (y-1) == -1 ? ny-1 : (y-1);
        //
        const int yp3 = (y+3) == ny ? ((y+3) - ny) : (y+3);
        const int ym3 = (y-3) == -1 ? (ny - (y+3)) : (y-3);        

        const int mask = (t*z*y) & 1; 
             
        const int parity_bit = parity ? (1 - mask) : mask; 
  
        const int fwd_stride1 = parity_bit ? +1 :  0; 
        const int bwd_stride1 = parity_bit ?  0 : +1;

        const int fwd_stride3 = parity_bit ? +2 :  +1; 
        const int bwd_stride3 = parity_bit ? +1 :  +2;               
  
        for(int x = 0; x < nxh; x++) {
          //      
          const int xp1 = (x+fwd_stride1) == nxh ? 0     : (x+fwd_stride1);
          const int xm1 = (x-bwd_stride1) == -1  ? nxh-1 : (x-bwd_stride1);      
          //
          const int xp3 = (x+fwd_stride3) >= nxh ? ((x+fwd_stride3) - nxh) : (x+fwd_stride3);
          const int xm3 = (x-bwd_stride3) <= -1  ? (nxh - (x+bwd_stride3)) : (x-bwd_stride3);
          //
          for (int c = 0; c < nColors; c++) {
            tmp[c] = constant * accum(x,y,z,t,c);          
          }
          for (int c1 = 0; c1 < nColors; c1++) {
            std::array<std::complex<Float_>, nColors> tmp2 = {std::complex<Float_>(0.), std::complex<Float_>(0.), std::complex<Float_>(0.)};
            //fwd gather:
            {
              for (int c2 = 0; c2 < nColors; c2++) {         
                tmp2[c1] += fat_links(x,y,z,t,c1,c2,0,parity)*in(xp1,y,z,t,c2);
                tmp2[c1] += fat_links(x,y,z,t,c1,c2,1,parity)*in(x,yp1,z,t,c2);
                tmp2[c1] += fat_links(x,y,z,t,c1,c2,2,parity)*in(x,y,zp1,t,c2);                                
                tmp2[c1] += fat_links(x,y,z,t,c1,c2,3,parity)*in(x,y,z,tp1,c2);                
              } 
            }
            //bwd gather:
            {
              for (int c2 = 0; c2 < nColors; c2++) {         
                tmp2[c1] -= conj(fat_links(xm1,y,z,t,c2,c1,0,other_parity))*in(xm1,y,z,t,c2);
                tmp2[c1] -= conj(fat_links(x,ym1,z,t,c2,c1,1,other_parity))*in(x,ym1,z,t,c2);
                tmp2[c1] -= conj(fat_links(x,y,zm1,t,c2,c1,2,other_parity))*in(x,y,zm1,t,c2);                                
                tmp2[c1] -= conj(fat_links(x,y,z,tm1,c2,c1,3,other_parity))*in(x,y,z,tm1,c2);                
              } 
            }
            //fwd gather:
            {
              for (int c2 = 0; c2 < nColors; c2++) {         
                tmp2[c1] += long_links(x,y,z,t,c1,c2,0,parity)*in(xp3,y,z,t,c2);
                tmp2[c1] += long_links(x,y,z,t,c1,c2,1,parity)*in(x,yp3,z,t,c2);
                tmp2[c1] += long_links(x,y,z,t,c1,c2,2,parity)*in(x,y,zp3,t,c2);                                
                tmp2[c1] += long_links(x,y,z,t,c1,c2,3,parity)*in(x,y,z,tp3,c2);                
              } 
            }
            //bwd gather:
            {
              for (int c2 = 0; c2 < nColors; c2++) {         
                tmp2[c1] -= conj(long_links(xm3,y,z,t,c2,c1,0,other_parity))*in(xm3,y,z,t,c2);
                tmp2[c1] -= conj(long_links(x,ym3,z,t,c2,c1,1,other_parity))*in(x,ym3,z,t,c2);
                tmp2[c1] -= conj(long_links(x,y,zm3,t,c2,c1,2,other_parity))*in(x,y,zm3,t,c2);                                
                tmp2[c1] -= conj(long_links(x,y,z,tm3,c2,c1,3,other_parity))*in(x,y,z,tm3,c2);                
              } 
            }            
            out(x,y,z,t,c1) = tmp[c1] -  tmp2[c1];  
          }
        }
      }
    }
  }
}



