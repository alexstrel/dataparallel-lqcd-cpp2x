#pragma once

using vector_tp         = std::vector<std::complex<Float>>;
using sloppy_vector_tp  = std::vector<std::complex<float>>;

using pmr_vector_tp         = impl::pmr::vector<std::complex<Float>>;
using sloppy_pmr_vector_tp  = impl::pmr::vector<std::complex<float>>;


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


void run_pmr_dslash_test(auto params, const auto dims, const int niter, const int test_type) {
  //
  constexpr int nSpinorParity = 2;
  constexpr int nGaugeParity  = 2;
  //
  constexpr bool clean_intermed_fields = true;
  //  
  const auto cs_param    = StaggeredSpinorFieldArgs<nSpinorParity>{dims};
  //
  const auto gauge_param = GaugeFieldArgs<nGaugeParity>{dims};

  // Create full precision gauge field:
  auto fat_lnks = create_field<vector_tp, decltype(gauge_param)>(gauge_param);
  //
  auto long_lnks = create_field<vector_tp, decltype(gauge_param)>(gauge_param);
  //
#if 0
  constructFatLongGaugeField<0, 1>(fat_lnks, long_lnks, 0.5, 5.0, test_type);
#else
  init_su3(fat_lnks);
  init_su3(long_lnks);
#endif

  // Create low precision gauge field (NOTE: by setting copy_gauge = true we migrate data on the device):  
  constexpr bool copy_gauge = true;

  auto sloppy_fat_lnks  = create_field<decltype(fat_lnks), sloppy_vector_tp, copy_gauge>(fat_lnks);   

  auto sloppy_long_lnks = create_field<decltype(long_lnks), sloppy_vector_tp, copy_gauge>(long_lnks);  

  auto src_spinor  = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);
  auto chk_spinor  = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);  
  
  using cs_param_tp = decltype(src_spinor.ExportArg());
  
  auto dst_spinor  = create_field_with_buffer<sloppy_pmr_vector_tp, cs_param_tp>(cs_param);
  //
  init_spinor(src_spinor);  

  // Setup dslash arguments:
  auto &&fl_ref  = sloppy_fat_lnks.View();
  auto &&ll_ref  = sloppy_long_lnks.View();

  using sloppy_gauge_tp = decltype(sloppy_fat_lnks.View());

  std::unique_ptr<StaggeredDslashArgs<sloppy_gauge_tp>> hisq_args_ptr(new StaggeredDslashArgs{fl_ref, ll_ref});

  auto &hisq_args = *hisq_args_ptr;

  // Create dslash matrix
  auto mat = Mat<decltype(hisq_args), StaggeredDslash, decltype(params)>{hisq_args, params};

  using arg_tp = decltype(src_spinor.Even().ExportArg());  
  //
  constexpr bool do_warmup = true;
  //
  if constexpr (do_warmup) {
    mat(dst_spinor, src_spinor);
  }
  std::cout << "Begin bench \n" << std::endl;
  //
  auto wall_start = std::chrono::high_resolution_clock::now(); 
  
  for(int i = 0; i < niter; i++) {

    mat(dst_spinor, src_spinor);    
  }

  auto wall_stop = std::chrono::high_resolution_clock::now();

  auto wall_diff = wall_stop - wall_start;
  
  auto wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  constexpr bool do_check = false;

  if constexpr (do_check) { 
    auto [even_chk, odd_chk] = chk_spinor.EODecompose();    
  
    StaggeredDslashRef<float>(even_chk, src_spinor.Odd(),  src_spinor.Even(), sloppy_fat_lnks, params.M, even_chk.GetCBDims(), 0); 
    StaggeredDslashRef<float>(odd_chk,  src_spinor.Even(),  src_spinor.Odd(),  sloppy_fat_lnks, params.M, odd_chk.GetCBDims(), 1);   
  
    auto &&chk_e = even_chk.Accessor();
    auto &&dst_e = dst_spinor.Even().Accessor();     
    //
    check_field(chk_e, dst_e, 1e-6);
    //
    auto &&chk_o = odd_chk.Accessor();
    auto &&dst_o = dst_spinor.Odd().Accessor();     
    //
    check_field(chk_o, dst_o, 1e-6);    
  }    
  
  std::cout << "Done for EO version : time per iteration is > " << wall_time << "sec." << std::endl; 
  
  src_spinor.show();
  if constexpr (clean_intermed_fields) {
    dst_spinor.destroy();
    src_spinor.destroy();
    chk_spinor.destroy();    
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "Run next src (re-use buffer)" << std::endl;

  auto src_spinor_v2   = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);
  auto dst_spinor_v2   = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);

  wall_start = std::chrono::high_resolution_clock::now();   

  for(int i = 0; i < niter; i++) {
    // Apply dslash	  
    mat(dst_spinor_v2, src_spinor_v2);
  }
  
  wall_stop = std::chrono::high_resolution_clock::now();

  wall_diff = wall_stop - wall_start;
  
  wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  std::cout << "Done for another EO version : time per iteration is > " << wall_time << "sec." << std::endl;  

  long_lnks.destroy();
  sloppy_long_lnks.destroy();
  
  fat_lnks.destroy();
  sloppy_fat_lnks.destroy();  
  
  src_spinor_v2.show();  
  
  dst_spinor_v2.destroy();
  src_spinor_v2.destroy();  

  if constexpr (not clean_intermed_fields) {
    dst_spinor.destroy();
    src_spinor.destroy();
  }
}


template<int N>
void run_mrhs_pmr_dslash_test(auto params, const auto dims, const int niter, const int test_type) {
  // 
  constexpr int nSpinorParity = 2;
  constexpr int nGaugeParity  = 2;
  // 
  constexpr bool do_warmup = true; 
  //
  const auto cs_param = StaggeredSpinorFieldArgs<nSpinorParity>{dims,FieldParity::InvalidFieldParity};
  //
  const auto gauge_param = GaugeFieldArgs<nGaugeParity>{dims};
  //
  auto fat_lnks = create_field<vector_tp, decltype(gauge_param)>(gauge_param);

  auto long_lnks = create_field<vector_tp, decltype(gauge_param)>(gauge_param);
#if 0
  constructFatLongGaugeField<0, 1>(fat_lnks, long_lnks, 0.5, 5.0, test_type);
#else
  init_su3(fat_lnks);
  init_su3(long_lnks);
#endif

  constexpr bool copy_gauge = true;
     
  auto sloppy_fat_lnks  = create_field<decltype(fat_lnks), sloppy_vector_tp, copy_gauge>(fat_lnks);   

  auto sloppy_long_lnks = create_field<decltype(long_lnks), sloppy_vector_tp, copy_gauge>(long_lnks);
  //
  // Setup dslash arguments:
  auto &&fl_ref  = sloppy_fat_lnks.View();
  auto &&ll_ref  = sloppy_long_lnks.View();

  using sloppy_gauge_tp = decltype(sloppy_fat_lnks.View());

  std::unique_ptr<StaggeredDslashArgs<sloppy_gauge_tp>> hisq_args_ptr(new StaggeredDslashArgs{fl_ref, ll_ref});

  auto &hisq_args = *hisq_args_ptr;

  // Create dslash matrix
  auto mat = Mat<decltype(hisq_args), StaggeredDslash, decltype(params)>{hisq_args, params};   
  //
  using sloppy_pmr_spinor_t  = Field<sloppy_pmr_vector_tp, decltype(cs_param)>;//

  auto src_block_spinor = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param) >(cs_param, N); 
  auto chk_block_spinor = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param) >(cs_param, N);

  for (int i = 0; i < src_block_spinor.nComponents(); i++) init_spinor( src_block_spinor.v[i] );
  
  auto dst_block_spinor = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param) >(cs_param, N);  
  //
  if constexpr (do_warmup) {
    mat(dst_block_spinor, src_block_spinor);    
  }

  std::cout << "Begin bench \n" << std::endl;

  auto wall_start = std::chrono::high_resolution_clock::now();   
 
  for(int i = 0; i < niter; i++) {
    mat(dst_block_spinor, src_block_spinor);      
  } 

  auto wall_stop = std::chrono::high_resolution_clock::now();

  auto wall_diff = wall_stop - wall_start;

  auto wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  std::cout << "Done for MRHS version (N =  " << N << ") : time per iteration is > " << wall_time << "sec." << std::endl;

  
  src_block_spinor[0].show();
  //
  src_block_spinor.destroy();
  dst_block_spinor.destroy();  

  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////

  auto src_block_spinor_v2 = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param)>(cs_param, N);
  auto dst_block_spinor_v2 = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param)>(cs_param, N);
  
  wall_start = std::chrono::high_resolution_clock::now();

  for(int i = 0; i < niter; i++) {
    mat(dst_block_spinor_v2, src_block_spinor_v2);
  }

  wall_stop = std::chrono::high_resolution_clock::now();

  wall_diff = wall_stop - wall_start;

  wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  std::cout << "Done for MRHS version (N =  " << N << ") : time per iteration is > " << wall_time << "sec." << std::endl;


  src_block_spinor_v2[0].show();

  src_block_spinor_v2.destroy();
  dst_block_spinor_v2.destroy();

  long_lnks.destroy();
  sloppy_long_lnks.destroy();
  
  fat_lnks.destroy();
  sloppy_fat_lnks.destroy(); 
}



