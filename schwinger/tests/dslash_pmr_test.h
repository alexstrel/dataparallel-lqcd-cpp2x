#pragma once

using vector_tp         = std::vector<std::complex<Float>>;
using sloppy_vector_tp  = std::vector<std::complex<float>>;

using pmr_vector_tp         = impl::pmr::vector<std::complex<Float>>;
using sloppy_pmr_vector_tp  = impl::pmr::vector<std::complex<float>>;

template<typename Float_>
void DslashRef(auto &out_spinor, const auto &in_spinor, const auto &accum_spinor, const auto &gauge_field, const Float mass, const Float r, const std::array<int, 2> n, const int parity) {//const int nx, const int ny
  
  constexpr bool is_constant = true;
  
  const Float_ constant = (mass + 2.0*r);
  
  const int nxh = n[0];
  const int ny  = n[1];
  //  
  std::complex<Float_> tmp = std::complex<Float_>(0.);
  std::complex<Float_> hlf = std::complex<Float_>(0.5, 0.);  
  
  auto I = [](auto x){ return std::complex<Float_>(-x.imag(), x.real());};  
  
  MDViewTp auto out          = out_spinor.Accessor();
  const MDViewTp auto in     = in_spinor.template Accessor<is_constant>();
  const MDViewTp auto accum  = accum_spinor.template Accessor<is_constant>();  
  //
  const MDViewTp auto gauge  = gauge_field.template Accessor<is_constant>(); 

  const int other_parity = 1 - parity;

  for(int y = 0; y < ny; y++) {
    const int yp1 = (y+1) == ny ? 0    : (y+1);
    const int ym1 = (y-1) == -1 ? ny-1 : (y-1);

    const std::complex<Float_> fwd_bndr = yp1 == 0 ? std::complex<Float_>{-1.0, 0.0} : std::complex<Float_>{+1.0, 0.0};
    const std::complex<Float_> bwd_bndr = y   == 0 ? std::complex<Float_>{-1.0, 0.0} : std::complex<Float_>{+1.0, 0.0};
    
    const int parity_bit = parity ? (1 - y & 1) : y & 1; 
  
    const int fwd_stride = parity_bit ? +1 :  0; 
    const int bwd_stride = parity_bit ?  0 : +1;       
  
    for(int x = 0; x < nxh; x++) {
      //      
      const int xp1 = (x+fwd_stride) == nxh ? 0     : (x+fwd_stride);
      const int xm1 = (x-bwd_stride) == -1  ? nxh-1 : (x-bwd_stride);      
      //
      tmp = constant * accum(x,y,0)
	- hlf*(gauge(x,y,0, parity) * (in(xp1,y,0) - in(xp1,y,1)) + conj(gauge(xm1,y,0, other_parity)) * (in(xm1,y,0) + in(xm1,y,1)))
	
	- hlf*(gauge(x,y,1, parity) * fwd_bndr*(in(x,yp1,0) + I(in(x,yp1,1))) + conj(gauge(x,ym1,1, other_parity)) * bwd_bndr*(in(x,ym1,0) - I(in(x,ym1,1))));
      out(x,y,0) = tmp;
      //
      tmp = constant * accum(x,y,1)
	- hlf*(gauge(x,y,0, parity) * (in(xp1,y,1) - in(xp1,y,0)) + conj(gauge(xm1,y,0, other_parity)) * (in(xm1,y,1) + in(xm1,y,0)))
	
	- hlf*(gauge(x,y,1, parity) * fwd_bndr*(in(x,yp1,1) - I(in(x,yp1,0))) + conj(gauge(x,ym1,1, other_parity)) * bwd_bndr*(in(x,ym1,1) + I(in(x,ym1,0))));
      out(x,y,1) = tmp;
    }
  }

}

void run_direct_pmr_dslash_test(auto params, const int X, const int T, const int niter) {
  //
  constexpr int nSpinorParity = 2;
  constexpr int nGaugeParity  = 2;
  //
  constexpr bool clean_intermed_fields = true;
  //  
  const auto cs_param    = SpinorFieldArgs<nSpinorParity>{{X, T}, {0, 0, 0, 0}};
  //
  const auto gauge_param = GaugeFieldArgs<nGaugeParity>{{X, T}, {0, 0}};

  // Create full precision gauge field:
  auto gauge = create_field<vector_tp, decltype(gauge_param)>(gauge_param);
  //
  init_u1(gauge);
  
  // Create low precision gauge field (NOTE: by setting copy_gauge = true we migrate data on the device):  
  constexpr bool copy_gauge = true;
    
  auto sloppy_gauge = create_field<decltype(gauge), sloppy_vector_tp, copy_gauge>(gauge);  
  
  auto src_spinor  = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);
  auto dst_spinor  = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);
  auto chk_spinor  = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);  
  //
  init_spinor(src_spinor);  

  // Setup dslash arguments:
  auto &&u_ref        = sloppy_gauge.View();
  //u_ref.destroy();
  using sloppy_gauge_tp = decltype(sloppy_gauge.View());

  std::unique_ptr<DslashArgs<sloppy_gauge_tp, decltype(cs_param)::nspin>> dslash_args_ptr(new DslashArgs{u_ref});

  auto &dslash_args = *dslash_args_ptr;

  // Create dslash matrix
  auto mat = DslashTransform<decltype(dslash_args), Dslash>{dslash_args};   
  
  const auto const1 = static_cast<float>(params.M + 2.0*params.r);
  const auto const2 = static_cast<float>(0.5);

  auto transformer = [=](const auto &x, const auto &y) {return (const1*x-const2*y);};
  //
  auto [even_src, odd_src] = src_spinor.EODecompose();
  auto [even_dst, odd_dst] = dst_spinor.EODecompose();
  //
  constexpr bool do_warmup = false;
  //
  if constexpr (do_warmup) {
    mat(even_dst, odd_src,  even_src, transformer, FieldParity::EvenFieldParity);
    mat(odd_dst,  even_src, odd_src,  transformer, FieldParity::OddFieldParity);
  }
  //
  auto wall_start = std::chrono::high_resolution_clock::now(); 
  
  for(int i = 0; i < niter; i++) {
    // Apply dslash	  
    mat(even_dst, odd_src,  even_src, transformer, FieldParity::EvenFieldParity);
    mat(odd_dst,  even_src, odd_src,  transformer, FieldParity::OddFieldParity); 
  }
  
  auto wall_stop = std::chrono::high_resolution_clock::now();

  auto wall_diff = wall_stop - wall_start;
  
  auto wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  constexpr bool do_check = false;  

  if constexpr (do_check) { 
    auto [even_chk, odd_chk] = chk_spinor.EODecompose();    
  
    DslashRef<float>(even_chk, odd_src,  even_src, sloppy_gauge, params.M, params.r, even_chk.GetCBDims(), 0); 
    DslashRef<float>(odd_chk,  even_src, odd_src,  sloppy_gauge, params.M, params.r, odd_chk.GetCBDims(), 1);   
  
    auto &&chk_e = even_chk.Accessor();
    auto &&dst_e = even_dst.Accessor();     
    //
    check_field(chk_e, dst_e, 1e-6);
    //
    auto &&chk_o = odd_chk.Accessor();
    auto &&dst_o = odd_dst.Accessor();     
    //
    check_field(chk_o, dst_o, 1e-6);    
  }
  
  std::cout << "Done for EO version : time per iteration is > " << wall_time << "sec." << std::endl; 
  
  src_spinor.show();
  if constexpr (clean_intermed_fields) {
    dst_spinor.destroy();
    src_spinor.destroy();
  }
  chk_spinor.destroy();

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "Run next src (re-use buffer)" << std::endl;

  auto src_spinor_v2   = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);
  auto dst_spinor_v2   = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);
  auto chk_spinor_v2   = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);  
  auto [even_src_v2, odd_src_v2] = src_spinor_v2.EODecompose();
  auto [even_dst_v2, odd_dst_v2] = dst_spinor_v2.EODecompose();

  wall_start = std::chrono::high_resolution_clock::now();   

  for(int i = 0; i < niter; i++) {
    // Apply dslash	  
    mat(even_dst_v2, odd_src_v2,  even_src_v2, transformer, FieldParity::EvenFieldParity);
    mat(odd_dst_v2,  even_src_v2, odd_src_v2,  transformer, FieldParity::OddFieldParity); 
  }
  
  wall_stop = std::chrono::high_resolution_clock::now();

  wall_diff = wall_stop - wall_start;
  
  wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;
  
  auto [even_chk_v2, odd_chk_v2] = chk_spinor_v2.EODecompose();    
  
  DslashRef<float>(even_chk_v2, odd_src_v2,  even_src_v2, sloppy_gauge, params.M, params.r, even_chk_v2.GetCBDims(), 0); 
  DslashRef<float>(odd_chk_v2,  even_src_v2, odd_src_v2,  sloppy_gauge, params.M, params.r, odd_chk_v2.GetCBDims(), 1);   

  {
    //check_field(chk_spinor, dst_spinor);
    auto &&chk_e = even_chk_v2.Accessor();
    auto &&dst_e = even_dst_v2.Accessor();     
    //
    check_field(chk_e, dst_e, 1e-6);
    //
    auto &&chk_o = odd_chk_v2.Accessor();
    auto &&dst_o = odd_dst_v2.Accessor();     
    //
    check_field(chk_o, dst_o, 1e-6);    
  }  

  std::cout << "Done for another EO version : time per iteration is > " << wall_time << "sec." << std::endl;  

  gauge.destroy();
  sloppy_gauge.destroy();
  
  src_spinor_v2.show();  
  
  dst_spinor_v2.destroy();
  src_spinor_v2.destroy();  
  chk_spinor_v2.destroy();    

  if constexpr (not clean_intermed_fields) {
    dst_spinor.destroy();
    src_spinor.destroy();
  }
}


/*
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
**/

void run_pmr_dslash_test(auto params, const int X, const int T, const int niter) {
  //
  constexpr int nSpinorParity = 2;
  constexpr int nGaugeParity  = 2;
  //
  constexpr bool clean_intermed_fields = true;
  //  
  const auto cs_param    = SpinorFieldArgs<nSpinorParity>{{X, T}, {0, 0, 0, 0}};
  //
  const auto gauge_param = GaugeFieldArgs<nGaugeParity>{{X, T}, {0, 0}};

  // Create full precision gauge field:
  auto gauge = create_field<vector_tp, decltype(gauge_param)>(gauge_param);
  //
  init_u1(gauge);
  
  // Create low precision gauge field (NOTE: by setting copy_gauge = true we migrate data on the device):  
  constexpr bool copy_gauge = true;
    
  auto sloppy_gauge = create_field<decltype(gauge), sloppy_vector_tp, copy_gauge>(gauge);  
  
  auto src_spinor  = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);
  
  using cs_param_tp = decltype(src_spinor.ExportArg());
  
  auto dst_spinor  = create_field_with_buffer<sloppy_pmr_vector_tp, cs_param_tp>(cs_param);
  auto chk_spinor  = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);  
  //
  init_spinor(src_spinor);  

  // Setup dslash arguments:
  auto &&u_ref        = sloppy_gauge.View();
  //u_ref.destroy();
  using sloppy_gauge_tp = decltype(sloppy_gauge.View());

  std::unique_ptr<DslashArgs<sloppy_gauge_tp, decltype(cs_param)::nspin>> dslash_args_ptr(new DslashArgs{u_ref});

  auto &dslash_args = *dslash_args_ptr;

  // Create dslash matrix
  auto mat = Mat<decltype(dslash_args), Dslash, decltype(params)>{dslash_args, params};

  using arg_tp = decltype(src_spinor.Even().ExportArg());  

  auto mat_precon = PreconMat< decltype(dslash_args), Dslash, decltype(params), arg_tp >{dslash_args, params, src_spinor.Even().ExportArg()}; 
  auto tmp        = create_field<sloppy_pmr_vector_tp, arg_tp>(src_spinor.Even().ExportArg());       
  //
  constexpr bool do_warmup = false;
  //
  if constexpr (do_warmup) {
    mat(dst_spinor, src_spinor);
    mat_precon(dst_spinor.Even(), src_spinor.Even());    
  }
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
  
    DslashRef<float>(even_chk, src_spinor.Odd(),  src_spinor.Even(), sloppy_gauge, params.M, params.r, even_chk.GetCBDims(), 0); 
    DslashRef<float>(odd_chk,  src_spinor.Even(),  src_spinor.Odd(),  sloppy_gauge, params.M, params.r, odd_chk.GetCBDims(), 1);   
  
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
  }
  chk_spinor.destroy();

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "Run next src (re-use buffer)" << std::endl;

  auto src_spinor_v2   = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);
  auto dst_spinor_v2   = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);
  auto chk_spinor_v2   = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);  

  wall_start = std::chrono::high_resolution_clock::now();   

  for(int i = 0; i < niter; i++) {
    // Apply dslash	  
    mat(dst_spinor_v2, src_spinor_v2);
  }
  
  wall_stop = std::chrono::high_resolution_clock::now();

  wall_diff = wall_stop - wall_start;
  
  wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;
  
  auto [even_chk_v2, odd_chk_v2] = chk_spinor_v2.EODecompose();    
  
  DslashRef<float>(even_chk_v2, src_spinor_v2.Odd(),  src_spinor_v2.Even(), sloppy_gauge, params.M, params.r, even_chk_v2.GetCBDims(), 0); 
  DslashRef<float>(odd_chk_v2,  src_spinor_v2.Even(), src_spinor_v2.Odd(),  sloppy_gauge, params.M, params.r, odd_chk_v2.GetCBDims(), 1);   

  {
    //check_field(chk_spinor, dst_spinor);
    auto &&chk_e = even_chk_v2.Accessor();
    auto &&dst_e = dst_spinor_v2.Even().Accessor();     
    //
    check_field(chk_e, dst_e, 1e-6);
    //
    auto &&chk_o = odd_chk_v2.Accessor();
    auto &&dst_o = dst_spinor_v2.Odd().Accessor();     
    //
    check_field(chk_o, dst_o, 1e-6);    
  }  

  std::cout << "Done for another EO version : time per iteration is > " << wall_time << "sec." << std::endl;  

  gauge.destroy();
  sloppy_gauge.destroy();
  
  src_spinor_v2.show();  
  
  dst_spinor_v2.destroy();
  src_spinor_v2.destroy();  
  chk_spinor_v2.destroy();    

  if constexpr (not clean_intermed_fields) {
    dst_spinor.destroy();
    src_spinor.destroy();
  }
}


/*============================================================================================*/

template<int N>
void run_direct_mrhs_pmr_dslash_test(auto params, const int X, const int T, const int niter) {
  // 
  constexpr int nSpinorParity = 2;
  constexpr int nGaugeParity  = 2;
  // 
  constexpr bool do_warmup = false; 
  //
  const auto cs_param = SpinorFieldArgs<nSpinorParity>{{X, T}, {0, 0, 0, 0}, FieldParity::EvenFieldParity};
  //
  const auto gauge_param = GaugeFieldArgs<nGaugeParity>{{X, T}, {0, 0}};
  //
  auto gauge = create_field<vector_tp, decltype(gauge_param)>(gauge_param);    
  //  
  init_u1(gauge);

  constexpr bool copy_gauge = true;
     
  auto sloppy_gauge = create_field<decltype(gauge), sloppy_vector_tp, copy_gauge>(gauge);
  //
  auto &&u_ref          = sloppy_gauge.View();
  using sloppy_gauge_tp = decltype(sloppy_gauge.View());

  std::unique_ptr<DslashArgs<sloppy_gauge_tp, decltype(cs_param)::nspin>> dslash_args_ptr(new DslashArgs{u_ref});

  auto &dslash_args = *dslash_args_ptr;

  // Create dslash matrix
  auto mat = DslashTransform<decltype(dslash_args), Dslash>{dslash_args};    
  //
  using sloppy_pmr_spinor_t  = Field<sloppy_pmr_vector_tp, decltype(cs_param)>;//
  //
  constexpr bool use_pmr_buffer = true;
  //
  auto src_block_spinor = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param), use_pmr_buffer >(cs_param, N); 
  auto chk_block_spinor = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param), use_pmr_buffer >(cs_param, N);

  for (int i = 0; i < src_block_spinor.nComponents(); i++) init_spinor( src_block_spinor.v[i] );
  
  auto dst_block_spinor = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param), use_pmr_buffer >(cs_param, N); 
  
  const auto const1 = static_cast<float>(params.M + 2.0*params.r);
  const auto const2 = static_cast<float>(0.5);

  auto [even_src_block, odd_src_block] = src_block_spinor.EODecompose();
  auto [even_dst_block, odd_dst_block] = dst_block_spinor.EODecompose();

  auto transformer = [=](const auto &x, const auto &y) {return (const1*x-const2*y);}; 
  //
  if constexpr (do_warmup) {
    mat(even_dst_block, odd_src_block,  even_src_block, transformer, FieldParity::EvenFieldParity);  
    mat(odd_dst_block,  even_src_block, odd_src_block,  transformer, FieldParity::OddFieldParity);
  }

  auto wall_start = std::chrono::high_resolution_clock::now();   
 
  for(int i = 0; i < niter; i++) {
    mat(even_dst_block, odd_src_block,  even_src_block, transformer, FieldParity::EvenFieldParity);   
    mat(odd_dst_block,  even_src_block, odd_src_block,  transformer, FieldParity::OddFieldParity);  
  } 

  auto wall_stop = std::chrono::high_resolution_clock::now();

  auto wall_diff = wall_stop - wall_start;

  auto wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  std::cout << "Done for MRHS version (N =  " << N << ") : time per iteration is > " << wall_time << "sec." << std::endl;

  constexpr bool do_check = false;
  
  if constexpr (do_check) {
    auto [even_chk_block, odd_chk_block] = chk_block_spinor.EODecompose();

    for (int i = 0; i < chk_block_spinor.nComponents(); i++) {

      DslashRef<float>(even_chk_block[i], odd_src_block[i],  even_src_block[i], sloppy_gauge, params.M, params.r, even_chk_block[i].GetCBDims(), 0);   
      DslashRef<float>(odd_chk_block[i],  even_src_block[i], odd_src_block[i],  sloppy_gauge, params.M, params.r, odd_chk_block[i].GetCBDims(), 1);

      auto &&chk_e = even_chk_block[i].Accessor();
      auto &&dst_e = even_dst_block[i].Accessor();
    
      check_field(chk_e, dst_e, 1e-6);
    
      auto &&chk_o = odd_chk_block[i].Accessor();
      auto &&dst_o = odd_dst_block[i].Accessor();
    
      check_field(chk_o, dst_o, 1e-6);
    }
  }

  src_block_spinor[0].show();
  //
  src_block_spinor.destroy();
  dst_block_spinor.destroy();  
  chk_block_spinor.destroy();
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////

  auto src_block_spinor_v2 = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param), use_pmr_buffer >(cs_param, N);
  auto chk_block_spinor_v2 = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param), use_pmr_buffer >(cs_param, N);

  auto dst_block_spinor_v2 = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param), use_pmr_buffer >(cs_param, N);

  auto [even_src_block_v2, odd_src_block_v2] = src_block_spinor_v2.EODecompose();
  auto [even_dst_block_v2, odd_dst_block_v2] = dst_block_spinor_v2.EODecompose();
  
  wall_start = std::chrono::high_resolution_clock::now();

  for(int i = 0; i < niter; i++) {
    mat(even_dst_block_v2, odd_src_block_v2,  even_src_block_v2, transformer, FieldParity::EvenFieldParity);
    mat(odd_dst_block_v2,  even_src_block_v2, odd_src_block_v2,  transformer, FieldParity::OddFieldParity);
  }

  wall_stop = std::chrono::high_resolution_clock::now();

  wall_diff = wall_stop - wall_start;

  wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  std::cout << "Done for MRHS version (N =  " << N << ") : time per iteration is > " << wall_time << "sec." << std::endl;

  if constexpr (do_check) {
    auto [even_chk_block_v2, odd_chk_block_v2] = chk_block_spinor_v2.EODecompose();

    for (int i = 0; i < chk_block_spinor_v2.nComponents(); i++) {

      DslashRef<float>(even_chk_block_v2[i], odd_src_block_v2[i],  even_src_block_v2[i], sloppy_gauge, params.M, params.r, even_chk_block_v2[i].GetCBDims(), 0);
      DslashRef<float>(odd_chk_block_v2[i],  even_src_block_v2[i], odd_src_block_v2[i],  sloppy_gauge, params.M, params.r, odd_chk_block_v2[i].GetCBDims(), 1);

      auto &&chk_e = even_chk_block_v2[i].Accessor();
      auto &&dst_e = even_dst_block_v2[i].Accessor();

      check_field(chk_e, dst_e, 1e-6);

      auto &&chk_o = odd_chk_block_v2[i].Accessor();
      auto &&dst_o = odd_dst_block_v2[i].Accessor();

      check_field(chk_o, dst_o, 1e-6);
    }
  }

  src_block_spinor_v2[0].show();

  src_block_spinor_v2.destroy();
  dst_block_spinor_v2.destroy();
  chk_block_spinor_v2.destroy();

  gauge.destroy();  
  sloppy_gauge.destroy();
}

/*
 *
 *
 *
 *
 *
 *
 *
 *
 */

template<int N>
void run_mrhs_pmr_dslash_test(auto params, const int X, const int T, const int niter) {
  // 
  constexpr int nSpinorParity = 2;
  constexpr int nGaugeParity  = 2;
  // 
  constexpr bool do_warmup = false; 
  //
  const auto cs_param = SpinorFieldArgs<nSpinorParity>{{X, T}, {0, 0, 0, 0}, FieldParity::InvalidFieldParity};
  //
  const auto gauge_param = GaugeFieldArgs<nGaugeParity>{{X, T}, {0, 0}};
  //
  auto gauge = create_field<vector_tp, decltype(gauge_param)>(gauge_param);    
  //  
  init_u1(gauge);

  constexpr bool copy_gauge = true;
     
  auto sloppy_gauge = create_field<decltype(gauge), sloppy_vector_tp, copy_gauge>(gauge);
  //
  auto &&u_ref          = sloppy_gauge.View();
  using sloppy_gauge_tp = decltype(sloppy_gauge.View());

  std::unique_ptr<DslashArgs<sloppy_gauge_tp, decltype(cs_param)::nspin>> dslash_args_ptr(new DslashArgs{u_ref});

  auto &dslash_args = *dslash_args_ptr;

  // Create dslash matrix
  auto mat = Mat<decltype(dslash_args), Dslash, decltype(params)>{dslash_args, params};   
  //
  using sloppy_pmr_spinor_t  = Field<sloppy_pmr_vector_tp, decltype(cs_param)>;//
  //
  constexpr bool use_pmr_buffer = true;

  auto src_block_spinor = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param), use_pmr_buffer >(cs_param, N); 
  auto chk_block_spinor = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param), use_pmr_buffer >(cs_param, N);

  for (int i = 0; i < src_block_spinor.nComponents(); i++) init_spinor( src_block_spinor.v[i] );
  
  auto dst_block_spinor = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param), use_pmr_buffer >(cs_param, N);  
  //
  if constexpr (do_warmup) {
    mat(dst_block_spinor, src_block_spinor);    
  }

  auto wall_start = std::chrono::high_resolution_clock::now();   
 
  for(int i = 0; i < niter; i++) {
    mat(dst_block_spinor, src_block_spinor);      
  } 

  auto wall_stop = std::chrono::high_resolution_clock::now();

  auto wall_diff = wall_stop - wall_start;

  auto wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  std::cout << "Done for MRHS version (N =  " << N << ") : time per iteration is > " << wall_time << "sec." << std::endl;

  constexpr bool do_check = false;
  
  if constexpr (do_check) {
    auto [even_chk_block, odd_chk_block] = chk_block_spinor.EODecompose();

    for (int i = 0; i < chk_block_spinor.nComponents(); i++) {

      DslashRef<float>(even_chk_block[i], src_block_spinor[i].Odd(), src_block_spinor[i].Even(), sloppy_gauge, params.M, params.r, even_chk_block[i].GetCBDims(), 0);   
      DslashRef<float>(odd_chk_block[i],  src_block_spinor[i].Even(), src_block_spinor[i].Odd(),  sloppy_gauge, params.M, params.r, odd_chk_block[i].GetCBDims(), 1);

      auto &&chk_e = even_chk_block[i].Accessor();
      auto &&dst_e = dst_block_spinor[i].Even().Accessor();
    
      check_field(chk_e, dst_e, 1e-6);
    
      auto &&chk_o = odd_chk_block[i].Accessor();
      auto &&dst_o = dst_block_spinor[i].Odd().Accessor();
    
      check_field(chk_o, dst_o, 1e-6);
    }
  }

  src_block_spinor[0].show();
  //
  src_block_spinor.destroy();
  dst_block_spinor.destroy();  
  chk_block_spinor.destroy();
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////

  auto src_block_spinor_v2 = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param), use_pmr_buffer >(cs_param, N);
  auto chk_block_spinor_v2 = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param), use_pmr_buffer >(cs_param, N);

  auto dst_block_spinor_v2 = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param), use_pmr_buffer >(cs_param, N);
  
  wall_start = std::chrono::high_resolution_clock::now();

  for(int i = 0; i < niter; i++) {
    mat(dst_block_spinor_v2, src_block_spinor_v2);
  }

  wall_stop = std::chrono::high_resolution_clock::now();

  wall_diff = wall_stop - wall_start;

  wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  std::cout << "Done for MRHS version (N =  " << N << ") : time per iteration is > " << wall_time << "sec." << std::endl;

  {
    auto [even_chk_block_v2, odd_chk_block_v2] = chk_block_spinor_v2.EODecompose();

    for (int i = 0; i < chk_block_spinor_v2.nComponents(); i++) {

      DslashRef<float>(even_chk_block_v2[i], src_block_spinor_v2[i].Odd(), src_block_spinor_v2[i].Even(), sloppy_gauge, params.M, params.r, even_chk_block_v2[i].GetCBDims(), 0);   
      DslashRef<float>(odd_chk_block_v2[i],  src_block_spinor_v2[i].Even(), src_block_spinor_v2[i].Odd(),  sloppy_gauge, params.M, params.r, odd_chk_block_v2[i].GetCBDims(), 1);

      auto &&chk_e = even_chk_block_v2[i].Accessor();
      auto &&dst_e = dst_block_spinor_v2[i].Even().Accessor();
    
      check_field(chk_e, dst_e, 1e-6);
    
      auto &&chk_o = odd_chk_block_v2[i].Accessor();
      auto &&dst_o = dst_block_spinor_v2[i].Odd().Accessor();
    
      check_field(chk_o, dst_o, 1e-6);
    }
  }

  src_block_spinor_v2[0].show();

  src_block_spinor_v2.destroy();
  dst_block_spinor_v2.destroy();
  chk_block_spinor_v2.destroy();

  gauge.destroy();  
  sloppy_gauge.destroy();
}




