#pragma once

#include <fields/field_descriptor.h>

// Create generic concept of allocator aware containers: 
template<typename T> concept AllocatorAwareContainerTp = ContainerTp<T> or PMRContainerTp<T>;

template<GenericContainerTp Ct, typename Arg>
class Field; // forward declare to make function definition possible
 
template <AllocatorAwareContainerTp alloc_container_tp, typename Arg>
decltype(auto) create_field(const Arg &arg) {
  return Field<alloc_container_tp, Arg>(arg);
}

template <PMRStaggeredSpinorFieldTp field_tp, PMRContainerTp dst_container_tp = field_tp::container_tp, bool do_copy = false, bool is_exclusive = true>
decltype(auto) create_field_with_buffer(field_tp &src) {
  //
  using src_container_tp = field_tp::container_tp;
  
  using src_data_tp      = field_tp::data_tp;  
  using dst_data_tp      = dst_container_tp::value_type; 
  
  using ArgTp = decltype(src.ExportArg());   
  //
  auto arg = ArgTp{src.ExportArg()};//need new arg structure
  
  arg.template RegisterPMRBuffer<dst_data_tp, is_exclusive>();  
  
  auto& pmr_pool_handle = *arg.pmr_buffer->Pool();//?
  //
  auto dst = Field<dst_container_tp, decltype(arg)>(pmr_pool_handle, arg);
  
  if constexpr (do_copy){
    //
    auto &&dst_view = dst.View();
    auto &&src_view = src.View();
    //    
    if constexpr (std::is_same_v< src_container_tp, dst_container_tp >) {
      std::copy( std::execution::par_unseq, dst_view.Begin(), dst_view.End(), src_view.Begin());
    } else {
      std::transform(std::execution::par_unseq, src_view.Begin(), src_view.End(), dst_view.Begin(), [=](const auto &in) { return static_cast<dst_data_tp>(in); } );
    }
  }
  //
  return dst;
}

template <FieldTp field_tp, ContainerTp dst_container_tp = field_tp::container_tp, bool do_copy = false>
decltype(auto) create_field(field_tp &src) {
  //
  using src_container_tp = field_tp::container_tp;
  
  using src_data_tp      = field_tp::data_tp;  
  using dst_data_tp      = dst_container_tp::value_type;    
  //
  auto arg = src.ExportArg();
  //
  auto dst = Field<dst_container_tp, decltype(arg)>(arg);
  
  if constexpr (do_copy){
    //
    auto &&dst_view = dst.View();
    auto &&src_view = src.View();
    //    
    if constexpr (std::is_same_v< src_container_tp, dst_container_tp >) {
      std::copy( std::execution::par_unseq, dst_view.Begin(), dst_view.End(), src_view.Begin());
    } else {
      std::transform(std::execution::par_unseq, src_view.Begin(), src_view.End(), dst_view.Begin(), [=](const auto &in) { return static_cast<dst_data_tp>(in); } );
    }
  }
  //
  return dst;
}

template <PMRContainerTp pmr_container_tp, typename Arg, bool is_exclusive = true>
decltype(auto) create_field_with_buffer(const Arg &arg_, const bool use_reserved = false) {
  using data_tp = pmr_container_tp::value_type;

  auto arg = Arg{arg_};

  if ( not use_reserved) {// if not use a reserved buffer, register new one
    arg.template RegisterPMRBuffer<data_tp, is_exclusive>();
  } else {
  // use reserved, e.g., by a block field constructor, 
  // arg must have a valid pmr_buffer with PMRStatus::Reserved
    if ( not arg.template IsReservedPMR<data_tp>() ) {
      std::cerr << "Incorrect PMR buffer state, check reservation." << std::endl;
      std::quick_exit( EXIT_FAILURE );   
    }
  }
  
  if ( not arg.IsExclusive() ) std::cout << "Warning: creating non-exclusive PMR field." << std::endl;

  auto& pmr_pool_handle = *arg.pmr_buffer->Pool();

  return Field<pmr_container_tp, Arg>(pmr_pool_handle, arg);
}

template <GenericContainerTp generic_container_tp, typename Arg>
class Field{
  public:	
    //
    using container_tp  = generic_container_tp;        
    using data_tp       = typename container_tp::value_type;

    static consteval std::size_t Ndim()    { return Arg::ndim;    }    
    static consteval std::size_t Ndir()    { return Arg::ndir;    }    
    static consteval std::size_t Nspin()   { return Arg::nspin;   }                        
    static consteval std::size_t Ncolor()  { return Arg::ncolor;  }    
    static consteval std::size_t Nparity() { return Arg::nparity; }                         
    static consteval FieldType   Type()    { return Arg::type;    }                             

  private: 
    const Arg arg;//copy of the arguments  
    
    container_tp v;
    container_tp ghost;

    template<ContainerTp T = container_tp>
    explicit Field(const Arg &arg_) : arg(arg_),
                                      v(arg.GetFieldSize()),
                                      ghost(arg.GetGhostZoneSize()){ }

    template<PMRContainerTp T = container_tp>
    explicit Field(const Arg &arg_) : arg( [src_arg = arg_]()->Arg {
                                             auto dst_arg = Arg{src_arg};
                                             dst_arg.template RegisterPMRBuffer<data_tp, true>();
                                             return dst_arg;}() ),
                                      v(arg.GetFieldSize(), &(*arg.pmr_buffer->Pool())),
                                      ghost(arg.GetGhostZoneSize(), &(*arg.pmr_buffer->Pool())){ }

    // 
    template <PMRContainerTp T = container_tp>
    explicit Field(std::pmr::monotonic_buffer_resource &pmr_pool, const Arg &arg) : arg(arg),
    										    v(arg.GetFieldSize(), &pmr_pool), 
                                                                                    ghost(arg.GetGhostZoneSize(), &pmr_pool) { }

    template <AllocatorAwareContainerTp alloc_container_tp, typename ArgTp>
    friend decltype(auto) create_field(const ArgTp &arg);    
    
    template <FieldTp field_tp, ContainerTp container_tp, bool do_copy>
    friend decltype(auto) create_field(field_tp &src);

    template <PMRContainerTp pmr_container_tp, typename ArgTp, bool is_exclusive>
    friend decltype(auto) create_field_with_buffer(const ArgTp &arg_, const bool use_reserved);
    
    template <PMRStaggeredSpinorFieldTp field_tp, PMRContainerTp container_tp, bool do_copy>
    friend decltype(auto) create_field_with_buffer(field_tp &src);    

  public:

    Field()              = default;
    Field(const Field &) = default;
    Field(Field &&)      = default;    
    //
    template <ContainerViewTp T>    
    explicit Field(const T &src, const T &ghost_src, const Arg &arg) : arg(arg), v{src}, ghost{ghost_src} {}
    
    // Needed for block-la operations
    constexpr std::size_t size() const { return 1ul; } 
    
    static constexpr FieldType type() { return Arg::type; }
    
    // Needed for block-la operations
    decltype(auto) operator[](int i) const { 
      if(i != 0) exit(-1); 
      //
      return *this; 
    }        
    //
    void show() {
      printf("PMR buffer pointer:\t %p\nPMR buffer size:\t %ld (alinged size: %ld )\nPMR buffer use count:\t %d\n", this->arg.pmr_buffer->Get(), this->arg.pmr_buffer->BaseBytes(), this->arg.pmr_buffer->Bytes(), this->arg.pmr_buffer.use_count());
    }
    
    void destroy() {
      static_assert(is_allocator_aware_type<container_tp> or is_pmr_allocator_aware_type<container_tp>, "Cannot resize a non-owner field!");

      v.resize(0ul);
      ghost.resize(0ul); 
      //
      if constexpr ( std::is_same_v< typename container_tp::allocator_type,  std::pmr::polymorphic_allocator<data_tp> > ) {
        arg.ReleasePMRBuffer();
      }
    }

    decltype(auto) ExportArg() const { return Arg{arg}; }

    auto Begin() { return v.begin(); }
    auto End()   { return v.end();   }

    //Return a reference to the data container
    auto& Data( ) { return v; }

    //Return the data raw ptr
    auto Get( ) const { return v.data(); }    

    //Return a reference to the object (data access via std::span )
    decltype(auto) View() {
      static_assert(is_allocator_aware_type<container_tp> or is_pmr_allocator_aware_type<container_tp>, "Cannot reference a non-owner field!");

      return Field<std::span<data_tp>, decltype(arg)>(std::span{v}, std::span{ghost}, arg);	    
    }

    decltype(auto) ParityView(const FieldParity parity ) {// return a reference to the parity component
      static_assert(is_allocator_aware_type<container_tp> or is_pmr_allocator_aware_type<container_tp>, "Cannot reference a non-owner field!");
      //
      if constexpr (Nparity() != 2) {
        std::cerr << "Cannot get a parity component from a non-full field, exiting...\n" << std::endl;
	std::quick_exit( EXIT_FAILURE );
      }
      //           
      constexpr std::size_t nparity = 1;
      
      auto parity_arg = FieldDescriptor<Ndim(), Ndir(), Nspin(), Ncolor(), nparity>(this->arg, parity);
      //
      const auto parity_length = GetParityLength();
      const auto parity_offset = parity == FieldParity::EvenFieldParity ? 0 : parity_length;

      const auto ghost_parity_length = GetGhostParityLength();
      const auto ghost_parity_offset = parity == FieldParity::EvenFieldParity ? 0 : ghost_parity_length;

      return Field<std::span<data_tp>, decltype(parity_arg)>(std::span{v}.subspan(parity_offset, parity_length), std::span{ghost}.subspan(ghost_parity_offset, ghost_parity_length) , parity_arg);
    }

    auto Even() { return ParityView(FieldParity::EvenFieldParity );}
    //
    auto Odd()  { return ParityView(FieldParity::OddFieldParity  );}

    auto EODecompose() {
      static_assert(Nparity() == 2);

      return std::make_tuple(this->Even(), this->Odd());
    }   

    auto GetLength()                 const { return v.size(); }
    auto GetBytes()                  const { return v.size()*sizeof(data_tp); }
    auto GetParityLength()           const { return v.size() / Nparity(); }

    auto GetGhostParityLength()      const { return ghost.size() / Nparity(); }

    auto GetDims()                   const { return arg.GetLatticeDims(); }
    auto GetCBDims()                 const { return arg.GetParityLatticeDims(); }
    
    auto GetCommDims()               const { return arg.GetCommDims(); }    
    auto GetCommDims( const int i )  const { return arg.GetCommDims(i); }
    
    auto GetFaceSize( const int i )  const { return arg.GetFaceSize(i); }    
   
    auto GetFieldOrder()   const { return arg.order; } 
    auto GetFieldParity()  const { return arg.parity; } 

    auto GetFieldSubset()  const { return arg.GetFieldSubset(); }  

    const auto& GetMDStrides() const { return arg.GetMDStrides(); }

    const auto& GetGhostMDStrides() const { return arg.GetGhostMDStrides(); }
    
    void Info() const {
      std::cout << "Full field dimensions: " << std::endl;
      int i = 0;
      for(auto d : GetDims()) std::cout << i++ << " : " << d << std::endl;
      std::cout << "Parity components dimensions: " << std::endl;
      i = 0;
      for(auto d : GetCBDims()) std::cout << i++ << " : " << d << std::endl;      
    }  
    
    template<bool is_constant, std::size_t... dofs>
    inline decltype(auto) mdaccessor(const std::array<std::size_t, (Ndim() + sizeof...(dofs))> &strides) const {
           
      using dyn_indx_type = std::size_t;
    
      using DynMDMap  = stdex::layout_stride::mapping<stdex::extents<dyn_indx_type, 
                                                                     stdex::dynamic_extent, 
                                                                     stdex::dynamic_extent,
                                                                     stdex::dynamic_extent,
                                                                     stdex::dynamic_extent, 
                                                                     dofs...>>;
      using ExtentsMD = stdex::extents<dyn_indx_type, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent, dofs...>;
      
      const std::array X = GetCBDims();       
      
      if constexpr (is_constant){
        return stdex::mdspan<const data_tp, ExtentsMD, stdex::layout_stride>{
                    v.data(), DynMDMap{ExtentsMD{X[0], X[1], X[2], X[3]}, strides}} ;
      } else {
        return stdex::mdspan<data_tp, ExtentsMD, stdex::layout_stride>{
                   const_cast<data_tp*>(v.data()), DynMDMap{ExtentsMD{X[0], X[1], X[2], X[3]}, strides}};
      }                 
    }      

    //Direct field accessors (note that ncolor always 1, so no slicing for this dof):
    template<bool is_constant = false>    
    auto Accessor() const {
      //
      constexpr std::size_t nDir   = Ndir();
      constexpr std::size_t nSpin  = Nspin();//NoP            
      constexpr std::size_t nColor = Ncolor();                  
      constexpr int nParity        = Nparity(); 
      //

      const auto& stridesMD = GetMDStrides();

      if constexpr (Arg::type == FieldType::VectorFieldType) {      
        if constexpr (nParity == 1) {
          return mdaccessor<is_constant, nColor, nColor, nDir>(stridesMD);       
        } else {
          return mdaccessor<is_constant, nColor, nColor, nDir, nParity>(stridesMD);               
        }
      } else if constexpr (Arg::type == FieldType::StaggeredSpinorFieldType) {
        if constexpr (nParity == 1) {
          return mdaccessor<is_constant, nColor>(stridesMD);       
        } else {
          return mdaccessor<is_constant, nColor, nParity>(stridesMD);               
        }        
      }
    }    
//??
#if 0
    template<bool is_constant, std::size_t... dofs>
    inline decltype(auto) ghost_mdaccessor(std::array<std::size_t, (Ndim()-1) + sizeof...(dofs)> strides, const int comm_dir) const {    
      constexpr int nFace         = Arg::nFace();     
           
      using dyn_indx_type = std::size_t;

      using DynMDMap  = stdex::layout_stride::mapping<stdex::extents<dyn_indx_type, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent, dofs...>>;
      using ExtentsMD = stdex::extents<dyn_indx_type, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent, dofs...>;
          
      const auto X = GetCommDims(comm_dir); 
      
      const auto offset = comm_dir*GetFaceSize(comm_dir)*nFace;//     
      
      if constexpr (is_constant){
        return stdex::mdspan<const data_tp, ExtentsMD, stdex::layout_stride>{
                    ghost.data()+offset, DynMDMap{ExtentsMD{X}, strides}} ;
      } else {
        return stdex::mdspan<data_tp, ExtentsMD, stdex::layout_stride>{
                   const_cast<data_tp*>(ghost.data() + offset), DynMDMap{ExtentsMD{X}, strides}};
      }                  
    }      
#endif    
    //Direct field accessors (note that ncolor always 1, so no slicing for this dof):
    template<bool is_constant = false>    
    auto GhostAccessor(const int comm_dir) const {
      //
#if 0      
      constexpr std::size_t nDim  = Ndim() - 1;//only one direction            
      
      constexpr std::size_t nSpin = Nspin();      
      constexpr int nParity       = Nparity();
      
      constexpr std::size_t nFace = Arg::nFace();//only one direction                  

      const auto& stridesMD = GetGhostMDStrides();

      if constexpr (Arg::type == FieldType::VectorFieldType) {      
        if constexpr (nParity == 1) {
          return ghost_mdaccessor<is_constant>(stridesMD, comm_dir);       
        } else {
          return ghost_mdaccessor<is_constant, nParity>(stridesMD, comm_dir);               
        }
      } else {
        if constexpr (nParity == 1) {
          return ghost_mdaccessor<is_constant, nSpin, nFace>(stridesMD, comm_dir);       
        } else {
          return ghost_mdaccessor<is_constant, nSpin, nParity, nFace>(stridesMD, comm_dir);               
        }        
      }
#else
      return nullptr;
#endif            
    }    
    
};

