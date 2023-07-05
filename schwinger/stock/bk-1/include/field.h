#pragma once

#include <common.h>
#include <enums.h>

template<int nSpin, int nDir>
class FieldArgs {
  public: 
    static constexpr int nspin  = nSpin;                   //number of spin dof (2 for spinor)
    static constexpr int ndir   = nDir;                    //vector field dim   (2 for U1 gauge) 							 
    const std::array<int, 2> dims;		
    const FieldSiteSubset    subset;
    //
    const FieldParity        parity;

    FieldArgs() = default;
    FieldArgs(const FieldArgs& ) = default;
    FieldArgs(FieldArgs&& )      = default;

    FieldArgs(const int L, 
	      const int T, 
	      const FieldSiteSubset subset   = FieldSiteSubset::FullSiteSubset,  
	      const FieldParity parity       = FieldParity::InvalidFieldParity) : 
	      dims{L, T},
	      subset(subset),
	      parity(parity) {} 

    FieldArgs(const FieldArgs &args, const FieldSiteSubset subset,  const FieldParity parity) : 
	    dims{subset == FieldSiteSubset::ParitySiteSubset && args.subset == FieldSiteSubset::FullSiteSubset ? args.dims[0] / 2 : args.dims[0], args.dims[1]},
	    subset(subset),
	    parity(parity) {}  

    auto operator=(const FieldArgs&) -> FieldArgs& = default;
    auto operator=(FieldArgs&&     ) -> FieldArgs& = default;
};

using GaugeFieldArgs  = FieldArgs<1,2>;
using SpinorFieldArgs = FieldArgs<2,1>;

template <GenericContainerTp container_tp, typename Arg>
class Field{
  public:	
    //
    static constexpr int nSpin   = Arg::nspin;                    
    static constexpr int nDir    = Arg::ndir;                    

  private: 
    container_tp v;

    const Arg arg;//copy of the arguments

  public:
    //
    Field(const Arg &arg) : v(arg.dims[0]*arg.dims[1]*nSpin*nDir), 
	                    arg(arg) {}
    Field(const container_tp &src, const Arg &arg) : v(src),
                            arg(arg) {}

    auto& Get( ) { return v; }

    auto Even() {
      if (arg.subset != FieldSiteSubset::FullSiteSubset) {
        std::cerr << "Cannot get a parity component from a non-full field\n" << std::endl;
      }
      using data_tp = typename container_tp::value_type;
      //
      auto even_arg = FieldArgs(this->arg, FieldSiteSubset::ParitySiteSubset, FieldParity::EvenFieldParity);
      //
      return Field<std::span<data_tp>, decltype(even_arg)>(std::span{v}.subspan(0, GetParityLength()), even_arg);
    }

    auto Odd() {
      if (arg.subset != FieldSiteSubset::FullSiteSubset) {
        std::cerr << "Cannot get a parity component from a non-full field\n" << std::endl;
      }
      using data_tp = typename container_tp::value_type;
      //
      auto odd_arg = FieldArgs(this->arg, FieldSiteSubset::ParitySiteSubset, FieldParity::OddFieldParity);
      //
      return Field<std::span<data_tp>, decltype(odd_arg)>(std::span{v}.subspan(GetParityLength(), GetParityLength()), odd_arg);
    }

    auto GetLength()       const { return v.size(); }
    auto GetParityLength() const { return v.size() / (arg.subset == FieldSiteSubset::FullSiteSubset ? 2 : 1); }
#if 0
    //List of accessors:
    auto GaugeAccessor() {
       using dyn_indx_type = int;
       using Dyn2DMap = stdex::layout_stride::mapping<stdex::extents<dyn_indx_type, stdex::dynamic_extent>>;

       return nullptr;	    
    }
#endif
};


