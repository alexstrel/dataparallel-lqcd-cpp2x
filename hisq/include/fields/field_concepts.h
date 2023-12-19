#pragma once

#include <core/common.h>

#include <memory_resource>

// Generic Spinor Field type:
template <typename T>
concept GenericStaggeredSpinorFieldTp = requires{
  requires GenericContainerTp<typename T::container_tp>;
  requires (T::Nspin()   == invalid_spin);
  requires (T::Ncolor()  == 1ul or T::Ncolor() == 3ul);
  requires (T::Ndir()    == invalid_dir); 
  //we don't require a paticular parity property for the spinor fields
  //requires (T::Nparity() == invalid_parity or T::Nparity() == 1 or T::Nparity() == 2);     //we don't require a paticular parity property for the spinor fields
};

// Generic Gauge Field type:
template <typename T>
concept GenericGaugeFieldTp = requires{
  requires GenericContainerTp<typename T::container_tp>;
  requires (T::Nspin()   == invalid_spin);
  requires (T::Ncolor()  == 1ul or T::Ncolor() == 3ul);
  requires (T::Ndir()    >= 2ul and T::Ndir()  <= 4ul);
  requires (T::Nparity() == invalid_parity or T::Nparity() == 1 or T::Nparity() == 2);       
};

// Generic Field type :
template <typename T>
concept GenericFieldTp    = GenericStaggeredSpinorFieldTp<T> or GenericGaugeFieldTp<T>;

// Allocated field type
template <typename T>
concept FieldTp           = GenericFieldTp<T> and is_allocator_aware_type<typename T::container_tp>;

// Reference field type
template <typename T>
concept FieldViewTp       = GenericFieldTp<T> and is_memory_non_owning_type<typename T::container_tp>;

// Allocated field type
template <typename T>
concept StaggeredSpinorFieldTp     = GenericStaggeredSpinorFieldTp<T> and is_allocator_aware_type<typename T::container_tp>;

// Reference field type
template <typename T>
concept StaggeredSpinorFieldViewTp = GenericStaggeredSpinorFieldTp<T> and is_memory_non_owning_type<typename T::container_tp>;

// PMR spinor field type
template <typename T>
concept PMRStaggeredSpinorFieldTp  = GenericStaggeredSpinorFieldTp<T> and is_pmr_allocator_aware_type<typename T::container_tp>;

// Allocated field type
template <typename T>
concept GaugeFieldTp      = GenericGaugeFieldTp<T> and is_allocator_aware_type<typename T::container_tp>;

template <typename T>
concept GaugeFieldViewTp  = GenericGaugeFieldTp<T> and is_memory_non_owning_type<typename T::container_tp>;

// Spinor block Field concepts
template <typename T>
concept GenericBlockStaggeredSpinorFieldTp  = ContainerTp<typename T::block_container_tp> and GenericStaggeredSpinorFieldTp< typename std::remove_pointer< decltype( std::declval<typename T::block_container_tp>().data() ) >::type >;

template <typename T>
concept BlockStaggeredSpinorFieldTp       = ContainerTp<typename T::block_container_tp> and StaggeredSpinorFieldTp< typename std::remove_pointer< decltype( std::declval<typename T::block_container_tp>().data() ) >::type >;

template <typename T>
concept PMRBlockStaggeredSpinorFieldTp    = ContainerTp<typename T::block_container_tp> and PMRStaggeredSpinorFieldTp< typename std::remove_pointer< decltype( std::declval<typename T::block_container_tp>().data() ) >::type >;

template <typename T>
concept BlockStaggeredSpinorFieldViewTp   = ContainerViewTp<T> and StaggeredSpinorFieldViewTp< typename std::remove_pointer< decltype( std::declval<T>().data() ) >::type >;


// Some even more specialized concepts:

template<typename T> concept GenericStaggeredParitySpinorFieldTp = GenericStaggeredSpinorFieldTp<T> and requires { requires (T::Nparity() == 1); };
//
template<typename T> concept GenericStaggeredFullSpinorFieldTp   = GenericStaggeredSpinorFieldTp<T> and requires { requires (T::Nparity() == 2); };

template <typename T>
concept GenericBlockStaggeredParitySpinorFieldTp  = ContainerTp<typename T::block_container_tp> and GenericStaggeredParitySpinorFieldTp< typename std::remove_pointer< decltype( std::declval<typename T::block_container_tp>().data() ) >::type >;

template <typename T>
concept GenericBlockStaggeredFullSpinorFieldTp    = ContainerTp<typename T::block_container_tp> and GenericStaggeredFullSpinorFieldTp< typename std::remove_pointer< decltype( std::declval<typename T::block_container_tp>().data() ) >::type >;

//??
template <typename T>
concept GenericStaggeredSpinorFieldViewTp = StaggeredSpinorFieldViewTp<T> or BlockStaggeredSpinorFieldViewTp<T>;



