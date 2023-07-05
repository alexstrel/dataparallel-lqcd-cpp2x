#pragma once

#include <core/common.h>

#include <memory_resource>

// Generic Spinor Field type:
template <typename T>
concept GenericSpinorFieldTp = requires{
  requires GenericContainerTp<typename T::container_tp>;
  requires (T::Nspin()   == 1ul or T::Nspin()  == 2ul or T::Nspin()  == 4ul);
  requires (T::Ncolor()  == 1ul or T::Ncolor() == 3ul);
  requires (T::Ndir()    == invalid_dir); 
  requires (T::Nparity() == invalid_parity or T::Nparity() == 1 or T::Nparity() == 2);     
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
concept GenericFieldTp    = GenericSpinorFieldTp<T> or GenericGaugeFieldTp<T>;

// Allocated field type
template <typename T>
concept FieldTp           = GenericFieldTp<T> and is_allocator_aware_type<typename T::container_tp>;

// Reference field type
template <typename T>
concept FieldViewTp       = GenericFieldTp<T> and is_memory_non_owning_type<typename T::container_tp>;

// Allocated field type
template <typename T>
concept SpinorFieldTp     = GenericSpinorFieldTp<T> and is_allocator_aware_type<typename T::container_tp>;

// Reference field type
template <typename T>
concept SpinorFieldViewTp = GenericSpinorFieldTp<T> and is_memory_non_owning_type<typename T::container_tp>;

// PMR spinor field type
template <typename T>
concept PMRSpinorFieldTp  = GenericSpinorFieldTp<T> and is_pmr_allocator_aware_type<typename T::container_tp>;

// Allocated field type
template <typename T>
concept GaugeFieldTp      = GenericGaugeFieldTp<T> and is_allocator_aware_type<typename T::container_tp>;

template <typename T>
concept GaugeFieldViewTp  = GenericGaugeFieldTp<T> and is_memory_non_owning_type<typename T::container_tp>;

// Spinor block Field concepts
template <typename T>
concept GenericBlockSpinorFieldTp  = ContainerTp<typename T::block_container_tp> and GenericSpinorFieldTp< typename std::remove_pointer< decltype( std::declval<typename T::block_container_tp>().data() ) >::type >;

template <typename T>
concept BlockSpinorFieldTp       = ContainerTp<typename T::block_container_tp> and SpinorFieldTp< typename std::remove_pointer< decltype( std::declval<typename T::block_container_tp>().data() ) >::type >;

template <typename T>
concept PMRBlockSpinorFieldTp    = ContainerTp<typename T::block_container_tp> and PMRSpinorFieldTp< typename std::remove_pointer< decltype( std::declval<typename T::block_container_tp>().data() ) >::type >;

template <typename T>
concept BlockSpinorFieldViewTp   = ContainerViewTp<T> and SpinorFieldViewTp< typename std::remove_pointer< decltype( std::declval<T>().data() ) >::type >;


template <typename T>
concept GenericSpinorFieldViewTp = SpinorFieldViewTp<T> or BlockSpinorFieldViewTp<T>;



