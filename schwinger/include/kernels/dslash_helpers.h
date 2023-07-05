#pragma once

template< ComplexTp T, std::size_t nspin>
inline decltype(auto) operator*(const T &scalar, const std::array<T,nspin> &a){
  std::array<T,nspin> result;
#pragma unroll
  for(int i = 0; i < nspin; i++){
    result[i] = scalar * a[i];
  }

  return result;
}

template< ComplexTp T, std::size_t nspin>
inline decltype(auto) operator+=(std::array<T,nspin> &a, const std::array<T,nspin> &b){

#pragma unroll
  for(int i = 0; i < nspin; i++){
    a[i] = a[i] + b[i];
  }

  return a;
}

template< ComplexTp T, std::size_t nspin>
inline decltype(auto) operator+(const std::array<T,nspin> &a, const std::array<T,nspin> &b){
  std::array<T,nspin> result;
#pragma unroll
  for(int i = 0; i < nspin; i++){
    result[i] = a[i] + b[i];
  }

  return result;
}

template< ComplexTp T, typename Link>
inline Link operator*(const T &a, const Link &U){
  return (a*U);
}

