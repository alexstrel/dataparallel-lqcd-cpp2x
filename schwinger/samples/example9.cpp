
// Example Polymorphic Memory Resource (std::pmr) Monotonic Buffer Resource.
// // J. Arrieta, Nabla Zero Labs
// //
// // * Pre-allocate two buffers, one with space for 2 chars and one with space for 256 chars.
// // * Filled pre-allocated buffers with underscores to peek into them later and see where memory was touched.
// // * Create two monotonic_buffer_resources, one for each buffer.
// // * Notice that the main pool receives a pointer to the "backup" pool.
// // * Create a vector of char that uses the main memory resource.
// // * Push back on the vector all chars from 'a' to 'z'.
// // * See how the first char was allocated using the first memory resource (pool).
// // * Subsequent pushbacks relied on the back-up memory resource.
// // * The vector contains the correct data.
// // * If `backup` is smaller (try), the vector would have started allocating on the heap




#include <memory_resource>
#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>

int main() {
  std::size_t buffer_size = 2;
  std::size_t backup_size = 256;

  auto buffer = std::make_shared<std::byte[]>(buffer_size);
  auto backup = std::make_shared<std::byte[]>(backup_size);

  //std::fill_n(buffer.get(), buffer_size, '_');
  //std::fill_n(backup.get(), backup_size, '_');

  auto upst = std::pmr::monotonic_buffer_resource(backup.get(), backup_size);
  auto pool = std::pmr::monotonic_buffer_resource(buffer.get(), buffer_size, &upst);
  
  std::pmr::vector<char> v(&pool);
  
  for (auto c = 'a'; c <= 'z'; ++c) {
      v.push_back(c);
  }
  std::cout << "main buffer contents:\n";
  //std::for_each_n(buffer.get(), buffer_size, [idx=0]  (auto x) mutable { std::cout << (idx++) << " " << x << "\n";});
  std::cout << "backup buffer contents:\n";
  //std::for_each_n(backup.get(), backup_size, [idx=0]  (auto x) mutable { std::cout << (idx++) << " " << x << "\n";});
  std::cout << "vector contents:\n";
  std::for_each(v.begin(), v.end(), [idx=0]  (auto x) mutable { std::cout << (idx++) << " " << x << "\n";});
}

