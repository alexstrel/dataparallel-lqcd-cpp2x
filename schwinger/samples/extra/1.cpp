#include <iostream>
#include <memory_resource>
#include <vector>

void useResource(std::pmr::monotonic_buffer_resource& resource) {
    // Use the memory resource with a polymorphic allocator
    std::pmr::vector<int> vec{&resource};

    // Add elements to the vector
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);

    // Print the vector elements
    printf("%p\n", vec.data());
    for (int element : vec) {
        std::cout << element << ' ';
    }
    std::cout << '\n';
}

int main() {
    std::pmr::monotonic_buffer_resource resource{4096};

    // Call the function that uses the memory resource
    useResource(resource);

    // Reset the memory resource
    resource.release();

    // Call the function again with the reset memory resource (that is, now vec container will re-use the buffer!)
    useResource(resource);

    return 0;
}


