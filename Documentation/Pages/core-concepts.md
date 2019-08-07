\page core_concepts  Core concepts

TNL is based on the following core concepts:

1. \ref TNL::Allocators "Allocators"
   - Allocator handles memory allocation and deallocation.
   - TNL allocators are fully compatible with the
     [standard C++ concept](https://en.cppreference.com/w/cpp/named_req/Allocator)
   - Multiple allocators can correspond to the same "memory space".
2. \ref TNL::Devices "Devices"
   (TODO: rename to `Executor` or something like that)
   - Device is responsible for the execution of algorithms in a specific way.
   - Algorithms can be specialized by the `Device` template parameter.
3. \ref TNL::Communicators "Communicators"
   - Communicators represent the main abstraction for distributed computations,
     where multiple programs (or instances of the same program) have to
     communicate with each other.
   - At present, there are only two communicators:
     \ref TNL::Communicators::MpiCommunicator "MpiCommunicator"
     (uses [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)) and
     \ref TNL::Communicators::NoDistrCommunicator "NoDistrCommunicator"
     (dummy communicator without any distribution support).
4. \ref TNL::Containers::Algorithms "Algorithms"
   - Basic (container-free) algorithms specialized by `Device`/`Executor`.
   - `ParallelFor`, `Reduction`, `MultiReduction`, `ArrayOperations`, ...
5. \ref TNL::Containers "Containers"
   - Classes for general data structures.
     (TODO: alternatively use "Dense" and "Sparse", because a dense matrix can
     be an extended alias for 2D array)
   - `Array`, `Vector` (also `VectorOperations`), `NDArray`, ...
6. Views
   - Views wrap only a raw pointer to data and some metadata (such as the array
     size), they do not do allocation and deallocation of the data. Hence, views
     have a fixed size which cannot be changed.
   - Views have a copy-constructor which does a shallow copy. As a result, views
     can be passed-by-value to CUDA kernels or captured-by-value by device
     lambda functions.
   - Views have a copy-assignment operator which does a deep copy.
   - Views have all other methods present in the relevant container (data
     structure).

TODO: formalize the concepts involving lambda functions (e.g. in `Reduction`)
