#pragma once

#include <stdlib.h>
#include <unordered_map>

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {

#ifdef HAVE_CUDA
class CudaStreamPool
{
   public:
      // stop the compiler generating methods of copy the object
      CudaStreamPool( CudaStreamPool const& copy ) = delete;
      CudaStreamPool& operator=( CudaStreamPool const& copy ) = delete;

      inline static CudaStreamPool& getInstance()
      {
         static CudaStreamPool instance;
         return instance;
      }

      const cudaStream_t& getStream( int s )
      {
         auto result = pool.insert( {s, cudaStream_t()} );
         cudaStream_t& stream = (*result.first).second;
         bool& inserted = result.second;
         if( inserted ) {
            cudaStreamCreate( &stream );
         }
         return stream;
      }

   private:
      // private constructor of the singleton
      inline CudaStreamPool()
      {
         atexit( CudaStreamPool::free_atexit );
      }

      inline static void free_atexit( void )
      {
         CudaStreamPool::getInstance().free();
      }

   protected:
      using MapType = std::unordered_map< int, cudaStream_t >;

      inline void free( void )
      {
         for( auto& p : pool )
            cudaStreamDestroy( p.second );
      }

      MapType pool;
};
#endif

} // namespace TNL

