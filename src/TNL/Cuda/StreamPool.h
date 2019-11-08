/***************************************************************************
                          StreamPool.h  -  description
                             -------------------
    begin                : Oct 14, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <stdlib.h>
#include <unordered_map>

namespace TNL {
namespace Cuda {

#ifdef HAVE_CUDA
class StreamPool
{
   public:
      // stop the compiler generating methods of copy the object
      StreamPool( StreamPool const& copy ) = delete;
      StreamPool& operator=( StreamPool const& copy ) = delete;

      inline static StreamPool& getInstance()
      {
         static StreamPool instance;
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
      inline StreamPool()
      {
         atexit( StreamPool::free_atexit );
      }

      inline static void free_atexit( void )
      {
         StreamPool::getInstance().free();
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

} // namespace Cuda
} // namespace TNL

