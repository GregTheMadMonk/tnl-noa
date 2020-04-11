/***************************************************************************
                          IndexPermutationApplier.h  -  description
                             -------------------
    begin                : Mar 10, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Pointers/DevicePointer.h>
#include <TNL/Algorithms/ParallelFor.h>

namespace TNL {
namespace Containers {
namespace Multimaps {

template< typename Multimap,
          typename PermutationArray >
void permuteMultimapKeys( Multimap& multimap, const PermutationArray& perm )
{
   static_assert( std::is_same< typename Multimap::DeviceType, typename PermutationArray::DeviceType >::value,
                  "The multimap and permutation vector must be stored on the same device." );
   using IndexType = typename Multimap::IndexType;
   using DeviceType = typename Multimap::DeviceType;
   TNL_ASSERT( multimap.getKeysRange() == perm.getSize(),
               std::cerr << "multimap keys range is " << multimap.getKeysRange()
                         << ", permutation size is " << perm.getSize() << std::endl; );

   // create temporary multimap for the permuted data
   Multimap multimapCopy;
   multimapCopy.setLike( multimap );

   // kernel to permute the rows of multimap into multimapCopy
   auto kernel = [] __cuda_callable__
      ( IndexType i,
        const Multimap* multimap,
        Multimap* multimapCopy,
        const typename PermutationArray::ValueType* perm )
   {
      const auto srcValues = multimap->getValues( perm[ i ] );
      auto destValues = multimapCopy->getValues( i );
      destValues = srcValues;
   };

   Pointers::DevicePointer< Multimap > multimapPointer( multimap );
   Pointers::DevicePointer< Multimap > multimapCopyPointer( multimapCopy );

   Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, multimap.getKeysRange(),
                                                kernel,
                                                &multimapPointer.template getData< DeviceType >(),
                                                &multimapCopyPointer.template modifyData< DeviceType >(),
                                                perm.getData() );

   // copy the permuted data back into the multimap
   multimap = multimapCopy;
}

template< typename Multimap,
          typename PermutationArray >
void permuteMultimapValues( Multimap& multimap, const PermutationArray& iperm )
{
   static_assert( std::is_same< typename Multimap::DeviceType, typename PermutationArray::DeviceType >::value,
                  "The multimap and permutation vector must be stored on the same device." );
   using IndexType = typename Multimap::IndexType;
   using DeviceType = typename Multimap::DeviceType;

   // kernel to permute the multimap values
   auto kernel = [] __cuda_callable__
      ( IndexType i,
        Multimap* multimap,
        const typename PermutationArray::ValueType* iperm )
   {
      auto values = multimap->getValues( i );
      for( typename Multimap::LocalIndexType v = 0; v < values.getSize(); v++ )
         values[ v ] = iperm[ values[ v ] ];
   };

   Pointers::DevicePointer< Multimap > multimapPointer( multimap );
   Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, multimap.getKeysRange(),
                                                kernel,
                                                &multimapPointer.template modifyData< DeviceType >(),
                                                iperm.getData() );
}

} // namespace Multimaps
} // namespace Containers
} // namespace TNL
