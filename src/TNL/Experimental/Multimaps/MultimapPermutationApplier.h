/***************************************************************************
                          IndexPermutationApplier.h  -  description
                             -------------------
    begin                : Mar 10, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/DevicePointer.h>
#include <TNL/ParallelFor.h>

namespace TNL {

template< typename Multimap,
          typename PermutationVector >
bool permuteMultimapKeys( Multimap& multimap, const PermutationVector& perm )
{
   static_assert( std::is_same< typename Multimap::DeviceType, typename PermutationVector::DeviceType >::value,
                  "The multimap and permutation vector must be stored on the same device." );
   using DeviceType = typename Multimap::DeviceType;
   TNL_ASSERT( multimap.getKeysRange() == perm.getSize(),
               std::cerr << "multimap keys range is " << multimap.getKeysRange()
                         << ", permutation size is " << perm.getSize() << std::endl; );

   // create temporary multimap for the permuted data
   Multimap multimapCopy;
   if( ! multimapCopy.setLike( multimap ) )
      return false;

   // kernel to permute the rows of multimap into multimapCopy
   auto kernel = [] __cuda_callable__
      ( typename Multimap::IndexType i,
        const Multimap* multimap,
        Multimap* multimapCopy,
        const typename PermutationVector::RealType* perm )
   {
      const auto srcValues = multimap->getValues( perm[ i ] );
      auto destValues = multimapCopy->getValues( i );
      destValues = srcValues;
   };

   DevicePointer< Multimap > multimapPointer( multimap );
   DevicePointer< Multimap > multimapCopyPointer( multimapCopy );

   ParallelFor< DeviceType >::exec( 0, multimap.getKeysRange(),
                                    kernel,
                                    &multimapPointer.template getData< DeviceType >(),
                                    &multimapCopyPointer.template modifyData< DeviceType >(),
                                    perm.getData() );

   // copy the permuted data back into the multimap
   multimap = multimapCopy;

   return true;
}

template< typename Multimap,
          typename PermutationVector >
bool permuteMultimapValues( Multimap& multimap, const PermutationVector& iperm )
{
   static_assert( std::is_same< typename Multimap::DeviceType, typename PermutationVector::DeviceType >::value,
                  "The multimap and permutation vector must be stored on the same device." );
   using DeviceType = typename Multimap::DeviceType;

   // kernel to permute the multimap values
   auto kernel = [] __cuda_callable__
      ( typename Multimap::IndexType i,
        Multimap* multimap,
        const typename PermutationVector::RealType* iperm )
   {
      auto values = multimap->getValues( i );
      for( typename Multimap::LocalIndexType v = 0; v < values.getSize(); v++ )
         values[ v ] = iperm[ values[ v ] ];
   };

   DevicePointer< Multimap > multimapPointer( multimap );
   ParallelFor< DeviceType >::exec( 0, multimap.getKeysRange(),
                                    kernel,
                                    &multimapPointer.template modifyData< DeviceType >(),
                                    iperm.getData() );
   return true;
}

} // namespace TNL
