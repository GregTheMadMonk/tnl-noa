/***************************************************************************
                          tnlMeshEntityOrientation.h  -  description
                             -------------------
    begin                : Aug 25, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <mesh/traits/tnlMeshTraits.h>

namespace TNL {

template< typename MeshConfig,
          typename EntityTopology>
class tnlMeshEntityOrientation
{
   template< typename, typename> friend class tnlMeshEntityReferenceOrientation;

   public:
      typedef typename tnlMeshTraits< MeshConfig >::IdPermutationArrayAccessorType IdPermutationArrayAccessorType;

      IdPermutationArrayAccessorType getSubvertexPermutation() const
      {
         IdPermutationArrayAccessorType accessor;
         accessor.bind( this->subvertexPermutation );
         return accessor;
         //return this->subvertexPermutation.subarray( 0, this->subvertexPermutation.getSize() );
      }

   private:
      typedef typename tnlMeshTraits< MeshConfig >::LocalIndexType        LocalIndexType;
      typedef typename tnlMeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, 0 >::IdPermutationArrayType IdPermutationArrayType;

      void setPermutationValue( LocalIndexType index, LocalIndexType value )
      {
         this->subvertexPermutation[ index ] = value;
      }

      IdPermutationArrayType subvertexPermutation;
};

} // namespace TNL

