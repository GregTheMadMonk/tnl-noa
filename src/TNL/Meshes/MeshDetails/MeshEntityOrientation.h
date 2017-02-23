/***************************************************************************
                          MeshEntityOrientation.h  -  description
                             -------------------
    begin                : Aug 25, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology>
class MeshEntityOrientation
{
   template< typename, typename> friend class MeshEntityReferenceOrientation;

   public:
      typedef typename MeshTraits< MeshConfig >::IdPermutationArrayAccessorType IdPermutationArrayAccessorType;

      IdPermutationArrayAccessorType getSubvertexPermutation() const
      {
         IdPermutationArrayAccessorType accessor;
         accessor.bind( this->subvertexPermutation );
         return accessor;
         //return this->subvertexPermutation.subarray( 0, this->subvertexPermutation.getSize() );
      }

   private:
      typedef typename MeshTraits< MeshConfig >::LocalIndexType        LocalIndexType;
      typedef typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, 0 >::IdPermutationArrayType IdPermutationArrayType;

      void setPermutationValue( LocalIndexType index, LocalIndexType value )
      {
         this->subvertexPermutation[ index ] = value;
      }

      IdPermutationArrayType subvertexPermutation;
};

} // namespace Meshes
} // namespace TNL

