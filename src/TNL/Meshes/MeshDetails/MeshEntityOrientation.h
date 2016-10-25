/***************************************************************************
                          MeshEntityOrientation.h  -  description
                             -------------------
    begin                : Aug 25, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology>
class MeshEntityOrientation
{
   template< typename, typename>
   friend class MeshEntityReferenceOrientation;

   public:
      using LocalIndexType = typename MeshTraits< MeshConfig >::LocalIndexType;
      using IdPermutationArrayType = typename MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, 0 >::IdPermutationArrayType;

      const IdPermutationArrayType& getSubvertexPermutation() const
      {
         return subvertexPermutation;
      }

   private:
      void setPermutationValue( LocalIndexType index, LocalIndexType value )
      {
         this->subvertexPermutation[ index ] = value;
      }

      IdPermutationArrayType subvertexPermutation;
};

} // namespace Meshes
} // namespace TNL

