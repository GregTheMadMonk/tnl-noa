/***************************************************************************
                          tnlMeshEntityOrientation.h  -  description
                             -------------------
    begin                : Aug 25, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLMESHENTITYORIENTATION_H
#define	TNLMESHENTITYORIENTATION_H

#include <mesh/traits/tnlMeshTraits.h>

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


#endif	/* TNLMESHENTITYORIENTATION_H */

