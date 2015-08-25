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

template< typename MeshConfig,
          typename EntityTopology>
class tnlMeshEntityOrientation
{
   template< typename, typename> friend class tnlMeshEntityReferenceOrientation;

   public:
      typedef typename tnlMeshConfigTraits< MeshConfig >::IdPermutationArrayAccessorType IdPermutationArrayAccessorType;

      IdPermutationArrayAccessorType getSubvertexPermutation() const
      {
         return this->subvertexPermutation.subarray( 0, this->subvertexPermutation.getSize() );
      }

   private:
      typedef typename tnlMeshConfigTraits< MeshConfig >::LocalIndexType        LocalIndexType;
      typedef typename tnlMeshConfigTraits< MeshConfig >::template SubentityTraits< EntityTopology, tnlDimensionsTag< 0 > >::IdPermutationArrayType IdPermutationArrayType;

      void setPermutationValue( LocalIndexType index, LocalIndexType value )
      {
         this->subvertexPermutation[ index ] = value;
      }

      IdPermutationArrayType subvertexPermutation;
};


#endif	/* TNLMESHENTITYORIENTATION_H */

