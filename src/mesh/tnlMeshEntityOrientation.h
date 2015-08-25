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
      typedef typename tnlMeshConfigTraits< MeshConfig >::IdPermutationArrayAccessor IdPermutationArrayAccessor;

      IdPermutationArrayAccessor getSubvertexPermutation() const
      {
         return this->subvertexPermutation.subarray( 0, this->subvertexPermutation.getSize() );
      }

   private:
      typedef typename ConfigTraits<TConfig>::TLocalIndex                                                            TLocalIndex;
      typedef typename ConfigTraits<TConfig>::template SubentityTraits<TEntityTopology, Dim<0>>::TIdPermutationArray TIdPermutationArray;

      void setPermutationValue(TLocalIndex index, TLocalIndex value)
      {
         this->subvertexPermutation[ index ] = value;
      }

      IdPermutationArray this->subvertexPermutation;
};


#endif	/* TNLMESHENTITYORIENTATION_H */

