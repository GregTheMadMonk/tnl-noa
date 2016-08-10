/***************************************************************************
                          ChunkedEllpackTestSetup.h  -  description
                             -------------------
    begin                : May 9, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef ChunkedEllpackTESTSETUP_H_
#define ChunkedEllpackTESTSETUP_H_

#include "tnlSparseMatrixTester.h"

using namespace TNL;

template< int SliceSize, int ChunkSize >
class ChunkedEllpackTestSetup
{
   public:

   enum { sliceSize = SliceSize };

   enum { chunkSize = ChunkSize };
};

template< typename Real,
          typename Device,
          typename Index,
          typename TestSetup >
class SparseTesterMatrixSetter< Matrices::ChunkedEllpack< Real, Device, Index >, TestSetup >
{
   public:

   typedef Matrices::ChunkedEllpack< Real, Device, Index > Matrix;
 
   static bool setup( Matrix& matrix )
   {
      matrix.setNumberOfChunksInSlice( TestSetup::sliceSize );
      matrix.setDesiredChunkSize( TestSetup::chunkSize );
      return true;
   }
};

#endif /* ChunkedEllpackTESTSETUP_H_ */
