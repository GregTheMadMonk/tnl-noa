/***************************************************************************
                          tnlChunkedEllpackMatrixTestSetup.h  -  description
                             -------------------
    begin                : May 9, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLCHUNKEDELLPACKMATRIXTESTSETUP_H_
#define TNLCHUNKEDELLPACKMATRIXTESTSETUP_H_

#include "tnlSparseMatrixTester.h"

template< int SliceSize, int ChunkSize >
class tnlChunkedEllpackMatrixTestSetup
{
   public:

   enum { sliceSize = SliceSize };

   enum { chunkSize = ChunkSize };
};

template< typename Real,
          typename Device,
          typename Index,
          typename TestSetup >
class tnlSparseMatrixTesterMatrixSetter< tnlChunkedEllpackMatrix< Real, Device, Index >, TestSetup >
{
   public:

   typedef tnlChunkedEllpackMatrix< Real, Device, Index > Matrix;
 
   static bool setup( Matrix& matrix )
   {
      matrix.setNumberOfChunksInSlice( TestSetup::sliceSize );
      matrix.setDesiredChunkSize( TestSetup::chunkSize );
      return true;
   }
};


#endif /* TNLCHUNKEDELLPACKMATRIXTESTSETUP_H_ */
