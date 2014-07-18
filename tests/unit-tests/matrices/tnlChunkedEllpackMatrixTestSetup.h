/***************************************************************************
                          tnlChunkedEllpackMatrixTestSetup.h  -  description
                             -------------------
    begin                : May 9, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLCHUNKEDELLPACKMATRIXTESTSETUP_H_
#define TNLCHUNKEDELLPACKMATRIXTESTSETUP_H_

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
