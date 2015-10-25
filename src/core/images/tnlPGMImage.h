/***************************************************************************
                          tnlPGMImage.h  -  description
                             -------------------
    begin                : Jul 20, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#ifndef TNLPGMIMAGE_H
#define	TNLPGMIMAGE_H

#include <core/tnlString.h>
#include <core/images/tnlImage.h>
#include <core/images/tnlRegionOfInterest.h>
#include <fstream>

template< typename Index = int >
class tnlPGMImage : public tnlImage< Index >
{
   public:
      
      typedef Index IndexType;
      
      tnlPGMImage();
       
      bool openForRead( const tnlString& fileName );
      
      template< typename Real,
                typename Device,
                typename Vector >
      bool read( const tnlRegionOfInterest< Index > roi,
                 const tnlGrid< 2, Real, Device, Index >& grid,
                 Vector& vector );
      
      template< typename Real,
                typename Device >
      bool openForWrite( const tnlString& fileName,
                         tnlGrid< 2, Real, Device, Index >& grid,
                         bool binary = true );
      
      template< typename Real,
                typename Device,
                typename Vector >
      bool write( const tnlGrid< 2, Real, Device, Index >& grid,
                  Vector& vector );

      
      void close();
      
      ~tnlPGMImage();
      
      protected:
         
         bool readHeader();
         
         template< typename Real,
                   typename Device >
         bool writeHeader( const tnlGrid< 2, Real, Device, Index >& grid,
                           bool binary );
         
         bool binary;
         
         IndexType maxColors;
         
         fstream file;
         
         bool fileOpen;
};

#include <core/images/tnlPGMImage_impl.h>

#endif	/* TNLPGMIMAGE_H */

