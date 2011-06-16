/***************************************************************************
                          tnlLongVector.h  -  description
                             -------------------
    begin                : 2007/06/16
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef tnlLongVectorH
#define tnlLongVectorH

#include <assert.h>
#include <string.h>
#include <core/tnlObject.h>
#include <core/param-types.h>
#include <core/tnlFile.h>
#include <core/tnlStatistics.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#else
#include <iostream>
using namespace std;
#endif

template< typename Real, typename Index = int > class tnlLongVectorBase : public tnlObject
{

   public:

   //! Constructor with given size
   tnlLongVectorBase( const tnlString& name )
   : tnlObject( name ), size( 0 ), data( 0 ), shared_data( false )
   {
   };
   
   virtual tnlString getType() const = 0;

   virtual void setSharedData( Real* _data, const Index _size ) = 0;

   virtual bool setSize( Index _size ) = 0;

   virtual void reset() = 0;

   Index getSize() const
   {
      return size;
   };

   //! Returns pointer to data
   /*! This is not clear from the OOP point of view however it is necessary for keeping 
       good performance of derived numerical structure like solvers.
    */
   const Real* getVector() const
   {
      if( size == 0 )
         return 0;
      return data;
   };

   //! Returns pointer to data
   Real* getVector()
   {
      if( size == 0 )
         return 0;
      return data;
   }
  
   operator bool() const
   {
      return ( getSize() != 0 );
   };

   //! Method for saving the object to a file as a binary data
   virtual bool save( tnlFile& file ) const = 0;

   //! Method for restoring the object from a file
   virtual bool load( tnlFile& file ) = 0;

   //! This method measures data transfers done by this vector.
   /*!***
    * Everytime one touches this grid touches * size * sizeof( Real ) bytes are added
    * to transfered bytes in tnlStatistics.
    */
   void touch( long int touches = 1 ) const
   {
     // TODO: fix this
     //defaultTnlStatistics. addTransferedBytes( touches * getSize() * sizeof( Real ) );
   };

   virtual ~tnlLongVectorBase(){};

   protected:

   Index size;

   Real* data;

   bool shared_data;
};

template< typename Real, tnlDevice Device = tnlHost, typename Index = int > class tnlLongVector : public tnlLongVectorBase< Real, Index >
{

};

#endif
