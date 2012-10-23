/***************************************************************************
                          tnlArrayManagerBase.h -  description
                             -------------------
    begin                : Jul 4, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLARRAYMANAGERBASE_H_
#define TNLARRAYMANAGERBASE_H_

#include <core/tnlObject.h>
#include <core/param-types.h>
#include <core/tnlFile.h>

template< typename ElementType, typename IndexType >
class tnlArrayManagerBase : public tnlObject

{
   tnlArrayManagerBase(){};

   public:
   //! Constructor with name parameter.
   tnlArrayManagerBase( const tnlString& name );

   virtual tnlString getType() const = 0;

   /*!
    * Use this method to point this array to already allocated data.
    * It is useful when TNL is combined with some other library. In this
    * case no data is freed by TNL in destructor of this array.
    */
   virtual void setSharedData( ElementType* _data, const IndexType _size ) = 0;

   /*!
    * This method is used for setting the size of the array and in
    * fact it is the main way how to allocate blacks of data in TNL.
    */
   virtual bool setSize( IndexType _size ) = 0;

   //! Array size getter.
   IndexType getSize() const;

   //! Free allocated memory.
   virtual void reset() = 0;

   /*!
    * Returns pointer to data. This is not clear from the OOP point of view. However,
    * it is necessary for keeping good performance of derived numerical structure
    * like solvers.
    */
   const ElementType* getData() const;

   /*!
    * Returns pointer to data
    */
   ElementType* getData();

   /*!
    * Returns true if non-zero size is set.
    */
   operator bool() const;

   /*!
    * If an array on different device like GPU is managed this set whether
    * memory transfers are automatically synchronized or not. If not, the
    * performance is usually better but one must be careful. By default, the
    * safe mode is OFF.
    */
   void setSafeMode( bool mode );

   /*!
    * This method serves for checking memory transaction mode. See the method
    * setSafeMode.
    */
   bool getSafeMode() const;

   //! This method measures data transfers done by this vector.
   /*!
    * Every time one touches this grid touches * size * sizeof( Real ) bytes are added
    * to transfered bytes in tnlStatistics.
    */
   void touch( long int touches = 1 ) const;

   //! Method for saving the object to a file as a binary data.
   virtual bool save( tnlFile& file ) const = 0;

   //! Method for restoring the object from a file.
   virtual bool load( tnlFile& file ) = 0;

   protected:

   //!Number of allocated elements
   IndexType size;

   //! Pointer to allocated data
   ElementType* data;

   /*!
    * The data can be shared with other object. Typically this happens
    * when TNL is used together with another numerical library. If
    * the data is shared it is not freed in destructor.
    */
   bool shared_data;

   /*!
    * In safe mode, all memory transactions concerning arrays
    * on different device are automatically synchronized.
    */
   bool safeMode;

};

template< typename ElementType, typename Device = tnlHost, typename IndexType = int >
class tnlArrayManager : public tnlArrayManagerBase< ElementType, IndexType >
{

};


template< typename ElementType, typename IndexType >
tnlArrayManagerBase< ElementType, IndexType > :: tnlArrayManagerBase( const tnlString& name )
: tnlObject( name ), size( 0 ), data( 0 ), shared_data( false ), safeMode( true )
{
};

template< typename ElementType, typename IndexType >
IndexType tnlArrayManagerBase< ElementType, IndexType > :: getSize() const
{
   return size;
};

template< typename ElementType, typename IndexType >
const ElementType* tnlArrayManagerBase< ElementType, IndexType > :: getData() const
{
   if( size == 0 )
      return 0;
   return data;
};

template< typename ElementType, typename IndexType >
ElementType* tnlArrayManagerBase< ElementType, IndexType > :: getData()
{
   if( size == 0 )
      return 0;
   return data;
}

template< typename ElementType, typename IndexType >
tnlArrayManagerBase< ElementType, IndexType > :: operator bool() const
{
   return ( getSize() != 0 );
};

template< typename ElementType, typename IndexType >
void tnlArrayManagerBase< ElementType, IndexType > :: setSafeMode( bool mode )
{
   safeMode = mode;
}

template< typename ElementType, typename IndexType >
bool tnlArrayManagerBase< ElementType, IndexType > :: getSafeMode() const
{
   return safeMode;
}

template< typename ElementType, typename IndexType >
void tnlArrayManagerBase< ElementType, IndexType > :: touch( long int touches ) const
{
  // TODO: fix this
  //defaultTnlStatistics. addTransferedBytes( touches * getSize() * sizeof( Real ) );
};

#endif /* TNLARRAYMANAGERBASE_H_ */
