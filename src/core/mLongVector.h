/***************************************************************************
                          mLongVector.h  -  description
                             -------------------
    begin                : 2007/06/16
    copyright            : (C) 2007 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef mLongVectorH
#define mLongVectorH

#include <assert.h>
#include "tnlObject.h"
#include "param-types.h"

template< class T > class mLongVector : public tnlObject
{

   public:

   //! Constructor with given size
   mLongVector( int _size = 0 )
   : size( _size ), shared_data( false )
   {
      data = new T[ size + 1 ];
      if( ! data )
      {
         cerr << "Unable to allocate new long vector with size " << size << "." << endl;
         abort();
      }
      data ++;
   };

   //! Constructor with another long vector as template
   mLongVector( const mLongVector& v )
   : tnlObject( v ), size( v. size ), shared_data( false )
   {
      data = new T[ size + 1 ];
      if( ! data )
      {
         cerr << "Unable to allocate new long vector with size " << size << "." << endl;
         abort();
      }
      data ++;
   };
   
   mString GetType()
   {
      T t;
      return mString( "mLongVector< " ) + mString( GetParameterType( t ) ) + mString( " >" );
   };

   bool SetNewSize( int _size )
   {
      if( size == _size ) return true;
      if( ! shared_data )
      {
         delete[] -- data;
         data = 0;
      }
      size = _size;
      data = new T[ size + 1 ];
      shared_data = false;
      if( ! data )
      {
         cerr << "Unable to allocate new long vector with size " << size << "." << endl;
         size = 0;
         return false;
      }
      data ++;
      return true;
   };

   bool SetNewSize( const mLongVector< T >& v )
   {
      return SetNewSize( v. GetSize() );
   };

   void SetSharedData( T* _data, const int _size )
   {
      if( data && ! shared_data ) delete -- data;
      data = _data;
      shared_data = true;
      size = _size;
   };

   int GetSize() const
   {
      return size;
   };

   //! Returns pointer to data
   /*! This is not clear from the OOP point of view however it is necessary for keeping 
       good performance of derived numerical structure like solvers.
    */
   const T* Data() const
   {
      return data;
   };

   //! Returns pointer to data
   T* Data()
   {
      return data;
   }
  
   operator bool() const
   {
      return ( GetSize() != 0 );
   };
   
   T& operator[] ( int i )
   {
      assert( i < size );
      return data[ i ];
   };
   
   const T& operator[] ( int i ) const
   {
      assert( i < size );
      return data[ i ];
   };

   void Zeros()
   {
      int i;
      for( i = 0; i < size; i ++ ) data[ i ] = ( T ) 0;
   };

   ~mLongVector()
   {
      if( data && ! shared_data ) delete -- data;
   };

   //! Method for saving the object to a file as a binary data
   bool Save( ostream& file ) const
   {
      if( ! tnlObject :: Save( file ) ) return false;
      file. write( ( char* ) &size, sizeof( int ) );
      if( file. bad() ) return false;
      file. write( ( char* ) data, size * sizeof( T ) );
      if( file. bad() ) return false;
      return true;
   };

   //! Method for restoring the object from a file
   bool Load( istream& file )
   {
      if( ! tnlObject :: Load( file ) ) return false;
      int _size;
      file. read( ( char* ) &_size, sizeof( int ) );
      if( _size <= 0 )
      {
         cerr << "Error: The size " << _size << " of the file is not a positive number." << endl;
         return false;
      }
      if( file. bad() ) return false;
      if( _size != size )
      {
         size = _size;
         if( ! shared_data ) delete[] --data;
         data = new T[ size + 1 ];
         if( ! data )
         {
            cerr << "Unable to allocate new long vector with size " << size << "." << endl;
            return false;
         }
         data ++;
      }
      file. read( ( char* ) data, size * sizeof( T ) );
      if( file. bad() ) return false;
      return true;
   };   

   protected:

   int size;

   T* data;

   bool shared_data;
};

template< typename T > ostream& operator << ( ostream& o, const mLongVector< T >& v )
{
   int size = v. GetSize();
   int i;
   for( i = 0; i < size - 1; i ++ )
      o << v[ i ] << ",";
   o << v[ size - 1 ];
   return o;
};

template< typename T > void Copy( const mLongVector< T >& v1,
                                  mLongVector< T >& v2 )
{
   assert( v1. GetSize() == v2. GetSize() );
   memcpy( v2. Data(), v1. Data(), v1. GetSize() * sizeof( T ) );
};

// Explicit instatiation
template class mLongVector< double >;
template ostream& operator << ( ostream& o, const mLongVector< double >& v );


#endif
