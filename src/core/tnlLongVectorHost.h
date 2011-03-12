/***************************************************************************
                          tnlLongVectorHost.h
                             -------------------
    begin                : Jun 9, 2010
    copyright            : (C) 2009 by Tomas Oberhuber
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

#ifndef TNLLONGVECTORHOST_H_
#define TNLLONGVECTORHOST_H_

#include <core/mfuncs.h>
#include <core/tnlAssert.h>
#include <core/tnlLongVectorBase.h>

template< typename Real, typename Index > class tnlLongVector< Real, tnlCuda, Index >;

template< typename Real, typename Index > class tnlLongVector< Real, tnlHost, Index > : public tnlLongVectorBase< Real, Index >
{
   //! We do not allow constructor without parameters.
   tnlLongVector(){};

   public:

   //! Basic constructor with given size
   tnlLongVector( const tnlString& name, Index _size = 0 );

   //! Constructor with another long vector as template
   tnlLongVector( const tnlString& name, const tnlLongVector< Real, tnlHost, Index >& v );

   //! Constructor with another long vector as template
   tnlLongVector( const tnlString& name, const tnlLongVector< Real, tnlCuda, Index >& v );

   //! Use this if you want to insert some data in this vector.
   /*! The data will not be deallocated by the destructor.
    *  Once setSize method is called the vector forgets the shared data.
    */
   virtual void setSharedData( Real* _data, const Index _size );

   //! Set size of the vector and allocate necessary amount of the memory.
   bool setSize( Index _size );

   //! Set size of the vector using another vector as a template
   bool setLike( const tnlLongVector< Real, tnlHost, Index >& v );

   //! Set size of the vector using another vector as a template
   bool setLike( const tnlLongVector< Real, tnlCuda, Index >& v );

   void swap( tnlLongVector< Real, tnlHost, Index >& u );

   //! Returns type of this vector written in a form of C++ template type.
   tnlString getType() const;

   void setElement( Index i, Real d );

   Real getElement( Index i ) const;

   Real& operator[] ( Index i );

   const Real& operator[] ( Index i ) const;

   bool operator == ( const tnlLongVector< Real, tnlHost, Index >& long_vector ) const;

   bool operator != ( const tnlLongVector< Real, tnlHost, Index >& long_vector ) const;

   tnlLongVector< Real, tnlHost, Index >& operator = ( const tnlLongVector< Real, tnlHost, Index >& long_vector );

   tnlLongVector< Real, tnlHost, Index >& operator = ( const tnlLongVector< Real, tnlCuda, Index >& cuda_vector );

   template< typename Real2, typename Index2 >
   tnlLongVector< Real, tnlHost, Index >& operator = ( const tnlLongVector< Real2, tnlHost, Index2 >& long_vector );

   void setValue( const Real& v );

    //! Method for saving the object to a file as a binary data
    bool save( tnlFile& file ) const;

    //! Method for restoring the object from a file
    bool load( tnlFile& file );

    virtual ~tnlLongVector();
};

template< typename Real, typename Index >
ostream& operator << ( ostream& str, const tnlLongVector< Real, tnlHost, Index >& vec );

/****
 * Here are some Blas style functions. They are not methods
 * because it would put too many restrictions on type Real.
 * In fact, we would like to use tnlLongVector to store objects
 * like edges or triangles in case of meshes. For these objects
 * operations like +, min or max are not defined.
 */

template< typename Real, typename Index >
Real tnlMax( const tnlLongVector< Real, tnlHost, Index >& v );

template< typename Real, typename Index >
Real tnlMin( const tnlLongVector< Real, tnlHost, Index >& v );

template< typename Real, typename Index >
Real tnlAbsMax( const tnlLongVector< Real, tnlHost, Index >& v );

template< typename Real, typename Index >
Real tnlAbsMin( const tnlLongVector< Real, tnlHost, Index >& v );

template< typename Real, typename Index >
Real tnlLpNorm( const tnlLongVector< Real, tnlHost, Index >& v, const Real& p );

template< typename Real, typename Index >
Real tnlSum( const tnlLongVector< Real, tnlHost, Index >& v );

template< typename Real, typename Index >
Real tnlDifferenceMax( const tnlLongVector< Real, tnlHost, Index >& u,
                       const tnlLongVector< Real, tnlHost, Index >& v );

template< typename Real, typename Index >
Real tnlDifferenceMin( const tnlLongVector< Real, tnlHost, Index >& u,
                       const tnlLongVector< Real, tnlHost, Index >& v );

template< typename Real, typename Index >
Real tnlDifferenceAbsMax( const tnlLongVector< Real, tnlHost, Index >& u,
                          const tnlLongVector< Real, tnlHost, Index >& v );

template< typename Real, typename Index >
Real tnlDifferenceAbsMin( const tnlLongVector< Real, tnlHost, Index >& u,
                          const tnlLongVector< Real, tnlHost, Index >& v );

template< typename Real, typename Index >
Real tnlDifferenceLpNorm( const tnlLongVector< Real, tnlHost, Index >& u,
                          const tnlLongVector< Real, tnlHost, Index >& v, const Real& p );

template< typename Real, typename Index >
Real tnlDifferenceSum( const tnlLongVector< Real, tnlHost, Index >& u,
                       const tnlLongVector< Real, tnlHost, Index >& v );

template< typename Real, typename Index >
void tnlScalarMultiplication( const Real& alpha,
                              tnlLongVector< Real, tnlHost, Index >& u );

//! Compute scalar dot product
template< typename Real, typename Index >
Real tnlSDOT( const tnlLongVector< Real, tnlHost, Index >& u ,
              const tnlLongVector< Real, tnlHost, Index >& v );

//! Compute SAXPY operation (Scalar Alpha X Pus Y ).
template< typename Real, typename Index >
void tnlSAXPY( const Real& alpha,
               const tnlLongVector< Real, tnlHost, Index >& x,
               tnlLongVector< Real, tnlHost, Index >& y );

//! Compute SAXMY operation (Scalar Alpha X Minus Y ).
/*!**
 * It is not a standart BLAS function but is useful for GMRES solver.
 */
template< typename Real, typename Index >
void tnlSAXMY( const Real& alpha,
               const tnlLongVector< Real, tnlHost, Index >& x,
               tnlLongVector< Real, tnlHost, Index >& y );


template< typename Real, typename Index >
tnlLongVector< Real, tnlHost, Index > :: tnlLongVector( const tnlString& name, Index _size )
: tnlLongVectorBase< Real, Index >( name )
{
   setSize( _size );
};

template< typename Real, typename Index >
tnlLongVector< Real, tnlHost, Index > :: tnlLongVector( const tnlString& name, const tnlLongVector& v )
: tnlLongVectorBase< Real, Index >( name )
{
  setSize( v. getSize() );
};

template< typename Real, typename Index >
tnlLongVector< Real, tnlHost, Index > :: tnlLongVector( const tnlString& name, const tnlLongVector< Real, tnlCuda, Index >& v )
: tnlLongVectorBase< Real, Index >( name )
{
  setSize( v. getSize() );
};

template< typename Real, typename Index > bool tnlLongVector< Real, tnlHost, Index > :: setSize( Index _size )
{
   tnlAssert( _size >= 0,
            cerr << "You try to set size of tnlLongVector to negative value."
                 << "Vector name: " << this -> getName() << endl
                 << "New size: " << _size << endl );
   /* In the case that we run without active macro tnlAssert
    * we will write at least warning.
    */
   if( _size < 0 )
   {
      cerr << "Negative size " << _size << " was passed to tnlLongVector " << this -> getName() << "." << endl;
      return false;
   }
   if( this -> size == _size && ! this -> shared_data ) return true;
   if( this -> data && ! this -> shared_data )
   {
      delete[] -- this -> data;
      this -> data = 0;
   }
   this -> size = _size;
   this -> data = new Real[ this -> size + 1 ];
   this -> shared_data = false;
   if( ! this -> data )
   {
      cerr << "Unable to allocate new long vector " << this -> getName() << " with size " << this -> size << "." << endl;
      this -> size = 0;
      return false;
   }
   this -> data ++;
   return true;
};

template< typename Real, typename Index >
void tnlLongVector< Real, tnlHost, Index > :: setSharedData( Real* _data, const Index _size )
{
   /****
    * First lets touch the last element to see what happens.
    * If the size is not set properly we will get SIGSEGV.
    * It is better to find it out as soon as possible.
    */
   Real a = _data[ _size - 1 ];

   if( this -> data && ! this -> shared_data ) delete -- this -> data;
   this -> data = _data;
   this -> shared_data = true;
   this -> size = _size;
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlHost, Index > :: setLike( const tnlLongVector< Real, tnlHost, Index >& v )
{
   return setSize( v. getSize() );
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlHost, Index > :: setLike( const tnlLongVector< Real, tnlCuda, Index >& v )
{
   return setSize( v. getSize() );
};

template< typename Real, typename Index >
void tnlLongVector< Real, tnlHost, Index > :: swap( tnlLongVector< Real, tnlHost, Index >& v )
{
   tnlAssert( this -> getSize() > 0, );
   tnlAssert( this -> getSize() == v. getSize(),
              cerr << "You try to swap two long vectors with different sizes." << endl
                   << "The first one is " << this -> getName() << " with size " << this -> getSize()
                   << " while the second one is " << v. getName() << " with size " << v. getSize() << "." );

   std :: swap( this -> data, v. data );
   std :: swap( this -> shared_data, v. shared_data );
};

template< typename Real, typename Index >
tnlString tnlLongVector< Real, tnlHost, Index > :: getType() const
{
   return tnlString( "tnlLongVector< " ) + tnlString( GetParameterType( Real() ) ) + tnlString( ", tnlHost >" );
};

template< typename Real, typename Index >
void tnlLongVector< Real, tnlHost, Index > :: setElement( Index i, Real d )
{
   tnlAssert( this -> size != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( i < this -> size,
            cerr << "You try to set non-existing element of the following vector."
                 << "Name: " << this -> getName() << endl
                 << "Size: " << this -> size << endl
                 << "Element number: " << i << endl; );
   this -> data[ i ] = d;
};

template< typename Real, typename Index >
Real tnlLongVector< Real, tnlHost, Index > :: getElement( Index i ) const
{
   tnlAssert( this -> size != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( i < this -> size,
            cerr << "You try to get non-existing element of the following vector."
                 << "Name: " << this -> getName() << endl
                 << "Size: " << this -> size << endl
                 << "Element number: " << i << endl; );
   return this -> data[ i ];
};

template< typename Real, typename Index >
Real& tnlLongVector< Real, tnlHost, Index > :: operator[] ( Index i )
{
    tnlAssert( i < this -> size,
           cerr << "Name: " << this -> getName() << endl
                << "Size: " << this -> size << endl
                << "i = " << i << endl; );
   return this -> data[ i ];
};

template< typename Real, typename Index >
const Real& tnlLongVector< Real, tnlHost, Index > :: operator[] ( Index i ) const
{
   tnlAssert( i < this -> size,
           cerr << "Name: " << this -> getName() << endl
                << "Size: " << this -> size << endl
                << "i = " << i << endl; );
   return this -> data[ i ];
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlHost, Index > :: operator == ( const tnlLongVector< Real, tnlHost, Index >& long_vector ) const
{
   tnlAssert( this -> getSize() > 0, );
   tnlAssert( this -> getSize() == long_vector. getSize(),
              cerr << "You try to compare two long vectors with different sizes." << endl
                   << "The first one is " << this -> getName() << " with size " << this -> getSize() 
                   << " while the second one is " << long_vector. getName() << " with size " << long_vector. getSize() << "." );
   if( memcmp( this -> getVector(), long_vector. getVector(), this -> getSize() * sizeof( Real ) ) == 0 )
      return true;
   return false;
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlHost, Index > :: operator != ( const tnlLongVector< Real, tnlHost, Index >& long_vector ) const
{
   return ! ( ( *this ) == long_vector );
};

template< typename Real, typename Index >
tnlLongVector< Real, tnlHost, Index >& tnlLongVector< Real, tnlHost, Index > :: operator =( const tnlLongVector< Real, tnlHost, Index >& long_vector )
{
   tnlAssert( long_vector. getSize() == this -> getSize(),
           cerr << "Source name: " << long_vector. getName() << endl
                << "Source size: " << long_vector. getSize() << endl
                << "Target name: " << this -> getName() << endl
                << "Target size: " << this -> getSize() << endl );

   memcpy( this -> data,
           long_vector. getVector(),
           this -> getSize() * sizeof( Real ) );
   return ( *this );
};

template< typename Real, typename Index >
   template< typename Real2, typename Index2 >
tnlLongVector< Real, tnlHost, Index >& tnlLongVector< Real, tnlHost, Index > :: operator = ( const tnlLongVector< Real2, tnlHost, Index2 >& long_vector )
{
   tnlAssert( long_vector. getSize() == this -> getSize(),
           cerr << "Source name: " << long_vector. getName() << endl
                << "Source size: " << long_vector. getSize() << endl
                << "Target name: " << this -> getName() << endl
                << "Target size: " << this -> getSize() << endl );

   for( Index i = 0; i < this -> getSize(); i ++ )
      this -> data[ i ] = ( Real ) long_vector. getElement( ( Index2 ) i );
   return ( *this );
};


template< typename Real, typename Index >
tnlLongVector< Real, tnlHost, Index > :: ~tnlLongVector()
{
   if( this -> data && ! this -> shared_data ) delete -- this -> data;
};

template< typename Real, typename Index >
ostream& operator << ( ostream& str, const tnlLongVector< Real, tnlHost, Index >& vec )
{
	for( Index i = 0; i < vec. getSize(); i ++ )
		str << vec[ i ] << " ";
	return str;
};

template< typename Real, typename Index >
void tnlLongVector< Real, tnlHost, Index > :: setValue( const Real& v )
{
   tnlAssert( this -> size != 0,
              cerr << "Vector name is " << this -> getName() );
   const Index n = this -> getSize();
   Real* data1 = this -> getVector();
   for( Index i = 0; i < n; i ++ )
      data1[ i ] = v;
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlHost, Index > :: save( tnlFile& file ) const
{
   tnlAssert( this -> size != 0,
              cerr << "You try to save empty vector. Its name is " << this -> getName() );
   if( ! tnlObject :: save( file ) )
      return false;
   if( ! file. write( &this -> size, 1 ) )
      return false;
   if( ! file. write( this -> data, this -> size ) )
   {
      cerr << "I was not able to WRITE the long vector " << this -> getName()
           << " with size " << this -> getSize() << endl;
      return false;
   }
   //cerr << "Writing " << this -> size << " elements from " << this -> getName() << "." << endl;
   return true;
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlHost, Index > :: load( tnlFile& file )
{
   if( ! tnlObject :: load( file ) )
      return false;
   int _size;
   if( ! file. read( &_size, 1 ) )
      return false;
   if( _size <= 0 )
   {
      cerr << "Error: The size " << _size << " of the file is not a positive number." << endl;
      return false;
   }
   setSize( _size );
   if( ! file. read( this -> data, this -> size ) )
   {
      cerr << "I was not able to READ the long vector " << this -> getName()
           << " with size " << this -> getSize() << endl;
      return false;
   }
   return true;
};

template< typename Real, typename Index >
tnlLongVector< Real, tnlHost, Index >& tnlLongVector< Real, tnlHost, Index > :: operator =  ( const tnlLongVector< Real, tnlCuda, Index >& cuda_vector )
{
#ifdef HAVE_CUDA
   tnlAssert( cuda_vector. getSize() == this -> getSize(),
              cerr << "You try to copy one vector to another with different size." << endl
                   << "The CUDA source vector " << cuda_vector. getName() << " size is: " << cuda_vector. getSize() << endl                 << " this -> getSize() = " << this -> getSize()
                   << "The target vector " << this -> getName() << " size is " << this -> getSize() << endl; );
   if( cudaMemcpy( this -> data,
                   cuda_vector. getVector(),
                   this -> getSize() * sizeof( Real ),
                   cudaMemcpyDeviceToHost ) != cudaSuccess )
   {
      cerr << "Transfer of data from CUDA device ( " << cuda_vector. getName()
           << " ) to CUDA host ( " << this -> getName() << " ) failed." << endl;
      return *this;
   }
   if( cuda_vector. getSafeMode() )
      cudaThreadSynchronize();
   return *this;
#else
   cerr << "CUDA support is missing in this system." << endl;
   return *this;
#endif
};

template< typename Real, typename Index >
Real tnlMax( const tnlLongVector< Real, tnlHost, Index >& v )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   Real result = v. getVector()[ 0 ];
   const Index n = v. getSize();
   const Real* data1 = v. getVector();
   for( Index i = 1; i < n; i ++ )
      result = Max( result, data1[ i ] );
   return result;
};

template< typename Real, typename Index >
Real tnlMin( const tnlLongVector< Real, tnlHost, Index >& v )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   Real result = v. getVector()[ 0 ];
   const Index n = v. getSize();
   const Real* data1 = v. getVector();
   for( Index i = 1; i < n; i ++ )
      result = Min( result, data1[ i ] );
   return result;
};

template< typename Real, typename Index >
Real tnlAbsMax( const tnlLongVector< Real, tnlHost, Index >& v )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   Real result = v. getVector()[ 0 ];
   const Index n = v. getSize();
   const Real* data1 = v. getVector();
   for( Index i = 1; i < n; i ++ )
      result = Max( result, ( Real ) fabs( data1[ i ] ) );
   return result;
};

template< typename Real, typename Index >
Real tnlAbsMin( const tnlLongVector< Real, tnlHost, Index >& v )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   Real result = v. getVector()[ 0 ];
   const Index n = v. getSize();
   const Real* data1 = v. getVector();
   for( Index i = 1; i < n; i ++ )
      result = Min( result, ( Real ) fabs( data1[ i ] ) );
   return result;
};

template< typename Real, typename Index >
Real tnlLpNorm( const tnlLongVector< Real, tnlHost, Index >& v, const Real& p )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   const Index n = v. getSize();
   const Real* data1 = v. getVector();
   Real result = pow( ( Real ) fabs( data1[ 0 ] ), ( Real ) p );
   for( Index i = 1; i < n; i ++ )
      result += pow( ( Real ) fabs( data1[ i ] ), ( Real ) p  );
   return pow( result, 1.0 / p );
};

template< typename Real, typename Index >
Real tnlSum( const tnlLongVector< Real, tnlHost, Index >& v )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   Real result = v. getVector()[ 0 ];
   const Index n = v. getSize();
   const Real* data1 = v. getVector();
   for( Index i = 1; i < n; i ++ )
      result += data1[ i ];
   return result;
};

template< typename Real, typename Index >
Real tnlDifferenceMax( const tnlLongVector< Real, tnlHost, Index >& u,
                       const tnlLongVector< Real, tnlHost, Index >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   Real result = u[ 0 ] - v[ 0 ];
   const Index n = v. getSize();
   for( Index i = 1; i < n; i ++ )
      result = Max( result, u[ i ] - v[ i ] );
   return result;
}

template< typename Real, typename Index >
Real tnlDifferenceMin( const tnlLongVector< Real, tnlHost, Index >& u,
                       const tnlLongVector< Real, tnlHost, Index >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   Real result = u[ 0 ] - v[ 0 ];
   const Index n = v. getSize();
   for( Index i = 1; i < n; i ++ )
      result = Min( result, u[ i ] - v[ i ] );
   return result;
}

template< typename Real, typename Index >
Real tnlDifferenceAbsMax( const tnlLongVector< Real, tnlHost, Index >& u,
                          const tnlLongVector< Real, tnlHost, Index >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   Real result = u[ 0 ] - v[ 0 ];
   const Index n = v. getSize();
   for( Index i = 1; i < n; i ++ )
      result = Max( result, ( Real ) fabs( u[ i ] - v[ i ] ) );
   return result;

}

template< typename Real, typename Index >
Real tnlDifferenceAbsMin( const tnlLongVector< Real, tnlHost, Index >& u,
                          const tnlLongVector< Real, tnlHost, Index >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   Real result = u[ 0 ] - v[ 0 ];
   const Index n = v. getSize();
   for( Index i = 1; i < n; i ++ )
      result = Min( result, ( Real ) fabs(  u[ i ] - v[ i ] ) );
   return result;
}

template< typename Real, typename Index >
Real tnlDifferenceLpNorm( const tnlLongVector< Real, tnlHost, Index >& u,
                          const tnlLongVector< Real, tnlHost, Index >& v, const Real& p )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   const Index n = v. getSize();
   Real result = pow( ( Real ) fabs( u[ 0 ] - v[ 0 ] ), ( Real ) p );
   for( Index i = 1; i < n; i ++ )
      result += pow( ( Real ) fabs( u[ i ] - v[ i ] ), ( Real ) p  );
   return pow( result, 1.0 / p );
}

template< typename Real, typename Index >
Real tnlDifferenceSum( const tnlLongVector< Real, tnlHost, Index >& u,
                       const tnlLongVector< Real, tnlHost, Index >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   Real result = u[ 0 ] - v[ 0 ];
   const Index n = u. getSize();
   for( Index i = 1; i < n; i ++ )
      result += u[ i ] - v[ i ];
   return result;
};

template< typename Real, typename Index >
void tnlScalarMultiplication( const Real& alpha,
                              tnlLongVector< Real, tnlHost, Index >& u )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   const Index n = u. getSize();
   for( Index i = 0; i < n; i ++ )
      u[ i ] *= alpha;
}

template< typename Real, typename Index >
Real tnlSDOT( const tnlLongVector< Real, tnlHost, Index >& u,
              const tnlLongVector< Real, tnlHost, Index >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
            cerr << "You try to compute SDOT of two vectors with different size." << endl
                 << "The first tnLongVector " << u. getName() << " has size " << u. getSize() << "." << endl
                 << "The second tnlLongVector " << v. getName() << " has size " << v. getSize() << "."  );
   Real result = ( Real ) 0;
   const Index n = u. getSize();
   const Real* data1 = u. getVector();
   const Real* data2 = v. getVector();
   for( Index i = 0; i < n; i ++ )
      result += data1[ i ] * data2[ i ];
   return result;
};

template< typename Real, typename Index >
void tnlSAXPY( const Real& alpha,
               const tnlLongVector< Real, tnlHost, Index >& x,
               tnlLongVector< Real, tnlHost, Index >& y )
{
   tnlAssert( y. getSize() != 0,
              cerr << "Vector name is " << y. getName() );
   tnlAssert( y. getSize() == x. getSize(),
            cerr << "You try to compute SAXPY with vector x having different size." << endl
                 << "The y tnLongVector " << y. getName() << " has size " << y. getSize() << "." << endl
                 << "The x tnlLongVector " << x. getName() << " has size " << x. getSize() << "."  );
   const Index n = y. getSize();
   Real* data1 = y. getVector();
   const Real* data2 = x. getVector();
   for( Index i = 0; i < n; i ++ )
      data1[ i ] += alpha * data2[ i ];
};

template< typename Real, typename Index >
void tnlSAXMY( const Real& alpha,
               const tnlLongVector< Real, tnlHost, Index >& x,
               tnlLongVector< Real, tnlHost, Index >& y )
{
   tnlAssert( y. getSize() != 0,
              cerr << "Vector name is " << y. getName() );
   tnlAssert( y. getSize() == x. getSize(),
            cerr << "You try to compute SAXPY with vector x having different size." << endl
                 << "The y tnLongVector " << y. getName() << " has size " << y. getSize() << "." << endl
                 << "The x tnlLongVector " << x. getName() << " has size " << x. getSize() << "."  );
   const Index n = y. getSize();
   Real* data1 = y. getVector();
   const Real* data2 = x. getVector();
   for( Index i = 0; i < n; i ++ )
      data1[ i ] = alpha * data2[ i ] - data1[ i ];
};



#endif /* TNLLONGVECTORHOST_H_ */
