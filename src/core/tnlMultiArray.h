/***************************************************************************
                          tnlMultiArray.h  -  description
                             -------------------
    begin                : Nov 25, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef TNLMULTIARRAY_H_
#define TNLMULTIARRAY_H_

#include <core/tnlVector.h>
#include <core/tnlTuple.h>
#include <core/tnlAssert.h>


template< int Dimensions, typename Real = double, typename device = tnlHost, typename Index = int >
class tnlMultiArray : public tnlVector< Real, device, Index >
{
   //! We do not allow constructor without parameters.
   tnlMultiArray();

   //! We do not allow copy constructor without object name.
   tnlMultiArray( const tnlMultiArray< Dimensions, Real, device, Index >& a );

   public:

   tnlMultiArray( const tnlString& name );

   tnlMultiArray( const tnlString& name,
             const tnlMultiArray< Dimensions, Real, tnlHost, Index >& array );

   tnlMultiArray( const tnlString& name,
             const tnlMultiArray< Dimensions, Real, tnlCuda, Index >& array );

   void setSharedData( Real* data,
                       tnlTuple< Dimensions, Index > dimensions );

   bool setDimensions( const tnlTuple< Dimensions, Index >& dimensions );

   //! Set dimensions of the array using another array as a template
   bool setLike( const tnlMultiArray< Dimensions, Real, tnlHost, Index >& v );

   //! Set dimensions of the array using another array as a template
   bool setLike( const tnlMultiArray< Dimensions, Real, tnlCuda, Index >& v );

   void reset();

   const tnlTuple< Dimensions, Index >& getDimensions() const;

   tnlString getType() const;

   //Index getLongVectorIndex( const tnlTuple< Dimensions, Index >& element ) const;
   
   Index getLongVectorIndex( const tnlTuple< 1, Index >& element ) const;

   Index getLongVectorIndex( const tnlTuple< 2, Index >& element ) const;

   Index getLongVectorIndex( const tnlTuple< 3, Index >& element ) const;

   void setElement( const tnlTuple< Dimensions, Index >& element, Real value );

   //! This method can be used for general access to the elements of the arrays.
   /*! It does not return reference but value. So it can be used to access
    *  arrays in different adress space (usualy GPU device).
    *  See also operator().
    */
   Real getElement( const tnlTuple< Dimensions, Index >& element ) const;

   //! This method is used by simpler versions of the getElement and setElement.
   /*! This method might be removed in the future.
    */
   Index getLongVectorIndex( const Index i1 ) const;

   //! This method is used by simpler versions of the getElement and setElement.
   /*! This method might be removed in the future.
    */
   Index getLongVectorIndex( const Index i1, const Index i2 ) const;

   //! This method is used by simpler versions of the getElement and setElement.
   /*! This method might be removed in the future.
    */
   Index getLongVectorIndex( const Index i1, const Index i2, Index i3 ) const;

   //! Easier array acces for 1D array. It may be removed in the future.
   void setElement( Index i1, const Real& value );

   //! Easier array acces for 1D array. It may be removed in the future.
   Real getElement( Index i1 ) const;

   //! Easier array acces for 2D array. It may be removed in the future.
   void setElement( Index i1, Index i2, const Real& value );

   //! Easier array acces for 2D array. It may be removed in the future.
   Real getElement( Index i1, Index i2 ) const;

   //! Easier array acces for 3D array. It may be removed in the future.
   void setElement( Index i1, Index i2, Index i3, const Real& value );

   //! Easier array acces for 3D array. It may be removed in the future.
   Real getElement( Index i1, Index i2, Index i3 ) const;

   //! Operator for accessing elements of the array.
   /*! It returns reference to given elements so it cannot be
    *  used to access elements of arrays in different adress space
    *  (GPU device usualy).
    */
   Real& operator()( const tnlTuple< Dimensions, Index >& element );

   const Real& operator()( const tnlTuple< Dimensions, Index >& element ) const;

   Real& operator() ( Index i1 );

   const Real& operator() ( Index i1 ) const;

   Real& operator() ( Index i1, Index i2 );

   const Real& operator() ( Index i1, Index i2 ) const;

   Real& operator() ( Index i1, Index i2, Index i3 );

   const Real& operator() ( Index i1, Index i2, Index i3 ) const;

   bool operator == ( const tnlMultiArray< Dimensions, Real, device, Index >& array ) const;

   bool operator != ( const tnlMultiArray< Dimensions, Real, device, Index >& array ) const;

   template< typename Real2, typename device2, typename Index2 >
   tnlMultiArray< Dimensions, Real, device, Index >& operator = ( const tnlMultiArray< Dimensions, Real2, device2, Index2 >& array );

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );


   protected:
   tnlTuple< Dimensions, Index > arrayDimensions;
};

template< int Dimensions, typename Real, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiArray< Dimensions, Real, device, Index >& array );

template< int Dimensions, typename Real, typename device, typename Index >
tnlMultiArray< Dimensions, Real, device, Index > :: tnlMultiArray( const tnlString& name )
: tnlVector< Real, device, Index >( name )
  {
  }

template< int Dimensions, typename Real, typename device, typename Index >
tnlMultiArray< Dimensions, Real, device, Index > :: tnlMultiArray( const tnlString& name,
                                                         const tnlMultiArray< Dimensions, Real, tnlHost, Index >& array )
: tnlVector< Real, device, Index >( name, array )
{
   this -> arrayDimensions = array. getDimensions();
}

template< int Dimensions, typename Real, typename device, typename Index >
tnlMultiArray< Dimensions, Real, device, Index > :: tnlMultiArray( const tnlString& name,
                                                         const tnlMultiArray< Dimensions, Real, tnlCuda, Index >& array )
: tnlVector< Real, device, Index >( name, array )
{
   this -> arrayDimensions = array. getDimensions();
}

template< int Dimensions, typename Real, typename device, typename Index >
void tnlMultiArray< Dimensions, Real, device, Index > :: setSharedData( Real* data,
                                                                   tnlTuple< Dimensions, Index > dimensions )
{
   Index size( 1 );
   for( int i = 0; i < Dimensions; i ++ )
      size *= arrayDimensions[ i ];
   tnlVector< Real, device, Index > :: setSharedData( data, size );
   arrayDimensions = dimensions;
}

template< int Dimensions, typename Real, typename device, typename Index >
bool tnlMultiArray< Dimensions, Real, device, Index > :: setDimensions( const tnlTuple< Dimensions, Index >& dimensions )
{
   arrayDimensions = dimensions;
   Index size( 1 );
   for( int i = 0; i < Dimensions; i ++ )
   {
      if( arrayDimensions[ i ] <= 1 )
      {
         cerr << "There is an attempt to set wrong dimensions ( " << arrayDimensions
              << " ) for array " << this -> getName() << ". Each dimensions must be greater then 1." << endl;
         return false;
      }
      size *= arrayDimensions[ i ];
   }
   /****
    * Do not remove this with this -> setSize().
    * This makes problems in case multiple inheritance - like in tnlFullMatrix.
    */
   return tnlVector< Real, device, Index > :: setSize( size );
}

template< int Dimensions, typename Real, typename device, typename Index >
bool tnlMultiArray< Dimensions, Real, device, Index > :: setLike( const tnlMultiArray< Dimensions, Real, tnlHost, Index >& v )
{
   return setDimensions( v. getDimensions() );
}

template< int Dimensions, typename Real, typename device, typename Index >
bool tnlMultiArray< Dimensions, Real, device, Index > :: setLike( const tnlMultiArray< Dimensions, Real, tnlCuda, Index >& v )
{
   return setDimensions( v. getDimensions() );
}

template< int Dimensions, typename Real, typename device, typename Index >
void tnlMultiArray< Dimensions, Real, device, Index > :: reset()
{
   tnlVector< Real, device, Index > :: reset();
   setDimensions( tnlTuple< Dimensions, Index >( 0 ) );
}


template< int Dimensions, typename Real, typename device, typename Index >
const tnlTuple< Dimensions, Index >& tnlMultiArray< Dimensions, Real, device, Index > :: getDimensions() const
{
   return arrayDimensions;
}

template< int Dimensions, typename Real, typename Device, typename Index >
tnlString tnlMultiArray< Dimensions, Real, Device, Index > :: getType() const
{
   return tnlString( "tnlMultiArray< ") +
          tnlString( Dimensions ) +
          tnlString( ", " ) +
          tnlString( getParameterType< Real >() ) +
          tnlString( ", " ) +
          tnlString( Device :: getDeviceType() ) +
          tnlString( ", " ) +
          tnlString( getParameterType< Index >() ) +
          tnlString( " >" );
}

/*template< int Dimensions, typename Real, typename device, typename Index >
Index tnlMultiArray< Dimensions, Real, device, Index > :: getLongVectorIndex( const tnlTuple< Dimensions, Index >& element ) const
{
}*/

template< int Dimensions, typename Real, typename device, typename Index >
Index tnlMultiArray< Dimensions, Real, device, Index > :: getLongVectorIndex( const tnlTuple< 1, Index >& element ) const
{
   tnlAssert( Dimensions == 1, );
   tnlAssert( element >= ( tnlTuple< 1, Index >( 0 ) ) && element. x() < this -> getDimensions(). x(),
              cerr << " element = ( " << element << " ) the dimensions is ( " << getDimensions()
                   << " ) array name is " << this -> getName() << endl; );

   return element. x();
}

template< int Dimensions, typename Real, typename device, typename Index >
Index tnlMultiArray< Dimensions, Real, device, Index > :: getLongVectorIndex( const tnlTuple< 2, Index >& element ) const
{
   tnlAssert( Dimensions == 2, );
   tnlAssert( element >= ( tnlTuple< 2, Index >( 0 ) ) &&
              element. x() < this -> getDimensions(). x() &&
              element. y() < this -> getDimensions(). y(),
              cerr << " element = ( " << element << " ) the dimensions is ( " << getDimensions()
                   << " ) array name is " << this -> getName() << endl; );

   return element. x() * this -> arrayDimensions. y() +
          element. y();
}

template< int Dimensions, typename Real, typename device, typename Index > 
Index tnlMultiArray< Dimensions, Real, device, Index > :: getLongVectorIndex( const tnlTuple< 3, Index >& element ) const
{
   tnlAssert( Dimensions == 3, );
   tnlAssert( element >= ( tnlTuple< 3, Index >( 0 ) ) &&
              element. x() < this -> getDimensions(). x() &&
              element. y() < this -> getDimensions(). y() &&
              element. z() < this -> getDimensions(). z(),
              cerr << " element = ( " << element << " ) the dimensions is ( " << getDimensions()
                   << " ) array name is " << this -> getName() << endl; );

   return element. x() * this -> arrayDimensions. y() * arrayDimensions. z() +
          element. y() * this -> arrayDimensions. z() +
          element. z();
}


template< int Dimensions, typename Real, typename device, typename Index >
void tnlMultiArray< Dimensions, Real, device, Index > :: setElement( const tnlTuple< Dimensions, Index >& element, Real value )
{
   tnlVector< Real, device, Index > :: setElement( getLongVectorIndex( element ), value );
}

template< int Dimensions, typename Real, typename device, typename Index >
Real tnlMultiArray< Dimensions, Real, device, Index > :: getElement( const tnlTuple< Dimensions, Index >& element ) const
{
   return tnlVector< Real, device, Index > :: getElement( getLongVectorIndex( element ) );
}

template< int Dimensions, typename Real, typename device, typename Index >
void tnlMultiArray< Dimensions, Real, device, Index > :: setElement( Index i1, const Real& value )
{
   tnlAssert( Dimensions == 1, );
   tnlAssert( i1 >=0 && i1 < this -> getDimensions(). x(),
              cerr << " i1 = " << i1 << " the dimensions is " << getDimensions(). x()
                   << " array name is " << this -> getName() << endl; );

   tnlVector< Real, device, Index > :: setElement( i1, value );
}

template< int Dimensions, typename Real, typename device, typename Index >
Real tnlMultiArray< Dimensions, Real, device, Index > :: getElement( Index i1 ) const
{
   tnlAssert( Dimensions == 1, );
   tnlAssert( i1 >=0 && i1 < this -> getDimensions(). x(),
              cerr << " i1 = " << i1 << " the dimensions is " << getDimensions(). x()
                   << " array name is " << this -> getName() << endl; );

   return tnlVector< Real, device, Index > :: getElement( i1 );
}

template< int Dimensions, typename Real, typename device, typename Index >
void tnlMultiArray< Dimensions, Real, device, Index > :: setElement( Index i1, Index i2, const Real& value )
{
   tnlAssert( Dimensions == 2, );
   tnlAssert( i1 >=0 && i1 < this -> getDimensions(). x(),
              cerr << " i1 = " << i1 << " the dimensions is " << getDimensions(). x()
                   << " array name is " << this -> getName() << endl; );
   tnlAssert( i2 >= 0 && i2 < this -> getDimensions(). y(),
              cerr << " i2 = " << i2 << " the dimensions is " << getDimensions(). y()
                   << " array name is " << this -> getName() << endl; );

   tnlVector< Real, device, Index > :: setElement( i1 * this -> arrayDimensions. y() + i2, value );
}

template< int Dimensions, typename Real, typename device, typename Index >
Real tnlMultiArray< Dimensions, Real, device, Index > :: getElement( Index i1, Index i2 ) const
{
   tnlAssert( Dimensions == 2, );
   tnlAssert( i1 >=0 && i1 < this -> getDimensions(). x(),
              cerr << " i1 = " << i1 << " the dimensions is " << getDimensions(). x()
                   << " array name is " << this -> getName() << endl; );
   tnlAssert( i2 >= 0 && i2 < this -> getDimensions(). y(),
              cerr << " i2 = " << i2 << " the dimensions is " << getDimensions(). y()
                   << " array name is " << this -> getName() << endl; );

   return tnlVector< Real, device, Index > :: getElement( i1 * this -> arrayDimensions. y() + i2 );
}

template< int Dimensions, typename Real, typename device, typename Index >
void tnlMultiArray< Dimensions, Real, device, Index > :: setElement( Index i1, Index i2, Index i3, const Real& value )
{
   tnlAssert( Dimensions == 3, );
   tnlAssert( i1 >=0 && i1 < this -> getDimensions(). x(),
              cerr << " i1 = " << i1 << " the dimensions is " << getDimensions(). x()
                   << " array name is " << this -> getName() << endl; );
   tnlAssert( i2 >= 0 && i2 < this -> getDimensions(). y(),
              cerr << " i2 = " << i2 << " the dimensions is " << getDimensions(). y()
                   << " array name is " << this -> getName() << endl; );
   tnlAssert( i3 >= 0 && i3 < this -> getDimensions(). z(),
              cerr << " i3 = " << i3 << " the dimensions is " << getDimensions(). z()
                   << " array name is " << this -> getName() << endl; );

   tnlVector< Real, device, Index > :: setElement( i1 * this -> arrayDimensions. y() * arrayDimensions. z() +
                                                       i2 * this -> arrayDimensions. z() + i3,
                                                       value );
}

template< int Dimensions, typename Real, typename device, typename Index >
Real tnlMultiArray< Dimensions, Real, device, Index > :: getElement( Index i1, Index i2, Index i3 ) const
{
   tnlAssert( Dimensions == 3, );
   tnlAssert( i1 >=0 && i1 < this -> getDimensions(). x(),
              cerr << " i1 = " << i1 << " the dimensions is " << getDimensions(). x()
                   << " array name is " << this -> getName() << endl; );
   tnlAssert( i2 >= 0 && i2 < this -> getDimensions(). y(),
              cerr << " i2 = " << i2 << " the dimensions is " << getDimensions(). y()
                   << " array name is " << this -> getName() << endl; );
   tnlAssert( i3 >= 0 && i3 < this -> getDimensions(). z(),
              cerr << " i3 = " << i3 << " the dimensions is " << getDimensions(). z()
                   << " array name is " << this -> getName() << endl; );

   return tnlVector< Real, device, Index > :: getElement( i1 * this -> arrayDimensions. y() * arrayDimensions. z() +
                                                              i2 * this -> arrayDimensions. z() + i3 );
}

template< int Dimensions, typename Real, typename device, typename Index >
Real& tnlMultiArray< Dimensions, Real, device, Index > :: operator()( const tnlTuple< Dimensions, Index >& element )
{
   return tnlVector< Real, device, Index > :: operator[]( getLongVectorIndex( element ) );
}

template< int Dimensions, typename Real, typename device, typename Index >
const Real& tnlMultiArray< Dimensions, Real, device, Index > :: operator()( const tnlTuple< Dimensions, Index >& element ) const
{
   return tnlVector< Real, device, Index > :: operator[]( getLongVectorIndex( element ) );
}

template< int Dimensions, typename Real, typename device, typename Index >
Real& tnlMultiArray< Dimensions, Real, device, Index > :: operator() ( Index i1 )
{
   return tnlVector< Real, device, Index > :: operator[]( getLongVectorIndex( tnlTuple< 1, Index >( i1 ) ) );
}

template< int Dimensions, typename Real, typename device, typename Index >
const Real& tnlMultiArray< Dimensions, Real, device, Index > :: operator() ( Index i1 ) const
{
   return tnlVector< Real, device, Index > :: operator[]( getLongVectorIndex( tnlTuple< 1, Index >( i1 ) ) );
}

template< int Dimensions, typename Real, typename device, typename Index >
Real& tnlMultiArray< Dimensions, Real, device, Index > :: operator() ( Index i1, Index i2 )
{
   return tnlVector< Real, device, Index > :: operator[]( getLongVectorIndex( tnlTuple< 2, Index >( i1, i2 ) ) );
}

template< int Dimensions, typename Real, typename device, typename Index >
const Real& tnlMultiArray< Dimensions, Real, device, Index > :: operator() ( Index i1, Index i2 ) const
{
   return tnlVector< Real, device, Index > :: operator[]( getLongVectorIndex( tnlTuple< 2, Index >( i1, i2 ) ) );
}

template< int Dimensions, typename Real, typename device, typename Index >
Real& tnlMultiArray< Dimensions, Real, device, Index > :: operator() ( Index i1, Index i2, Index i3 )
{
   return tnlVector< Real, device, Index > :: operator[]( getLongVectorIndex( tnlTuple< 3, Index >( i1, i2, i3 ) ) );
}

template< int Dimensions, typename Real, typename device, typename Index >
const Real& tnlMultiArray< Dimensions, Real, device, Index > :: operator() ( Index i1, Index i2, Index i3 ) const
{
   return tnlVector< Real, device, Index > :: operator[]( getLongVectorIndex( tnlTuple< 3, Index >( i1, i2, i3 ) ) );
}

template< int Dimensions, typename Real, typename device, typename Index >
bool tnlMultiArray< Dimensions, Real, device, Index > :: operator == ( const tnlMultiArray< Dimensions, Real, device, Index >& array ) const
{
   tnlAssert( this -> getDimensions() == array. getDimensions(),
              cerr << "You are attempting to compare two arrays with different dimensions." << endl
                   << "First array name is " << this -> getName()
                   << " dimensions are ( " << this -> getDimensions() << " )" << endl
                   << "Second array is " << array. getName()
                   << " dimensions are ( " << array. getDimensions() << " )" << endl; );
   return tnlVector< Real, device, Index > :: operator == ( array );
}

template< int Dimensions, typename Real, typename device, typename Index >
bool tnlMultiArray< Dimensions, Real, device, Index > :: operator != ( const tnlMultiArray< Dimensions, Real, device, Index >& array ) const
{
   return ! ( (* this ) == array );
}

template< int Dimensions, typename Real, typename device, typename Index >
  template< typename Real2, typename device2, typename Index2 >
tnlMultiArray< Dimensions, Real, device, Index >&
tnlMultiArray< Dimensions, Real, device, Index >
:: operator = ( const tnlMultiArray< Dimensions, Real2, device2, Index2 >& array )
{
   tnlAssert( this -> getDimensions() == array. getDimensions(),
              cerr << "You are attempting to assign two arrays with different dimensions." << endl
                   << "First array name is " << this -> getName()
                   << " dimensions are ( " << this -> getDimensions() << " )" << endl
                   << "Second array is " << array. getName()
                   << " dimensions are ( " << array. getDimensions() << " )" << endl; );
   tnlVector< Real, device, Index > :: operator = ( array );
   return ( *this );
}

template< int Dimensions, typename Real, typename device, typename Index >
bool tnlMultiArray< Dimensions, Real, device, Index > :: save( tnlFile& file ) const
{
   if( ! tnlVector< Real, device, Index > :: save( file ) )
   {
      cerr << "I was not able to write the tnlVector of the tnlMultiArray "
           << this -> getName() << endl;
      return false;
   }
   if( ! arrayDimensions. save( file ) )
   {
      cerr << "I was not able to write the dimensions of the tnlMultiArray "
           << this -> getName() << endl;
      return false;
   }
   return true;
}

template< int Dimensions, typename Real, typename device, typename Index >
bool tnlMultiArray< Dimensions, Real, device, Index > :: load( tnlFile& file )
{
   if( ! tnlVector< Real, device, Index > :: load( file ) )
   {
      cerr << "I was not able to read the tnlVector of the tnlMultiArray "
           << this -> getName() << endl;
      return false;
   }
   if( ! arrayDimensions. load( file ) )
   {
      cerr << "I was not able to read the dimensions of the tnlMultiArray "
           << this -> getName() << endl;
      return false;
   }
   return true;
}

template< typename Real, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiArray< 1, Real, device, Index >& array )
{
   tnlTuple< 1, Index > dims = array. getDimensions();
   for( Index i = 0; i < dims[ tnlX ]; i ++ )
   {
      tnlTuple< 1, Index > ind;
      ind[ tnlX ] = i;
      str << array. getElement( ind ) << " ";
   }
   return str;
}

template< typename Real, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiArray< 2, Real, device, Index >& array )
{
   tnlTuple< 2, Index > dims = array. getDimensions();
   for( Index i = 0; i < dims[ tnlX ]; i ++ )
   {
      for( Index j = 0; j < dims[ tnlY ]; j ++ )
      {
         tnlTuple< 2, Index > ind;
         ind[ 0 ] = i;
         ind[ 1 ] = j;
         str << array. getElement( ind ) << " ";
      }
      str << endl;
   }
   return str;
}

template< typename Real, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiArray< 3, Real, device, Index >& array )
{
   tnlTuple< 3, Index > dims = array. getDimensions();
   for( Index i = 0; i < dims[ tnlX ]; i ++ )
   {
      for( Index j = 0; j < dims[ tnlY ]; j ++ )
      {
         for( Index k = 0; k < dims[ tnlZ ]; k ++ )
         {
            tnlTuple< 3, Index > ind;
            ind[ 0 ] = i;
            ind[ 1 ] = j;
            ind[ 2 ] = k;
            str << array. getElement( ind ) << " ";
         }
         str << endl;
      }
      str << endl;
   }
   return str;
}

template< int Dimensions, typename Real, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiArray< Dimensions, Real, device, Index >& array )
{
   tnlAssert( false,
              cerr << "Operator << is not yet implemented for arrays with the dimensions greater than 3." << endl; );
   return str;
};


#endif /* TNLMULTIARRAY_H_ */
