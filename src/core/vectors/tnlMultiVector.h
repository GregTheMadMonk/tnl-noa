/***************************************************************************
                          tnlMultiVector.h  -  description
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

#ifndef TNLMULTIVECTOR_H_
#define TNLMULTIVECTOR_H_

#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlStaticVector.h>
#include <core/tnlAssert.h>


template< int Dimensions, typename Real = double, typename Device = tnlHost, typename Index = int >
class tnlMultiVector : public tnlVector< Real, Device, Index >
{
};

template< typename Real, typename Device, typename Index >
class tnlMultiVector< 1, Real, Device, Index > : public tnlVector< Real, Device, Index >
{
   public:
   enum { Dimensions = 1};
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlMultiVector< Dimensions, Real, tnlHost, Index > HostType;
   typedef tnlMultiVector< Dimensions, Real, tnlCuda, Index > CudaType;

   tnlMultiVector();

   tnlMultiVector( const tnlString& name );

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   bool setDimensions( const Index iSize );

   bool setDimensions( const tnlStaticVector< Dimensions, Index >& dimensions );

   void getDimensions( Index& iSize ) const;

   const tnlStaticVector< 1, Index >& getDimensions() const;

   //! Set dimensions of the Vector using another Vector as a template
   template< typename MultiVector >
   bool setLike( const tnlMultiVector& v );
   
   Index getElementIndex( const Index i ) const;

   void setElement( const Index i, Real value );

   //! This method can be used for general access to the elements of the Vectors.
   /*! It does not return reference but value. So it can be used to access
    *  Vectors in different adress space (usualy GPU device).
    *  See also operator().
    */
   Real getElement( const Index i ) const;

   //! Operator for accessing elements of the Vector.
   /*! It returns reference to given elements so it cannot be
    *  used to access elements of Vectors in different adress space
    *  (GPU device usualy).
    */
   Real& operator()( const Index i );

   const Real& operator()( const Index i ) const;

   template< typename MultiVector >
   bool operator == ( const MultiVector& Vector ) const;

   template< typename MultiVector >
   bool operator != ( const MultiVector& Vector ) const;

   tnlMultiVector< 1, Real, Device, Index >& operator = ( const tnlMultiVector< 1, Real, Device, Index >& Vector );

   template< typename MultiVector >
   tnlMultiVector< 1, Real, Device, Index >& operator = ( const MultiVector& Vector );

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   protected:

   tnlStaticVector< Dimensions, Index > dimensions;
};

template< typename Real, typename Device, typename Index >
class tnlMultiVector< 2, Real, Device, Index > : public tnlVector< Real, Device, Index >
{
   public:
   enum { Dimensions = 2 };
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlMultiVector< Dimensions, Real, tnlHost, Index > HostType;
   typedef tnlMultiVector< Dimensions, Real, tnlCuda, Index > CudaType;

   tnlMultiVector();

   tnlMultiVector( const tnlString& name );

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   bool setDimensions( const Index jSize, const Index iSize );

   bool setDimensions( const tnlStaticVector< 2, Index >& dimensions );

   void getDimensions( Index& jSize, Index& iSize ) const;

   const tnlStaticVector< 2, Index >& getDimensions() const;

   //! Set dimensions of the Vector using another Vector as a template
   template< typename MultiVector >
   bool setLike( const MultiVector& v );

   Index getElementIndex( const Index j, const Index i ) const;

   void setElement( const Index j, const Index i, Real value );

   //! This method can be used for general access to the elements of the Vectors.
   /*! It does not return reference but value. So it can be used to access
    *  Vectors in different adress space (usualy GPU device).
    *  See also operator().
    */
   Real getElement( const Index j, const Index i ) const;

   //! Operator for accessing elements of the Vector.
   /*! It returns reference to given elements so it cannot be
    *  used to access elements of Vectors in different adress space
    *  (GPU device usualy).
    */
   Real& operator()( const Index j, const Index i );

   const Real& operator()( const Index j, const Index i ) const;

   template< typename MultiVector >
   bool operator == ( const MultiVector& Vector ) const;

   template< typename MultiVector >
   bool operator != ( const MultiVector& Vector ) const;

   tnlMultiVector< 2, Real, Device, Index >& operator = ( const tnlMultiVector< 2, Real, Device, Index >& Vector );

   template< typename MultiVector >
   tnlMultiVector< 2, Real, Device, Index >& operator = ( const MultiVector& Vector );

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   protected:

   tnlStaticVector< 2, Index > dimensions;
};

template< typename Real, typename Device, typename Index >
class tnlMultiVector< 3, Real, Device, Index > : public tnlVector< Real, Device, Index >
{
   public:

   enum { Dimensions = 3 };
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlMultiVector< Dimensions, Real, tnlHost, Index > HostType;
   typedef tnlMultiVector< Dimensions, Real, tnlCuda, Index > CudaType;

   tnlMultiVector();

   tnlMultiVector( const tnlString& name );

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   bool setDimensions( const Index k, const Index j, const Index iSize );

   bool setDimensions( const tnlStaticVector< 3, Index >& dimensions );

   void getDimensions( Index& k, Index& j, Index& iSize ) const;

   const tnlStaticVector< 3, Index >& getDimensions() const;

   //! Set dimensions of the Vector using another Vector as a template
   template< typename MultiVector >
   bool setLike( const MultiVector& v );

   Index getElementIndex( const Index k, const Index j, const Index i ) const;

   void setElement( const Index k, const Index j, const Index i, Real value );

   //! This method can be used for general access to the elements of the Vectors.
   /*! It does not return reference but value. So it can be used to access
    *  Vectors in different adress space (usualy GPU device).
    *  See also operator().
    */
   Real getElement( const Index k, const Index j, const Index i ) const;

   //! Operator for accessing elements of the Vector.
   /*! It returns reference to given elements so it cannot be
    *  used to access elements of Vectors in different adress space
    *  (GPU device usualy).
    */
   Real& operator()( const Index k, const Index j, const Index i );

   const Real& operator()( const Index k, const Index j, const Index i ) const;

   template< typename MultiVector >
   bool operator == ( const MultiVector& Vector ) const;

   template< typename MultiVector >
   bool operator != ( const MultiVector& Vector ) const;

   tnlMultiVector< 3, Real, Device, Index >& operator = ( const tnlMultiVector< 3, Real, Device, Index >& Vector );

   template< typename MultiVector >
   tnlMultiVector< 3, Real, Device, Index >& operator = ( const MultiVector& Vector );

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   protected:

   tnlStaticVector< 3, Index > dimensions;
};

template< typename Real, typename Device, typename Index >
class tnlMultiVector< 4, Real, Device, Index > : public tnlVector< Real, Device, Index >
{
   public:

   enum { Dimensions = 4 };
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlMultiVector< Dimensions, Real, tnlHost, Index > HostType;
   typedef tnlMultiVector< Dimensions, Real, tnlCuda, Index > CudaType;

   tnlMultiVector();

   tnlMultiVector( const tnlString& name );

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   bool setDimensions( const Index l, const Index k, const Index j, const Index iSize );

   bool setDimensions( const tnlStaticVector< 4, Index >& dimensions );

   void getDimensions( Index& l, Index& k, Index& j, Index& iSize ) const;

   const tnlStaticVector< 4, Index >& getDimensions() const;

   //! Set dimensions of the Vector using another Vector as a template
   template< typename MultiVector >
   bool setLike( const tnlMultiVector& v );

   Index getElementIndex( const Index l, const Index k, const Index j, const Index i ) const;

   void setElement( const Index l, const Index k, const Index j, const Index i, Real value );

   //! This method can be used for general access to the elements of the Vectors.
   /*! It does not return reference but value. So it can be used to access
    *  Vectors in different adress space (usualy GPU device).
    *  See also operator().
    */
   Real getElement( const Index l, const Index k, const Index j, const Index i ) const;

   //! Operator for accessing elements of the Vector.
   /*! It returns reference to given elements so it cannot be
    *  used to access elements of Vectors in different adress space
    *  (GPU device usualy).
    */
   Real& operator()( const Index l, const Index k, const Index j, const Index i );

   const Real& operator()( const Index l, const Index k, const Index j, const Index i ) const;

   template< typename MultiVector >
   bool operator == ( const MultiVector& Vector ) const;

   template< typename MultiVector >
   bool operator != ( const MultiVector& Vector ) const;

   tnlMultiVector< 4, Real, Device, Index >& operator = ( const tnlMultiVector< 4, Real, Device, Index >& Vector );

   template< typename MultiVector >
   tnlMultiVector< 4, Real, Device, Index >& operator = ( const MultiVector& Vector );

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   protected:

   tnlStaticVector< 4, Index > dimensions;
};

template< typename Real, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiVector< 1, Real, device, Index >& Vector );

template< typename Real, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiVector< 2, Real, device, Index >& Vector );

template< typename Real, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiVector< 3, Real, device, Index >& Vector );

template< typename Real, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiVector< 4, Real, device, Index >& Vector );


#include <core/vectors/tnlMultiVector1D_impl.h>
#include <core/vectors/tnlMultiVector2D_impl.h>
#include <core/vectors/tnlMultiVector3D_impl.h>
#include <core/vectors/tnlMultiVector4D_impl.h>

#endif /* TNLMULTIVECTOR_H_ */
