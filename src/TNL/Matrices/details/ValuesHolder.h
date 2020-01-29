/***************************************************************************
                          ValuesHolder.h  -  description
                             -------------------
    begin                : Jan 27, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
   namespace Matrices {
      namespace details {


template< typename Real,
          typename Device,
          typename Index >
struct ValuesHolderView
: public Containers::VectorView< Real, Device, Index >
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;

   using Containers::VectorView< Real, Device, Index >::VectorView;
   using Containers::VectorView< Real, Device, Index >::operator=;
   /*__cuda_callable__
   ValuesHolderView() = default;

   __cuda_callable__
   explicit ValuesHolderView( const ValuesHolderView& ) = default;

   __cuda_callable__
   ValuesHolderView( ValuesHolderView&& ) = default;*/

};

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
struct ValuesHolder
: public Containers::Vector< Real, Device, Index, Allocator >
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using AllocatorType = Allocator;
   using ViewType = ValuesHolderView< Real, Device, Index >;

   using Containers::Vector< Real, Device, Index, Allocator >::Vector;
   using Containers::Vector< Real, Device, Index, Allocator >::operator=;
   /*ValuesHolder() = default;

   explicit ValuesHolder( const ValuesHolder& ) = default;

   explicit ValuesHolder( const ValuesHolder& vector, const AllocatorType& allocator );

   ValuesHolder( ValuesHolder&& ) = default;*/

};

template< typename Device,
          typename Index >
class BooleanValuesHolder
{
   public:

      using RealType = bool;
      using DeviceType = Device;
      using IndexType = Index;
      using ViewType = BooleanValuesHolder;

      BooleanValuesHolder()
      : size( 0 ){};

      BooleanValuesHolder( const IndexType& size )
      : size( size ){};

      void setSize( const IndexType& size ) { this->size = size; };

      __cuda_callable__
      IndexType getSize() const { return this->size; };

      __cuda_callable__
      bool operator[]( const IndexType& i ) const { return true; };

   protected:

      IndexType size;
};

/**
 * \brief Serialization of values holder into binary files.
 */
template< typename Device, typename Index, typename Allocator >
File& operator<<( File& file, const ValuesHolder< bool, Device, Index, Allocator >& holder ) {
   file << holder.getSize();
   return file; };

template< typename Device, typename Index, typename Allocator >
File& operator<<( File&& file, const ValuesHolder< bool, Device, Index, Allocator >& holder ) {
   file << holder.getSize();
   return file; };

/**
 * \brief Deserialization of values holder from binary files.
 */
template< typename Device, typename Index, typename Allocator >
File& operator>>( File& file, ValuesHolder< bool, Device, Index, Allocator >& holder ) {
   Index size;
   file >> size;
   holder.setSize( size );
   return file; };

template< typename Device, typename Index, typename Allocator >
File& operator>>( File&& file, ValuesHolder< bool, Device, Index, Allocator >& holder ) {
   Index size;
   file >> size;
   holder.setSize( size );
   return file; };

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
struct ValuesHolderSetter
{
   using type = ValuesHolder< Real, Device, Index, RealAllocator >;
};

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
struct SparseMatrixValuesHolderSetter
{
   using type = ValuesHolder< Real, Device, Index, RealAllocator >;
};

template< typename Device,
          typename Index,
          typename RealAllocator >
struct SparseMatrixValuesHolderSetter< bool, Device, Index, RealAllocator >
{
   using type = BooleanValuesHolder< Device, Index >;
};

      } //namespace details
   } //namepsace Matrices
} //namespace TNL
