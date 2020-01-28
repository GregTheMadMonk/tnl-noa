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
          typename Index,
          typename RealAllocator >
class ValuesHolder
: public Containers::Vector< Real, Device, Index, RealAllocator >
{};

template< typename Device,
          typename Index >
class BooleanValuesHolder
{
   public:

      using RealType = bool;
      using DeviceType = Device;
      using IndexType = Index;

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
