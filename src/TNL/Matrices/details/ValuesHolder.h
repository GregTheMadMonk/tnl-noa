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
          typename Index,
          typename RealAllocator >
class ValuesHolder< bool, Device, Index, RealAllocator >
{
   public:

      using RealType = bool;
      using DeviceType = Device;
      using IndexType = Index;

      ValuesHolder()
      : size( 0 ){};

      ValuesdHolder( const IndexType& size )
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
 * \brief Serialization of arrays into binary files.
 */
template< typename Device, typename Index, typename Allocator >
File& operator<<( File& file, const ValuesHolder< bool, Device, Index, Allocator >& array ) { return file; };

template< typename Device, typename Index, typename Allocator >
File& operator<<( File&& file, const ValuesHolder< bool, Device, Index, Allocator >& array ) { return file; };

/**
 * \brief Deserialization of arrays from binary files.
 */
template< typename Device, typename Index, typename Allocator >
File& operator>>( File& file, ValuesHolder< bool, Device, Index, Allocator >& array ) { return file; };

template< typename Device, typename Index, typename Allocator >
File& operator>>( File&& file, ValuesHolder< bool, Device, Index, Allocator >& array ) { return file; };


      } //namespace details
   } //namepsace Matrices
} //namespace TNL