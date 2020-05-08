/***************************************************************************
                          SparseMatrix.h  -  description
                             -------------------
    begin                : Jan 5, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/ArrayView.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Containers/DistributedArray.h>
#include <TNL/Containers/DistributedVector.h>


namespace TNL {
   namespace Matrices {
      namespace details {

template< typename Vector >
struct CompressedRowLengthVectorSizeSetter
{
   static void setSize( Vector& v, typename Vector::IndexType size )
   {
      v.setSize( size );
   }
};

template< typename Value,
   typename Device,
   typename Index >
struct CompressedRowLengthVectorSizeSetter< Containers::ArrayView< Value, Device, Index > >
{
   static void setSize( Containers::ArrayView< Value, Device, Index >& v, Index size )
   {
      TNL_ASSERT_EQ( v.getSize(), size, "ArrayView has wrong size, different from number of matrix rows." );
   }
};

template< typename Value,
   typename Device,
   typename Index >
struct CompressedRowLengthVectorSizeSetter< Containers::VectorView< Value, Device, Index > >
{
   static void setSize( Containers::VectorView< Value, Device, Index >& v, Index size )
   {
      TNL_ASSERT_EQ( v.getSize(), size, "VectorView has wrong size, different from number of matrix rows." );
   }
};

template< typename Value,
   typename Device,
   typename Index,
   typename Communicator >
struct CompressedRowLengthVectorSizeSetter< Containers::DistributedArray< Value, Device, Index, Communicator > >
{
   static void setSize( Containers::DistributedArray< Value, Device, Index, Communicator >& v, Index size )
   {
      TNL_ASSERT_EQ( v.getSize(), size, "DistributedArray has wrong size, different from number of matrix rows." );
   }
};

template< typename Value,
   typename Device,
   typename Index,
   typename Communicator >
struct CompressedRowLengthVectorSizeSetter< Containers::DistributedVector< Value, Device, Index, Communicator > >
{
   static void setSize( Containers::DistributedVector< Value, Device, Index, Communicator >& v, Index size )
   {
      TNL_ASSERT_EQ( v.getSize(), size, "DistributedVector has wrong size, different from number of matrix rows." );
   }
};

      } //namespace details
   } //namepsace Matrices
} //namespace TNL
