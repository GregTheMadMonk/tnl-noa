/***************************************************************************
                          SparseMatrix.h  -  description
                             -------------------
    begin                : Jan 5, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/TypeTraits.h>

namespace TNL {
   namespace Matrices {
      namespace details {

template< typename VectorOrView,
          std::enable_if_t< HasSetSizeMethod< VectorOrView >::value, bool > = true >
static void set_size_if_resizable( VectorOrView& v, typename VectorOrView::IndexType size )
{
   v.setSize( size );
}

template< typename VectorOrView,
          std::enable_if_t< ! HasSetSizeMethod< VectorOrView >::value, bool > = true >
static void set_size_if_resizable( VectorOrView& v, typename VectorOrView::IndexType size )
{
   TNL_ASSERT_EQ( v.getSize(), size, "view has wrong size" );
}

      } //namespace details
   } //namepsace Matrices
} //namespace TNL
