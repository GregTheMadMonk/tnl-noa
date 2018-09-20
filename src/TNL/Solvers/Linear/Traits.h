/***************************************************************************
                          Traits.h  -  description
                             -------------------
    begin                : Sep 20, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Containers/VectorView.h>
#include <TNL/DistributedContainers/DistributedVectorView.h>
#include <TNL/DistributedContainers/DistributedMatrix.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
struct Traits
{
   using VectorType = Containers::Vector
         < typename Matrix::RealType,
           typename Matrix::DeviceType,
           typename Matrix::IndexType >;
   using VectorViewType = Containers::VectorView
         < typename Matrix::RealType,
           typename Matrix::DeviceType,
           typename Matrix::IndexType >;
   using ConstVectorViewType = Containers::VectorView
         < typename std::add_const< typename Matrix::RealType >::type,
           typename Matrix::DeviceType,
           typename Matrix::IndexType >;

   // compatibility aliases
   using LocalVectorType = VectorType;
   using LocalVectorViewType = VectorViewType;
   using ConstLocalVectorViewType = ConstVectorViewType;

   // compatibility wrappers for some DistributedMatrix methods
   static const Matrix& getLocalMatrix( const Matrix& m ) { return m; }
   static ConstLocalVectorViewType getLocalVectorView( ConstVectorViewType v ) { return v; }
   static LocalVectorViewType getLocalVectorView( VectorViewType v ) { return v; }
};

template< typename Matrix, typename Communicator >
struct Traits< DistributedContainers::DistributedMatrix< Matrix, Communicator > >
{
   using VectorType = DistributedContainers::DistributedVector
         < typename Matrix::RealType,
           typename Matrix::DeviceType,
           typename Matrix::IndexType,
           Communicator >;
   using VectorViewType = DistributedContainers::DistributedVectorView
         < typename Matrix::RealType,
           typename Matrix::DeviceType,
           typename Matrix::IndexType,
           Communicator >;
   using ConstVectorViewType = DistributedContainers::DistributedVectorView
         < typename std::add_const< typename Matrix::RealType >::type,
           typename Matrix::DeviceType,
           typename Matrix::IndexType,
           Communicator >;

   using LocalVectorType = Containers::Vector
         < typename Matrix::RealType,
           typename Matrix::DeviceType,
           typename Matrix::IndexType >;
   using LocalVectorViewType = Containers::VectorView
         < typename Matrix::RealType,
           typename Matrix::DeviceType,
           typename Matrix::IndexType >;
   using ConstLocalVectorViewType = Containers::VectorView
         < typename std::add_const< typename Matrix::RealType >::type,
           typename Matrix::DeviceType,
           typename Matrix::IndexType >;

   // compatibility wrappers for some DistributedMatrix methods
   static const Matrix& getLocalMatrix( const DistributedContainers::DistributedMatrix< Matrix, Communicator >& m )
   { return m.getLocalMatrix(); }
   static ConstLocalVectorViewType getLocalVectorView( ConstVectorViewType v ) { return v.getLocalVectorView(); }
   static LocalVectorViewType getLocalVectorView( VectorViewType v ) { return v.getLocalVectorView(); }
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL
