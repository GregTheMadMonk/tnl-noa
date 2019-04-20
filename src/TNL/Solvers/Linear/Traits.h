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

#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Containers/DistributedVector.h>
#include <TNL/Containers/DistributedVectorView.h>
#include <TNL/Matrices/DistributedMatrix.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
struct Traits
{
   using CommunicatorType = Communicators::NoDistrCommunicator;

   using VectorType = Containers::Vector
         < typename Matrix::RealType,
           typename Matrix::DeviceType,
           typename Matrix::IndexType >;
   using VectorViewType = Containers::VectorView
         < typename Matrix::RealType,
           typename Matrix::DeviceType,
           typename Matrix::IndexType >;
   using ConstVectorViewType = Containers::VectorView
         < std::add_const_t< typename Matrix::RealType >,
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

   static typename CommunicatorType::CommunicationGroup getCommunicationGroup( const Matrix& m ) { return CommunicatorType::AllGroup; }
};

template< typename Matrix, typename Communicator >
struct Traits< Matrices::DistributedMatrix< Matrix, Communicator > >
{
   using CommunicatorType = Communicator;

   using VectorType = Containers::DistributedVector
         < typename Matrix::RealType,
           typename Matrix::DeviceType,
           typename Matrix::IndexType,
           Communicator >;
   using VectorViewType = Containers::DistributedVectorView
         < typename Matrix::RealType,
           typename Matrix::DeviceType,
           typename Matrix::IndexType,
           Communicator >;
   using ConstVectorViewType = Containers::DistributedVectorView
         < std::add_const_t< typename Matrix::RealType >,
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
         < std::add_const_t< typename Matrix::RealType >,
           typename Matrix::DeviceType,
           typename Matrix::IndexType >;

   // compatibility wrappers for some DistributedMatrix methods
   static const Matrix& getLocalMatrix( const Matrices::DistributedMatrix< Matrix, Communicator >& m )
   { return m.getLocalMatrix(); }
   static ConstLocalVectorViewType getLocalVectorView( ConstVectorViewType v ) { return v.getLocalVectorView(); }
   static LocalVectorViewType getLocalVectorView( VectorViewType v ) { return v.getLocalVectorView(); }

   static typename CommunicatorType::CommunicationGroup getCommunicationGroup( const Matrices::DistributedMatrix< Matrix, Communicator >& m ) { return m.getCommunicationGroup(); }
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL
