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

#include <TNL/MPI/Wrappers.h>
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
   using LocalViewType = VectorViewType;
   using ConstLocalViewType = ConstVectorViewType;

   // compatibility wrappers for some DistributedMatrix methods
   static const Matrix& getLocalMatrix( const Matrix& m ) { return m; }
   static ConstLocalViewType getConstLocalView( ConstVectorViewType v ) { return v; }
   static LocalViewType getLocalView( VectorViewType v ) { return v; }

   static MPI_Comm getCommunicationGroup( const Matrix& m ) { return MPI::AllGroup(); }
   static void startSynchronization( VectorViewType v ) {}
   static void waitForSynchronization( VectorViewType v ) {}
};

template< typename Matrix >
struct Traits< Matrices::DistributedMatrix< Matrix > >
{
   using VectorType = Containers::DistributedVector
         < typename Matrix::RealType,
           typename Matrix::DeviceType,
           typename Matrix::IndexType >;
   using VectorViewType = Containers::DistributedVectorView
         < typename Matrix::RealType,
           typename Matrix::DeviceType,
           typename Matrix::IndexType >;
   using ConstVectorViewType = Containers::DistributedVectorView
         < std::add_const_t< typename Matrix::RealType >,
           typename Matrix::DeviceType,
           typename Matrix::IndexType >;

   using LocalVectorType = Containers::Vector
         < typename Matrix::RealType,
           typename Matrix::DeviceType,
           typename Matrix::IndexType >;
   using LocalViewType = Containers::VectorView
         < typename Matrix::RealType,
           typename Matrix::DeviceType,
           typename Matrix::IndexType >;
   using ConstLocalViewType = Containers::VectorView
         < std::add_const_t< typename Matrix::RealType >,
           typename Matrix::DeviceType,
           typename Matrix::IndexType >;

   // compatibility wrappers for some DistributedMatrix methods
   static const Matrix& getLocalMatrix( const Matrices::DistributedMatrix< Matrix >& m ) { return m.getLocalMatrix(); }
   static ConstLocalViewType getConstLocalView( ConstVectorViewType v ) { return v.getConstLocalView(); }
   static LocalViewType getLocalView( VectorViewType v ) { return v.getLocalView(); }

   static MPI_Comm getCommunicationGroup( const Matrices::DistributedMatrix< Matrix >& m ) { return m.getCommunicationGroup(); }
   static void startSynchronization( VectorViewType v ) { v.startSynchronization(); }
   static void waitForSynchronization( VectorViewType v ) { v.waitForSynchronization(); }
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL
