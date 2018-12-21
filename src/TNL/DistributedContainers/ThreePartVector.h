/***************************************************************************
                          ThreePartVector.h  -  description
                             -------------------
    begin                : Dec 19, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>

namespace TNL {
namespace DistributedContainers {

template< typename Real,
          typename Device = Devices::Host,
          typename Index = int >
class ThreePartVectorView
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using VectorView = Containers::VectorView< Real, Device, Index >;

   ThreePartVectorView() = default;
   ThreePartVectorView( const ThreePartVectorView& ) = default;
   ThreePartVectorView( ThreePartVectorView&& ) = default;

   ThreePartVectorView( VectorView view_left, VectorView view_mid, VectorView view_right )
   {
      bind( view_left, view_mid, view_right );
   }

   void bind( VectorView view_left, VectorView view_mid, VectorView view_right )
   {
      left.bind( view_left );
      middle.bind( view_mid );
      right.bind( view_right );
   }

   void reset()
   {
      left.reset();
      middle.reset();
      right.reset();
   }

//   __cuda_callable__
//   Real& operator[]( Index i )
//   {
//      if( i < left.getSize() )
//         return left[ i ];
//      else if( i < left.getSize() + middle.getSize() )
//         return middle[ i - left.getSize() ];
//      else
//         return right[ i - left.getSize() - middle.getSize() ];
//   }

   __cuda_callable__
   const Real& operator[]( Index i ) const
   {
      if( i < left.getSize() )
         return left[ i ];
      else if( i < left.getSize() + middle.getSize() )
         return middle[ i - left.getSize() ];
      else
         return right[ i - left.getSize() - middle.getSize() ];
   }

   friend std::ostream& operator<<( std::ostream& str, const ThreePartVectorView& v )
   {
      str << "[\n\tleft: " << v.left << ",\n\tmiddle: " << v.middle << ",\n\tright: " << v.right << "\n]";
      return str;
   }

protected:
   VectorView left, middle, right;
};

template< typename Real,
          typename Device = Devices::Host,
          typename Index = int >
class ThreePartVector
{
   using ConstReal = typename std::add_const< Real >::type;
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using Vector = Containers::Vector< Real, Device, Index >;
   using VectorView = Containers::VectorView< Real, Device, Index >;
   using ConstVectorView = Containers::VectorView< ConstReal, Device, Index >;

   ThreePartVector() = default;
   ThreePartVector( ThreePartVector& ) = default;

   void init( Index size_left, ConstVectorView view_mid, Index size_right )
   {
      left.setSize( size_left );
      middle.bind( view_mid );
      right.setSize( size_right );
   }

   void reset()
   {
      left.reset();
      middle.reset();
      right.reset();
   }

   ThreePartVectorView< ConstReal, Device, Index > getConstView()
   {
      return {left, middle, right};
   }

//   __cuda_callable__
//   Real& operator[]( Index i )
//   {
//      if( i < left.getSize() )
//         return left[ i ];
//      else if( i < left.getSize() + middle.getSize() )
//         return middle[ i - left.getSize() ];
//      else
//         return right[ i - left.getSize() - middle.getSize() ];
//   }

   __cuda_callable__
   const Real& operator[]( Index i ) const
   {
      if( i < left.getSize() )
         return left[ i ];
      else if( i < left.getSize() + middle.getSize() )
         return middle[ i - left.getSize() ];
      else
         return right[ i - left.getSize() - middle.getSize() ];
   }

   friend std::ostream& operator<<( std::ostream& str, const ThreePartVector& v )
   {
      str << "[\n\tleft: " << v.left << ",\n\tmiddle: " << v.middle << ",\n\tright: " << v.right << "\n]";
      return str;
   }

protected:
   Vector left, right;
   ConstVectorView middle;
};

} // namespace DistributedContainers
} // namespace TNL
