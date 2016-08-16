/***************************************************************************
                          FDMGradientNorm.h  -  description
                             -------------------
    begin                : Jan 11, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Operators/fdm/ForwardFiniteDifference.h>
#include <TNL/Operators/geometric/ExactGradientNorm.h>
#include <TNL/Operators/Operator.h>

namespace TNL {
namespace Operators {   

template< typename Mesh,
          template< typename, int, int, int, typename, typename > class DifferenceOperatorTemplate = ForwardFiniteDifference,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class FDMGradientNorm
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          template< typename, int, int, int, typename, typename > class DifferenceOperatorTemplate,
          typename Real,
          typename Index >
class FDMGradientNorm< Meshes::Grid< 1,MeshReal, Device, MeshIndex >, DifferenceOperatorTemplate, Real, Index >
   : public Operator< Meshes::Grid< 1, MeshReal, Device, MeshIndex >,
                         Functions::MeshInteriorDomain, 1, 1, Real, Index >
{
   public:
 
   typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef ExactGradientNorm< 1, RealType > ExactOperatorType;
 
   template< typename MeshEntity = typename MeshType::Cell >
   using XDifferenceOperatorType = DifferenceOperatorTemplate< typename MeshEntity::MeshType, 1, 0, 0, Real, Index >;
 
   FDMGradientNorm()
   : epsSquare( 0.0 ){}

   static String getType()
   {
      return String( "FDMGradientNorm< " ) +
         MeshType::getType() + ", " +
        TNL::getType< Real >() + ", " +
        TNL::getType< Index >() + " >";
   }

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const
   {
      XDifferenceOperatorType< MeshEntity > XDifference;
      const RealType u_x = XDifference( u, entity );
      return ::sqrt( this->epsSquare + u_x * u_x );
   }
 
   void setEps( const Real& eps )
   {
      this->epsSquare = eps*eps;
   }
 
   private:
 
   RealType epsSquare;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          template< typename, int, int, int, typename, typename > class DifferenceOperatorTemplate,
          typename Real,
          typename Index >
class FDMGradientNorm< Meshes::Grid< 2,MeshReal, Device, MeshIndex >, DifferenceOperatorTemplate, Real, Index >
   : public Operator< Meshes::Grid< 2, MeshReal, Device, MeshIndex >,
                         Functions::MeshInteriorDomain, 2, 2, Real, Index >
{
   public:
 
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef ExactGradientNorm< 2, RealType > ExactOperatorType;
 
      template< typename MeshEntity >
      using XDifferenceOperatorType = DifferenceOperatorTemplate< typename MeshEntity::MeshType, 1, 0, 0, Real, Index >;
      template< typename MeshEntity >
      using YDifferenceOperatorType = DifferenceOperatorTemplate< typename MeshEntity::MeshType, 0, 1, 0, Real, Index >;

      FDMGradientNorm()
      : epsSquare( 0.0 ){}


      static String getType()
      {
         return String( "FDMGradientNorm< " ) +
            MeshType::getType() + ", " +
           TNL::getType< Real >() + ", " +
           TNL::getType< Index >() + " >";

      }

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         XDifferenceOperatorType< MeshEntity > XDifference;
         YDifferenceOperatorType< MeshEntity > YDifference;
         const RealType u_x = XDifference( u, entity );
         const RealType u_y = YDifference( u, entity );
         return ::sqrt( this->epsSquare + u_x * u_x + u_y * u_y );
      }



      void setEps( const Real& eps )
      {
         this->epsSquare = eps*eps;
      }
 
   private:
 
      RealType epsSquare;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          template< typename, int, int, int, typename, typename > class DifferenceOperatorTemplate,
          typename Real,
          typename Index >
class FDMGradientNorm< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, DifferenceOperatorTemplate, Real, Index >
   : public Operator< Meshes::Grid< 3, MeshReal, Device, MeshIndex >,
                         Functions::MeshInteriorDomain, 3, 3, Real, Index >
{
   public:
 
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef ExactGradientNorm< 3, RealType > ExactOperatorType;
 
      template< typename MeshEntity >
      using XDifferenceOperatorType = DifferenceOperatorTemplate< typename MeshEntity::MeshType, 1, 0, 0, Real, Index >;
      template< typename MeshEntity >
      using YDifferenceOperatorType = DifferenceOperatorTemplate< typename MeshEntity::MeshType, 0, 1, 0, Real, Index >;
      template< typename MeshEntity >
      using ZDifferenceOperatorType = DifferenceOperatorTemplate< typename MeshEntity::MeshType, 0, 0, 1, Real, Index >;

 
      FDMGradientNorm()
      : epsSquare( 0.0 ){}

      static String getType()
      {
         return String( "FDMGradientNorm< " ) +
            MeshType::getType() + ", " +
           TNL::getType< Real >() + ", " +
           TNL::getType< Index >() + " >";
      }

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         XDifferenceOperatorType< MeshEntity > XDifference;
         YDifferenceOperatorType< MeshEntity > YDifference;
         ZDifferenceOperatorType< MeshEntity > ZDifference;

         const RealType u_x = XDifference( u, entity );
         const RealType u_y = YDifference( u, entity );
         const RealType u_z = ZDifference( u, entity );
         return ::sqrt( this->epsSquare + u_x * u_x + u_y * u_y + u_z * u_z );
      }


      void setEps(const Real& eps)
      {
         this->epsSquare = eps*eps;
      }
 
   private:
 
      RealType epsSquare;
};

} // namespace Operators
} // namespace TNL
