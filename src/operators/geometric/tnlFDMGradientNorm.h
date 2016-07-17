/***************************************************************************
                          tnlFDMGradientNorm.h  -  description
                             -------------------
    begin                : Jan 11, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <operators/fdm/tnlForwardFiniteDifference.h>
#include <operators/geometric/tnlExactGradientNorm.h>
#include <operators/tnlOperator.h>

namespace TNL {

template< typename Mesh,
          template< typename, int, int, int, typename, typename > class DifferenceOperatorTemplate = tnlForwardFiniteDifference,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlFDMGradientNorm
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          template< typename, int, int, int, typename, typename > class DifferenceOperatorTemplate,
          typename Real,
          typename Index >
class tnlFDMGradientNorm< tnlGrid< 1,MeshReal, Device, MeshIndex >, DifferenceOperatorTemplate, Real, Index >
   : public tnlOperator< tnlGrid< 1, MeshReal, Device, MeshIndex >,
                         MeshInteriorDomain, 1, 1, Real, Index >
{
   public:
 
   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlExactGradientNorm< 1, RealType > ExactOperatorType;
 
   template< typename MeshEntity = typename MeshType::Cell >
   using XDifferenceOperatorType = DifferenceOperatorTemplate< typename MeshEntity::MeshType, 1, 0, 0, Real, Index >;
 
   tnlFDMGradientNorm()
   : epsSquare( 0.0 ){}

   static tnlString getType()
   {
      return tnlString( "tnlFDMGradientNorm< " ) +
         MeshType::getType() + ", " +
         ::getType< Real >() + ", " +
         ::getType< Index >() + " >";
   }

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const
   {
      XDifferenceOperatorType< MeshEntity > XDifference;
      const RealType u_x = XDifference( u, entity );
      return sqrt( this->epsSquare + u_x * u_x );
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
class tnlFDMGradientNorm< tnlGrid< 2,MeshReal, Device, MeshIndex >, DifferenceOperatorTemplate, Real, Index >
   : public tnlOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >,
                         MeshInteriorDomain, 2, 2, Real, Index >
{
   public:
 
      typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlExactGradientNorm< 2, RealType > ExactOperatorType;
 
      template< typename MeshEntity >
      using XDifferenceOperatorType = DifferenceOperatorTemplate< typename MeshEntity::MeshType, 1, 0, 0, Real, Index >;
      template< typename MeshEntity >
      using YDifferenceOperatorType = DifferenceOperatorTemplate< typename MeshEntity::MeshType, 0, 1, 0, Real, Index >;

      tnlFDMGradientNorm()
      : epsSquare( 0.0 ){}


      static tnlString getType()
      {
         return tnlString( "tnlFDMGradientNorm< " ) +
            MeshType::getType() + ", " +
            ::getType< Real >() + ", " +
            ::getType< Index >() + " >";

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
         return sqrt( this->epsSquare + u_x * u_x + u_y * u_y );
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
class tnlFDMGradientNorm< tnlGrid< 3, MeshReal, Device, MeshIndex >, DifferenceOperatorTemplate, Real, Index >
   : public tnlOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >,
                         MeshInteriorDomain, 3, 3, Real, Index >
{
   public:
 
      typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlExactGradientNorm< 3, RealType > ExactOperatorType;
 
      template< typename MeshEntity >
      using XDifferenceOperatorType = DifferenceOperatorTemplate< typename MeshEntity::MeshType, 1, 0, 0, Real, Index >;
      template< typename MeshEntity >
      using YDifferenceOperatorType = DifferenceOperatorTemplate< typename MeshEntity::MeshType, 0, 1, 0, Real, Index >;
      template< typename MeshEntity >
      using ZDifferenceOperatorType = DifferenceOperatorTemplate< typename MeshEntity::MeshType, 0, 0, 1, Real, Index >;

 
      tnlFDMGradientNorm()
      : epsSquare( 0.0 ){}

      static tnlString getType()
      {
         return tnlString( "tnlFDMGradientNorm< " ) +
            MeshType::getType() + ", " +
            ::getType< Real >() + ", " +
            ::getType< Index >() + " >";
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
         return sqrt( this->epsSquare + u_x * u_x + u_y * u_y + u_z * u_z );
      }


      void setEps(const Real& eps)
      {
         this->epsSquare = eps*eps;
      }
 
   private:
 
      RealType epsSquare;
};

} // namespace TNL

