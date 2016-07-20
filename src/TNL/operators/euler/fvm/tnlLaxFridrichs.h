/***************************************************************************
                          tnlLaxFridrichs.h  -  description
                             -------------------
    begin                : Mar 1, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Vectors/SharedVector.h>
#include <TNL/mesh/tnlGrid.h>
#include <TNL/mesh/tnlIdenticalGridGeometry.h>
#include <TNL/operators/gradient/tnlCentralFDMGradient.h>

namespace TNL {

template< typename Mesh,
          typename PressureGradient = tnlCentralFDMGradient< Mesh > >
class tnlLaxFridrichs
{
};

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient,
          template< int, typename, typename, typename > class GridGeometry >
class tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index, GridGeometry >, PressureGradient >
{
   public:

   typedef tnlGrid< 2, Real, Device, Index, GridGeometry > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename MeshType :: VertexType VertexType;
   typedef typename MeshType :: CoordinatesType CoordinatesType;

   tnlLaxFridrichs();

   static String getTypeStatic();

   void getExplicitRhs( const IndexType centralVolume,
                        RealType& rho_t,
                        RealType& rho_u1_t,
                        RealType& rho_u2_t,
                        const RealType& tau ) const;

   void getExplicitRhs( const IndexType centralVolume,
                        RealType& rho_t,
                        RealType& rho_u1_t,
                        RealType& rho_u2_t,
                        RealType& e_t,
                        const RealType& tau ) const;

   void setRegularization( const RealType& epsilon );

   void setViscosityCoefficient( const RealType& v );

   void bindMesh( const MeshType& mesh );

   const MeshType& getMesh() const;

   template< typename Vector >
   void setRho( Vector& rho ); // TODO: add const

   template< typename Vector >
   void setRhoU1( Vector& rho_u1 ); // TODO: add const

   template< typename Vector >
   void setRhoU2( Vector& rho_u2 ); // TODO: add const

   template< typename Vector >
   void setE( Vector& e ); // TODO: add const


   template< typename Vector >
   void setPressureGradient( Vector& grad_p ); // TODO: add const

   protected:

   RealType regularize( const RealType& r ) const;

   RealType regularizeEps, viscosityCoefficient;

   const MeshType* mesh;

   const PressureGradient* pressureGradient;

   tnlSharedVector< RealType, DeviceType, IndexType > rho, rho_u1, rho_u2, e;
};

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
class tnlLaxFridrichs< tnlGrid< 2, Real, Device, Index, tnlIdenticalGridGeometry >, PressureGradient >
{
   public:

   typedef tnlGrid< 2, Real, Device, Index, tnlIdenticalGridGeometry > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename MeshType :: VertexType VertexType;
   typedef typename MeshType :: CoordinatesType CoordinatesType;

   tnlLaxFridrichs();

   void getExplicitRhs( const IndexType centralVolume,
                        RealType& rho_t,
                        RealType& rho_u1_t,
                        RealType& rho_u2_t,
                        const RealType& tau ) const;

   void getExplicitRhs( const IndexType centralVolume,
                        RealType& rho_t,
                        RealType& rho_u1_t,
                        RealType& rho_u2_t,
                        RealType& e_t,
                        const RealType& tau ) const;

   void setRegularization( const RealType& epsilon );

   void setViscosityCoefficient( const RealType& v );

   void bindMesh( const MeshType& mesh );

   template< typename Vector >
   void setRho( Vector& rho ); // TODO: add const

   template< typename Vector >
   void setRhoU1( Vector& rho_u1 ); // TODO: add const

   template< typename Vector >
   void setRhoU2( Vector& rho_u2 ); // TODO: add const

   template< typename Vector >
   void setE( Vector& e ); // TODO: add const

   template< typename Vector >
   void setP( Vector& p ); // TODO: add const



   template< typename Vector >
   void setPressureGradient( Vector& grad_p ); // TODO: add const

   protected:

   RealType regularize( const RealType& r ) const;

   RealType regularizeEps, viscosityCoefficient;

   const MeshType* mesh;

   const PressureGradient* pressureGradient;

   tnlSharedVector< RealType, DeviceType, IndexType > rho, rho_u1, rho_u2, energy, p;
};

} // namespace TNL

#include <TNL/implementation/operators/euler/fvm/tnlLaxFridrichs_impl.h>
