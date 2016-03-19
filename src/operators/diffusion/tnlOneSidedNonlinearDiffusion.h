/***************************************************************************
                          tnlOneSidedNonlinearDiffusion.h  -  description
                             -------------------
    begin                : Feb 16, 2016
    copyright            : (C) 2016 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/


#ifndef TNLONESIDEDNONLINEARDIFFUSION_H
#define	TNLONESIDEDNONLINEARDIFFUSION_H

#include <core/vectors/tnlVector.h>
#include <mesh/tnlGrid.h>
#include <operators/diffusion/tnlExactNonlinearDiffusion.h>

template< typename Mesh,
          typename Nonlinearity,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlOneSidedNonlinearDiffusion
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Nonlinearity,
          typename Real,
          typename Index >
class tnlOneSidedNonlinearDiffusion< tnlGrid< 1,MeshReal, Device, MeshIndex >, Nonlinearity, Real, Index >
{
   public: 
   
      typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Nonlinearity NonlinearityType;
      typedef typename MeshType::template MeshEntity< MeshType::getMeshDimensions() > CellType;
      typedef tnlExactNonlinearDiffusion< MeshType::getMeshDimensions(), typename Nonlinearity::ExactOperatorType, Real > ExactOperatorType;

      tnlOneSidedNonlinearDiffusion( const Nonlinearity& nonlinearity )
      : nonlinearity( nonlinearity ){}
      
      static tnlString getType()
      {
         return tnlString( "tnlOneSidedNonlinearDiffusion< " ) +
            MeshType::getType() + ", " +
            Nonlinearity::getType() + "," +
            ::getType< Real >() + ", " +
            ::getType< Index >() + " >";         
      }     

      template< typename MeshFunction,
                typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();
         const typename MeshEntity::MeshType& mesh = entity.getMesh();
         const RealType& hx_div = entity.getMesh().template getSpaceStepsProducts< -2 >();
         const IndexType& center = entity.getIndex();
         const IndexType& east = neighbourEntities.template getEntityIndex<  1 >();
         const IndexType& west = neighbourEntities.template getEntityIndex< -1 >();
         const RealType& u_c = u[ center ];
         const RealType u_x_f = ( u[ east ] - u_c );
         const RealType u_x_b = ( u_c - u[ west ] );
         return ( u_x_f * this->nonlinearity[ center ] -
                  u_x_b * this->nonlinearity[ west ] ) * hx_div;
         
      }

      template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const
      {
         return 3;
      }

      template< typename PreimageFunction,
                typename MeshEntity,
                typename Matrix,
                typename Vector >
      __cuda_callable__
      inline void setMatrixElements( const PreimageFunction& u,
                                     const MeshEntity& entity,
                                     const RealType& time,
                                     const RealType& tau,
                                     Matrix& matrix,
                                     Vector& b ) const
      {
         typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
         const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();
         const IndexType& center = entity.getIndex();
         const IndexType& east = neighbourEntities.template getEntityIndex<  1 >();
         const IndexType& west = neighbourEntities.template getEntityIndex< -1 >();
         const RealType lambda_x = tau * entity.getMesh().template getSpaceStepsProducts< -2 >();
         const RealType& nonlinearity_center = this->nonlinearity[ center ];
         const RealType& nonlinearity_west = this->nonlinearity[ west ];
         const RealType aCoef = -lambda_x * nonlinearity_west;
         const RealType bCoef = lambda_x * ( nonlinearity_center + nonlinearity_west );              
         const RealType cCoef = -lambda_x * nonlinearity_center;
         matrixRow.setElement( 0, west,   aCoef );
         matrixRow.setElement( 1, center, bCoef );
         matrixRow.setElement( 2, east,   cCoef );
      }

   public:
       
      const Nonlinearity& nonlinearity;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Nonlinearity,
          typename Real,
          typename Index >
class tnlOneSidedNonlinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Nonlinearity, Real, Index >
{
   public: 
   
      typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Nonlinearity NonlinearityType;
      typedef tnlExactNonlinearDiffusion< MeshType::getMeshDimensions(), typename Nonlinearity::ExactOperatorType, Real > ExactOperatorType;      

      tnlOneSidedNonlinearDiffusion( const Nonlinearity& nonlinearity )
      : nonlinearity( nonlinearity ){}      
      
      static tnlString getType()
      {
         return tnlString( "tnlOneSidedNonlinearDiffusion< " ) +
            MeshType::getType() + ", " +
            Nonlinearity::getType() + "," +
            ::getType< Real >() + ", " +
            ::getType< Index >() + " >";         
      }      

      template< typename MeshFunction,
                typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
         const typename MeshEntity::MeshType& mesh = entity.getMesh();
         const RealType& hx_div = entity.getMesh().template getSpaceStepsProducts< -2,  0 >();
         const RealType& hy_div = entity.getMesh().template getSpaceStepsProducts<  0, -2 >();
         const IndexType& center = entity.getIndex();
         const IndexType& east = neighbourEntities.template getEntityIndex<  1, 0 >();
         const IndexType& west = neighbourEntities.template getEntityIndex< -1, 0 >();
         const IndexType& north = neighbourEntities.template getEntityIndex< 0,  1 >();
         const IndexType& south = neighbourEntities.template getEntityIndex< 0, -1 >();         
         const RealType& u_c = u[ center ];
         const RealType u_x_f = ( u[ east ] - u_c );
         const RealType u_x_b = ( u_c - u[ west ] );
         const RealType u_y_f = ( u[ north ] - u_c );
         const RealType u_y_b = ( u_c - u[ south ] );
         
         const RealType& nonlinearity_center = this->nonlinearity[ center ];
         return ( u_x_f * nonlinearity_center - u_x_b * this->nonlinearity[ west ] ) * hx_div +
                ( u_y_f * nonlinearity_center - u_y_b * this->nonlinearity[ south ] ) * hy_div;
      }

      template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const
      {
         return 5;
      }

      template< typename PreimageFunction,
                typename MeshEntity,
                typename Matrix,
                typename Vector >
      __cuda_callable__
      inline void setMatrixElements( const PreimageFunction& u,
                                     const MeshEntity& entity,
                                     const RealType& time,
                                     const RealType& tau,
                                     Matrix& matrix,
                                     Vector& b ) const
      {
         typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
         const IndexType& center = entity.getIndex();
         const IndexType& east  = neighbourEntities.template getEntityIndex<  1,  0 >();
         const IndexType& west  = neighbourEntities.template getEntityIndex< -1,  0 >();
         const IndexType& north = neighbourEntities.template getEntityIndex<  0,  1 >();
         const IndexType& south = neighbourEntities.template getEntityIndex<  0, -1 >();                  
         const RealType lambda_x = tau * entity.getMesh().template getSpaceStepsProducts< -2,  0 >();
         const RealType lambda_y = tau * entity.getMesh().template getSpaceStepsProducts<  0, -2 >();
         const RealType& nonlinearity_center = this->nonlinearity[ center ];
         const RealType& nonlinearity_west = this->nonlinearity[ west ];
         const RealType& nonlinearity_south = this->nonlinearity[ south ];
         const RealType aCoef = -lambda_y * nonlinearity_south;
         const RealType bCoef = -lambda_x * nonlinearity_west;
         const RealType cCoef = lambda_x * ( nonlinearity_center + nonlinearity_west ) +
                                lambda_y * ( nonlinearity_center + nonlinearity_south );
         const RealType dCoef = -lambda_x * nonlinearity_center;
         const RealType eCoef = -lambda_y * nonlinearity_center;
         matrixRow.setElement( 0, south,  aCoef );
         matrixRow.setElement( 1, west,   bCoef );
         matrixRow.setElement( 2, center, cCoef );
         matrixRow.setElement( 3, east,   dCoef );
         matrixRow.setElement( 4, north,  eCoef );         
      }
   
   public:
       
      const Nonlinearity& nonlinearity;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Nonlinearity,
          typename Real,
          typename Index >
class tnlOneSidedNonlinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Nonlinearity, Real, Index >
{
   public: 
   
      typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Nonlinearity NonlinearityType;
      typedef tnlExactNonlinearDiffusion< MeshType::getMeshDimensions(), typename Nonlinearity::ExactOperatorType, Real > ExactOperatorType;

      tnlOneSidedNonlinearDiffusion( const Nonlinearity& nonlinearity )
      : nonlinearity( nonlinearity ){}
      
      static tnlString getType()
      {
         return tnlString( "tnlOneSidedNonlinearDiffusion< " ) +
            MeshType::getType() + ", " +
            Nonlinearity::getType() + "," +
            ::getType< Real >() + ", " +
            ::getType< Index >() + " >";         
      }

      template< typename MeshFunction,
                typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();
         const typename MeshEntity::MeshType& mesh = entity.getMesh();
         const RealType& hx_div = entity.getMesh().template getSpaceStepsProducts< -2,  0,  0 >();
         const RealType& hy_div = entity.getMesh().template getSpaceStepsProducts<  0, -2,  0 >();
         const RealType& hz_div = entity.getMesh().template getSpaceStepsProducts<  0,  0, -2 >();
         const IndexType& center = entity.getIndex();
         const IndexType& east  = neighbourEntities.template getEntityIndex<  1,  0,  0 >();
         const IndexType& west  = neighbourEntities.template getEntityIndex< -1,  0,  0 >();
         const IndexType& north = neighbourEntities.template getEntityIndex<  0,  1,  0 >();
         const IndexType& south = neighbourEntities.template getEntityIndex<  0, -1,  0 >();
         const IndexType& up    = neighbourEntities.template getEntityIndex<  0,  0,  1 >();
         const IndexType& down  = neighbourEntities.template getEntityIndex<  0,  0, -1 >();         
         
         const RealType& u_c = u[ center ];
         const RealType u_x_f = ( u[ east ] - u_c );
         const RealType u_x_b = ( u_c - u[ west ] );
         const RealType u_y_f = ( u[ north ] - u_c );
         const RealType u_y_b = ( u_c - u[ south ] );
         const RealType u_z_f = ( u[ up ] - u_c );
         const RealType u_z_b = ( u_c - u[ down ] );         
         
         const RealType& nonlinearity_center = this->nonlinearity[ center ];
         return ( u_x_f * nonlinearity_center - u_x_b * this->nonlinearity[ west ] ) * hx_div +
                ( u_y_f * nonlinearity_center - u_y_b * this->nonlinearity[ south ] ) * hx_div +
                ( u_z_f * nonlinearity_center - u_z_b * this->nonlinearity[ down ] ) * hz_div;
         
      }

      template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const
      {
         return 7;
      }

      template< typename PreimageFunction,
                typename MeshEntity,
                typename Matrix,
                typename Vector >
      __cuda_callable__
      inline void setMatrixElements( const PreimageFunction& u,
                                     const MeshEntity& entity,
                                     const RealType& time,
                                     const RealType& tau,
                                     Matrix& matrix,
                                     Vector& b ) const
      {
         typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
         const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();
         const IndexType& center = entity.getIndex();
         const IndexType& east  = neighbourEntities.template getEntityIndex<  1,  0,  0 >();
         const IndexType& west  = neighbourEntities.template getEntityIndex< -1,  0,  0 >();
         const IndexType& north = neighbourEntities.template getEntityIndex<  0,  1,  0 >();
         const IndexType& south = neighbourEntities.template getEntityIndex<  0, -1,  0 >();
         const IndexType& up    = neighbourEntities.template getEntityIndex<  0,  0,  1 >();
         const IndexType& down  = neighbourEntities.template getEntityIndex<  0,  0, -1 >();                  
         
         
         const RealType lambda_x = tau * entity.getMesh().template getSpaceStepsProducts< -2,  0,  0 >();
         const RealType lambda_y = tau * entity.getMesh().template getSpaceStepsProducts<  0, -2,  0 >();
         const RealType lambda_z = tau * entity.getMesh().template getSpaceStepsProducts<  0,  0, -2 >();
         const RealType& nonlinearity_center = this->nonlinearity[ center ];
         const RealType& nonlinearity_west   = this->nonlinearity[ west ];
         const RealType& nonlinearity_south  = this->nonlinearity[ south ];
         const RealType& nonlinearity_down   = this->nonlinearity[ down ];
         const RealType aCoef = -lambda_z * nonlinearity_down;
         const RealType bCoef = -lambda_y * nonlinearity_south;
         const RealType cCoef = -lambda_x * nonlinearity_west;
         const RealType dCoef = lambda_x * ( nonlinearity_center + nonlinearity_west ) +
                                lambda_y * ( nonlinearity_center + nonlinearity_south ) +
                                lambda_z * ( nonlinearity_center + nonlinearity_down );
         const RealType eCoef = -lambda_x * nonlinearity_center;
         const RealType fCoef = -lambda_y * nonlinearity_center;
         const RealType gCoef = -lambda_z * nonlinearity_center;
         matrixRow.setElement( 0, down,   aCoef );
         matrixRow.setElement( 1, south,  bCoef );
         matrixRow.setElement( 2, west,   cCoef );
         matrixRow.setElement( 3, center, dCoef );
         matrixRow.setElement( 4, east,   eCoef );
         matrixRow.setElement( 5, north,  fCoef );
         matrixRow.setElement( 5, up,     gCoef );
      }
     
   public:
       
      const Nonlinearity& nonlinearity;
};

#endif	/* TNLONESIDEDNONLINEARDIFFUSION_H */
