/***************************************************************************
                          tnlNavierStokesSolverSolver.h  -  description
                             -------------------
    begin                : Oct 22, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef tnlNavierStokesSolver_H_
#define tnlNavierStokesSolver_H_

#include <core/tnlString.h>
#include <core/vectors/tnlVector.h>

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
class tnlNavierStokesSolver
{
   public:

   typedef AdvectionScheme AdvectionSchemeType;
   typedef DiffusionScheme DiffusionSchemeType;
   typedef BoundaryConditions BoundaryConditionsType;
   typedef typename AdvectionScheme::MeshType MeshType;
   typedef typename AdvectionScheme::RealType RealType;
   typedef typename AdvectionScheme::DeviceType DeviceType;
   typedef typename AdvectionScheme::IndexType IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType > VectorType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > DofVectorType;

   tnlNavierStokesSolver();

   static tnlString getTypeStatic();

   void setAdvectionScheme( AdvectionSchemeType& advection );

   void setDiffusionScheme( DiffusionSchemeType& u1Viscosity,
                            DiffusionSchemeType& u2Viscosity,
                            DiffusionSchemeType& temperatureViscosity);

   void setBoundaryConditions( BoundaryConditionsType& boundaryConditions );

   void setMesh( MeshType& mesh );

   void setMu( const RealType& mu );

   const RealType& getMu() const;

   void setR( const RealType& R );

   const RealType& getR() const;

   void setT( const RealType& T );

   const RealType& getT() const;

   void setHeatCapacityRatio( const RealType& gamma );

   const RealType& getHeatCapacityRatio() const;

   void setGravity( const RealType& gravity );

   const RealType& getGravity() const;

   VectorType& getRho();

   const VectorType& getRho() const;

   VectorType& getU1();

   const VectorType& getU1() const;

   VectorType& getU2();

   const VectorType& getU2() const;

   VectorType& getU();

   const VectorType& getU() const;

   VectorType& getPressure();

   const VectorType& getPressure() const;

   VectorType& getEnergy();

   const VectorType& getEnergy() const;


   IndexType getDofs() const;

   void bindDofVector( RealType* );

   DofVectorType& getDofVector();

   template< typename Vector >
   void updatePhysicalQuantities( const Vector& rho,
                                  const Vector& rho_u1,
                                  const Vector& rho_u2,
                                  const Vector& e );

   template< typename SolverVectorType >
   void getExplicitRhs( const RealType& time,
                        const RealType& tau,
                        SolverVectorType& u,
                        SolverVectorType& fu );

   bool writePhysicalVariables( const RealType& t,
                                const IndexType step );

   bool writeConservativeVariables( const RealType& t,
                                    const IndexType step );

   template< typename DofVector >
   bool writeExplicitRhs( const RealType& t,
                          const IndexType step,
                          DofVector& rhs );

   protected:

   RealType computeEnergy( const RealType& rho,
                           const RealType& temperature,
                           const RealType& gamma,
                           const RealType& u1,
                           const RealType& u2 ) const;

   AdvectionSchemeType* advection;

   DiffusionSchemeType  *u1Viscosity, *u2Viscosity, *energyViscosity;

   BoundaryConditionsType* boundaryConditions;

   MeshType* mesh;

   VectorType rho, u1, u2, u, p, energy;

   RealType mu, gravity, R, T, gamma;

   DofVectorType dofVector;

   VectorType rhsDofVector;

};

#include <solvers/cfd/navier-stokes/tnlNavierStokesSolver_impl.h>

#endif /* tnlNavierStokesSolver_H_ */
