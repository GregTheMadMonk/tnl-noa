/***************************************************************************
                          tnlGMRESSolver.h  -  description
                             -------------------
    begin                : 2007/07/31
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef tnlGMRESSolverH
#define tnlGMRESSolverH


#include <math.h>
#include <core/tnlObject.h>
#include <core/tnlVector.h>
#include <core/tnlSharedVector.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/tnlIterativeSolver.h>

template< typename Matrix,
           typename Preconditioner = tnlDummyPreconditioner< typename Matrix :: RealType,
                                                                typename Matrix :: Device,
                                                                typename Matrix :: IndexType> >
class tnlGMRESSolver : public tnlObject,
                        public tnlIterativeSolver< typename Matrix :: RealType,
                                                    typename Matrix :: IndexType >
{

   typedef typename Matrix :: RealType RealType;
   typedef typename Matrix :: IndexType IndexType;
   typedef typename Matrix :: Device Device;
   typedef Matrix MatrixType;
   typedef Preconditioner PreconditionerType;

   public:

   tnlGMRESSolver();

   tnlString getType() const;

   void setRestarting( IndexType rest );

   void setMatrix( const MatrixType& matrix );

   void setPreconditioner( const Preconditioner& preconditioner );

   template< typename Vector >
   bool solve( const Vector& b, Vector& x );

   ~tnlGMRESSolver();

   protected:

   void update( IndexType k,
                 IndexType m,
                 const tnlVector< RealType, tnlHost, IndexType >& H,
                 const tnlVector< RealType, tnlHost, IndexType >& s,
                 tnlVector< RealType, Device, IndexType >& v,
                 tnlVector< RealType, Device, IndexType >& x );

   void generatePlaneRotation( RealType &dx,
                                  RealType &dy,
                                  RealType &cs,
                                  RealType &sn );

   void applyPlaneRotation( RealType &dx,
                               RealType &dy,
                               RealType &cs,
                               RealType &sn );


   bool setSize( IndexType _size, IndexType m );

   tnlVector< RealType, Device, IndexType > _r, _w, _v, _M_tmp;
   tnlVector< RealType, tnlHost, IndexType > _s, _cs, _sn, _H;

   IndexType size, restarting, maxIterations;

   RealType maxResidue;

   const MatrixType* matrix;
   const PreconditionerType* preconditioner;
};

#include <solvers/linear/krylov/implementation/tnlGMRESSolver_impl.h>

#endif
