/***************************************************************************
                          tnlGrid.h  -  description
                             -------------------
    begin                : Dec 12, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef TNLGRID_H_
#define TNLGRID_H_

#include <iomanip>
#include <fstream>
#include <core/tnlAssert.h>
#include <core/tnlMultiVector.h>
#include <core/tnlVector.h>

using namespace std;

template< int Dimensions, typename Real = double, typename Device = tnlHost, typename Index = int >
class tnlGrid : public tnlMultiVector< Dimensions, Real, Device, Index >
{
};

template< typename Real, typename Device, typename Index >
class tnlGrid< 1, Real, Device, Index > : public tnlMultiVector< 1, Real, Device, Index >
{
   public:

   tnlGrid();

   //! We do not allow copy constructor without object name.
   //tnlGrid( const tnlGrid< Dimensions, Real, Device, Index >& a );

   tnlGrid( const tnlString& name );

   tnlGrid( const tnlString& name,
            const tnlGrid< 1, Real, tnlHost, Index >& grid );

   tnlGrid( const tnlString& name,
            const tnlGrid< 1, Real, tnlCuda, Index >& grid );

   const tnlTuple< 1, Index >& getDimensions() const;

   //! Sets the dimensions
   /***
    * This method also must recompute space steps. It is save to call setDimensions and
    * setDomain in any order. Both recompute the space steps.
    */
   bool setDimensions( const tnlTuple< 1, Index >& dimensions );

   //! Sets the computation domain in form of "rectangle".
   /***
    * This method also must recompute space steps. It is save to call setDimensions and
    * setDomain in any order. Both recompute the space steps.
    */

   bool setDomain( const tnlTuple< 1, Real >& lowerCorner,
                   const tnlTuple< 1, Real >& upperCorner );

   template< typename Grid >
   bool setLike( const Grid& v );

   const tnlTuple< 1, Real >& getDomainLowerCorner() const;

   const tnlTuple< 1, Real >& getDomainUpperCorner() const;

   const tnlTuple< 1, Real >& getSpaceSteps() const;

   tnlString getType() const;

   bool operator == ( const tnlGrid< 1, Real, Device, Index >& array ) const;

   bool operator != ( const tnlGrid< 1, Real, Device, Index >& array ) const;

   template< typename Real2, typename Device2, typename Index2 >
   tnlGrid< 1, Real, Device, Index >& operator = ( const tnlGrid< 1, Real2, Device2, Index2 >& array );

   //! This method interpolates value at given point.
   Real getValue( const tnlTuple< 1, Real >& point ) const;

   //! Interpolation for 1D grid.
   Real getValue( const Real& x ) const;

   //! Forward difference w.r.t x
   Real Partial_x_f( const Index i1 ) const;

   //! Backward difference w.r.t x
   Real Partial_x_b( const Index i1 ) const;

   //! Central difference w.r.t. x
   Real Partial_x( const Index i1 ) const;

   //! Second order difference w.r.t. x
   Real Partial_xx( const Index i1 ) const;

   //! Set space dependent Dirichlet boundary conditions
   void setDirichletBC( const tnlGrid< 1, Real, Device, Index >&bc,
                        const tnlTuple< 1, bool >& lowerBC,
                        const tnlTuple< 1, bool >& upperBC );

   //! Set constant Dirichlet boundary conditions
   void setDirichletBC( const Real& bc,
                        const tnlTuple< 1, bool >& lowerBC,
                        const tnlTuple< 1, bool >& upperBC );

   //! Set space dependent Neumann boundary conditions
   void setNeumannBC( const tnlGrid< 1, Real, Device, Index >&bc,
                      const tnlTuple< 1, bool >& lowerBC,
                      const tnlTuple< 1, bool >& upperBC );

   //! Set constant Neumann boundary conditions
   void setNeumannBC( const Real& bc,
                      const tnlTuple< 1, bool >& lowerBC,
                      const tnlTuple< 1, bool >& upperBC );

   Real getLpNorm() const;

   Real getDifferenceLpNorm( const tnlVector< Real, Device, Index >& v, const Real& p ) const;

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   //! This method writes the grid in some format suitable for some other preprocessing.
   /*! Possible formats are:
    *  1) Gnuplot format (gnuplot)
    *  2) VTI format (vti)
    *  3) Povray format (povray) - only for 3D.
    */
   bool draw( const tnlString& fileName,
              const tnlString& format,
              const tnlTuple< 1, Index > steps = ( tnlTuple< 1, Index > ) 1 ) const;

   protected:
   tnlTuple< 1, Real > domainLowerCorner, domainUpperCorner, spaceSteps;
};

template< typename Real, typename Device, typename Index >
class tnlGrid< 2, Real, Device, Index > : public tnlMultiVector< 2, Real, Device, Index >
{
   public:

   tnlGrid();

   //! We do not allow copy constructor without object name.
   //tnlGrid( const tnlGrid< Dimensions, Real, Device, Index >& a );

   tnlGrid( const tnlString& name );

   tnlGrid( const tnlString& name,
            const tnlGrid< 2, Real, tnlHost, Index >& grid );

   tnlGrid( const tnlString& name,
            const tnlGrid< 2, Real, tnlCuda, Index >& grid );

   const tnlTuple< 2, Index >& getDimensions() const;

   //! Sets the dimensions
   /***
    * This method also must recompute space steps. It is save to call setDimensions and
    * setDomain in any order. Both recompute the space steps.
    */
   bool setDimensions( const tnlTuple< 2, Index >& dimensions );

   //! Sets the computation domain in form of "rectangle".
   /***
    * This method also must recompute space steps. It is save to call setDimensions and
    * setDomain in any order. Both recompute the space steps.
    */

   bool setDomain( const tnlTuple< 2, Real >& lowerCorner,
                   const tnlTuple< 2, Real >& upperCorner );

   template< typename Grid >
   bool setLike( const Grid& v );

   const tnlTuple< 2, Real >& getDomainLowerCorner() const;

   const tnlTuple< 2, Real >& getDomainUpperCorner() const;

   const tnlTuple< 2, Real >& getSpaceSteps() const;

   tnlString getType() const;

   bool operator == ( const tnlGrid< 2, Real, Device, Index >& array ) const;

   bool operator != ( const tnlGrid< 2, Real, Device, Index >& array ) const;

   template< typename Real2, typename Device2, typename Index2 >
   tnlGrid< 2, Real, Device, Index >& operator = ( const tnlGrid< 2, Real2, Device2, Index2 >& array );

   //! This method interpolates value at given point.
   Real getValue( const tnlTuple< 2, Real >& point ) const;

   //! Interpolation for 2D grid.
   Real getValue( const Real& x,
                  const Real& y ) const;

   //! Forward difference w.r.t x in two dimensions
   Real Partial_x_f( const Index i1,
                     const Index i2 ) const;

   //! Backward difference w.r.t x in two dimensions
   Real Partial_x_b( const Index i1,
                     const Index i2 ) const;

   //! Central difference w.r.t. x in two dimensions
   Real Partial_x( const Index i1,
                   const Index i2 ) const;

   //! Second order difference w.r.t. x in two dimensions
   Real Partial_xx( const Index i1,
                    const Index i2 ) const;

   //! Forward difference w.r.t y
   Real Partial_y_f( const Index i1,
                     const Index i2 ) const;

   //! Backward difference w.r.t y
   Real Partial_y_b( const Index i1,
                     const Index i2 ) const;

   //! Central difference w.r.t y
   Real Partial_y( const Index i1,
                   const Index i2 ) const;

   //! Second order difference w.r.t. y
   Real Partial_yy( const Index i1,
                    const Index i2 ) const;

   //! Set space dependent Dirichlet boundary conditions
   void setDirichletBC( const tnlGrid< 2, Real, Device, Index >&bc,
                        const tnlTuple< 2, bool >& lowerBC,
                        const tnlTuple< 2, bool >& upperBC );

   //! Set constant Dirichlet boundary conditions
   void setDirichletBC( const Real& bc,
                        const tnlTuple< 2, bool >& lowerBC,
                        const tnlTuple< 2, bool >& upperBC );

   //! Set space dependent Neumann boundary conditions
   void setNeumannBC( const tnlGrid< 2, Real, Device, Index >&bc,
                      const tnlTuple< 2, bool >& lowerBC,
                      const tnlTuple< 2, bool >& upperBC );

   //! Set constant Neumann boundary conditions
   void setNeumannBC( const Real& bc,
                      const tnlTuple< 2, bool >& lowerBC,
                      const tnlTuple< 2, bool >& upperBC );

   Real getLpNorm() const;

   Real getDifferenceLpNorm( const tnlVector< Real, Device, Index >& v, const Real& p ) const;

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   //! This method writes the grid in some format suitable for some other preprocessing.
   /*! Possible formats are:
    *  1) Gnuplot format (gnuplot)
    *  2) VTI format (vti)
    *  3) Povray format (povray) - only for 3D.
    */
   bool draw( const tnlString& fileName,
              const tnlString& format,
              const tnlTuple< 2, Index > steps = ( tnlTuple< 2, Index > ) 1 ) const;

   protected:
   tnlTuple< 2, Real > domainLowerCorner, domainUpperCorner, spaceSteps;
};

template< typename Real, typename Device, typename Index >
class tnlGrid< 3, Real, Device, Index > : public tnlMultiVector< 3, Real, Device, Index >
{
   public:

   tnlGrid();

   //! We do not allow copy constructor without object name.
   //tnlGrid( const tnlGrid< Dimensions, Real, Device, Index >& a );

   tnlGrid( const tnlString& name );

   tnlGrid( const tnlString& name,
            const tnlGrid< 3, Real, tnlHost, Index >& grid );

   tnlGrid( const tnlString& name,
            const tnlGrid< 3, Real, tnlCuda, Index >& grid );

   const tnlTuple< 3, Index >& getDimensions() const;

   //! Sets the dimensions
   /***
    * This method also must recompute space steps. It is save to call setDimensions and
    * setDomain in any order. Both recompute the space steps.
    */
   bool setDimensions( const tnlTuple< 3, Index >& dimensions );

   //! Sets the computation domain in form of "rectangle".
   /***
    * This method also must recompute space steps. It is save to call setDimensions and
    * setDomain in any order. Both recompute the space steps.
    */

   bool setDomain( const tnlTuple< 3, Real >& lowerCorner,
                   const tnlTuple< 3, Real >& upperCorner );

   template< typename Grid >
   bool setLike( const Grid& v );

   const tnlTuple< 3, Real >& getDomainLowerCorner() const;

   const tnlTuple< 3, Real >& getDomainUpperCorner() const;

   const tnlTuple< 3, Real >& getSpaceSteps() const;

   tnlString getType() const;

   bool operator == ( const tnlGrid< 3, Real, Device, Index >& array ) const;

   bool operator != ( const tnlGrid< 3, Real, Device, Index >& array ) const;

   template< typename Real2, typename Device2, typename Index2 >
   tnlGrid< 3, Real, Device, Index >& operator = ( const tnlGrid< 3, Real2, Device2, Index2 >& array );

   //! This method interpolates value at given point.
   Real getValue( const tnlTuple< 3, Real >& point ) const;

   //! Interpolation for 3D grid.
   Real getValue( const Real& x,
                  const Real& y,
                  const Real& z ) const;

   //! Forward difference w.r.t x in three dimensions
   Real Partial_x_f( const Index i1,
                     const Index i2,
                     const Index i3 ) const;

   //! Backward difference w.r.t x in three dimensions
   Real Partial_x_b( const Index i1,
                     const Index i2,
                     const Index i3 ) const;

   //! Central difference w.r.t. x
   Real Partial_x( const Index i1,
                   const Index i2,
                   const Index i3 ) const;

   //! Second order difference w.r.t. x
   Real Partial_xx( const Index i1,
                    const Index i2,
                    const Index i3 ) const;

   //! Forward difference w.r.t y in three dimensions
   Real Partial_y_f( const Index i1,
                     const Index i2,
                     const Index i3 ) const;

   //! Backward difference w.r.t y in three dimensions
   Real Partial_y_b( const Index i1,
                     const Index i2,
                     const Index i3 ) const;

   //! Central difference w.r.t y
   Real Partial_y( const Index i1,
                   const Index i2,
                   const Index i3 ) const;

   //! Second order difference w.r.t. y in three dimensions
   Real Partial_yy( const Index i1,
                    const Index i2,
                    const Index i3 ) const;

   //! Forward difference w.r.t z
   Real Partial_z_f( const Index i1,
                     const Index i2,
                     const Index i3 ) const;

   //! Backward difference w.r.t z
   Real Partial_z_b( const Index i1,
                     const Index i2,
                     const Index i3 ) const;

   //! Central difference w.r.t z
   Real Partial_z( const Index i1,
                   const Index i2,
                   const Index i3 ) const;

   //! Second order difference w.r.t. z
   Real Partial_zz( const Index i1,
                    const Index i2,
                    const Index i3 ) const;

   //! Set space dependent Dirichlet boundary conditions
   void setDirichletBC( const tnlGrid< 3, Real, Device, Index >&bc,
                        const tnlTuple< 3, bool >& lowerBC,
                        const tnlTuple< 3, bool >& upperBC );

   //! Set constant Dirichlet boundary conditions
   void setDirichletBC( const Real& bc,
                        const tnlTuple< 3, bool >& lowerBC,
                        const tnlTuple< 3, bool >& upperBC );

   //! Set space dependent Neumann boundary conditions
   void setNeumannBC( const tnlGrid< 3, Real, Device, Index >&bc,
                      const tnlTuple< 3, bool >& lowerBC,
                      const tnlTuple< 3, bool >& upperBC );

   //! Set constant Neumann boundary conditions
   void setNeumannBC( const Real& bc,
                      const tnlTuple< 3, bool >& lowerBC,
                      const tnlTuple< 3, bool >& upperBC );

   Real getLpNorm() const;

   Real getDifferenceLpNorm( const tnlVector< Real, Device, Index >& v, const Real& p ) const;

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   //! This method writes the grid in some format suitable for some other preprocessing.
   /*! Possible formats are:
    *  1) Gnuplot format (gnuplot)
    *  2) VTI format (vti)
    *  3) Povray format (povray) - only for 3D.
    */
   bool draw( const tnlString& fileName,
              const tnlString& format,
              const tnlTuple< 3, Index > steps = ( tnlTuple< 3, Index > ) 1 ) const;

   protected:
   tnlTuple< 3, Real > domainLowerCorner, domainUpperCorner, spaceSteps;
};

#include <mesh/implementation/tnlGrid1D_impl.h>
#include <mesh/implementation/tnlGrid2D_impl.h>
#include <mesh/implementation/tnlGrid3D_impl.h>

#endif /* TNLGRID_H_ */


#ifdef UNDEF
   Real getMax() const;

   Real getMin() const;

   Real getAbsMax() const;

   Real getAbsMin() const;

   Real getLpNorm() const;

   Real getSum() const;

   Real getDifferenceMax( const tnlVector< Real, Device, Index >& v ) const;

   Real getDifferenceMin( const tnlVector< Real, Device, Index >& v ) const;

   Real getDifferenceAbsMax( const tnlVector< Real, Device, Index >& v ) const;

   Real getDifferenceAbsMin( const tnlVector< Real, Device, Index >& v ) const;

   Real getDifferenceLpNorm( const tnlVector< Real, Device, Index >& v, const Real& p ) const;

   Real getDifferenceSum( const tnlVector< Real, Device, Index >& v ) const;

   void scalarMultiplication( const Real& alpha );

   //! Compute scalar dot product
   Real sdot( const tnlVector< Real, Device, Index >& v ) const;

   //! Compute SAXPY operation (Scalar Alpha X Pus Y ).
   void saxpy( const Real& alpha,
                const tnlVector< Real, Device, Index >& x );

   //! Compute SAXMY operation (Scalar Alpha X Minus Y ).
   /*!**
    * It is not a standart BLAS function but is useful for GMRES solver.
    */
   void saxmy( const Real& alpha,
                const tnlVector< Real, Device, Index >& x );
#endif
