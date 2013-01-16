/***************************************************************************
                          tnlExplicitTimeStepper.h  -  description
                             -------------------
    begin                : Jan 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#include <core/tnlObject.h>
#include <core/tnlHost.h>
#include <core/tnlTuple.h>

template< int Dimensions,
          typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlGrid : public tnlObject
{
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGrid< 1, Real, Device, Index> : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   enum { Dimensions = 1};

   tnlGrid();

   static tnlString getTypeStatic();

   void setDimensions( const Index xSize );

   const tnlTuple< 1, Index >& getDimensions() const;

   void setLowerCorner( const tnlTuple< 1, Real >& lowerCorner );

   const tnlTuple< 1, Real >& getLowerCorner() const;

   void setUpperCorner( const tnlTuple< 1, Real >& upperCorner );

   const tnlTuple< 1, Real >& getUpperCorner() const;

   void setSpaceStep( const tnlTuple< 1, Real >& spaceStep );

   tnlTuple< 1, Real > getSpaceStep() const;

   Index getNodeIndex( const Index i ) const;

   Index getDofs() const;

   protected:

   tnlTuple< 1, IndexType > dimensions;

   tnlTuple< 1, RealType > lowerCorner, upperCorner;

   IndexType dofs;

};

template< typename Real,
          typename Device,
          typename Index >
class tnlGrid< 2, Real, Device, Index> : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   enum { Dimensions = 2};

   tnlGrid();

   static tnlString getTypeStatic();

   void setDimensions( const Index ySize, const Index xSize );

   const tnlTuple< 2, Index >& getDimensions() const;

   void setLowerCorner( const tnlTuple< 2, Real >& lowerCorner );

   const tnlTuple< 2, Real >& getLowerCorner() const;

   void setUpperCorner( const tnlTuple< 2, Real >& upperCorner );

   const tnlTuple< 2, Real >& getUpperCorner() const;

   void setSpaceStep( const tnlTuple< 2, Real >& spaceStep );

   tnlTuple< 2, Real > getSpaceStep() const;

   Index getNodeIndex( const Index j, const Index i ) const;

   Index getDofs() const;

   protected:

   tnlTuple< 2, IndexType > dimensions;

   tnlTuple< 2, RealType > lowerCorner, upperCorner;

   IndexType dofs;

};

template< typename Real,
          typename Device,
          typename Index >
class tnlGrid< 3, Real, Device, Index> : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   enum { Dimensions = 3};

   tnlGrid();

   static tnlString getTypeStatic();

   void setDimensions( const Index zSize, const Index ySize, const Index xSize );

   const tnlTuple< 3, Index >& getDimensions() const;

   void setLowerCorner( const tnlTuple< 3, Real >& lowerCorner );

   const tnlTuple< 3, Real >& getLowerCorner() const;

   void setUpperCorner( const tnlTuple< 3, Real >& upperCorner );

   const tnlTuple< 3, Real >& getUpperCorner() const;

   void setSpaceStep( const tnlTuple< 3, Real >& spaceStep );

   tnlTuple< 3, Real > getSpaceStep() const;

   Index getNodeIndex( const Index k, const Index j, const Index i ) const;

   Index getDofs() const;

   protected:

   tnlTuple< 3, IndexType > dimensions;

   tnlTuple< 3, RealType > lowerCorner, upperCorner;

   IndexType dofs;

};

#include <implementation/mesh/tnlGrid1D_impl.h>
#include <implementation/mesh/tnlGrid2D_impl.h>
#include <implementation/mesh/tnlGrid3D_impl.h>


#endif /* TNLGRID_H_ */
