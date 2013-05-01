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
#include <mesh/tnlIdenticalGridGeometry.h>

template< int Dimensions,
          typename Real = double,
          typename Device = tnlHost,
          typename Index = int,
          template< int, typename, typename, typename > class Geometry = tnlIdenticalGridGeometry >
class tnlGrid : public tnlObject
{
};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
class tnlGrid< 1, Real, Device, Index, Geometry > : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   enum { Dimensions = 1};

   tnlGrid();

   static tnlString getTypeStatic();

   tnlString getType() const;

   void setDimensions( const Index xSize );

   void setDimensions( const tnlTuple< 1, Index >& );

   const tnlTuple< 1, Index >& getDimensions() const;

   void setOrigin( const tnlTuple< 1, Real >& origin );

   const tnlTuple< 1, Real >& getOrigin() const;

   void setProportions( const tnlTuple< 1, Real >& proportions );

   const tnlTuple< 1, Real >& getProportions() const;

   void setSpaceStep( const tnlTuple< 1, Real >& spaceStep );

   tnlTuple< 1, Real > getSpaceStep() const;

   Index getElementIndex( const Index i ) const;

   Index getDofs() const;

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   template< typename MeshFunction >
   bool write( const MeshFunction& function,
                const tnlString& fileName,
                const tnlString& format ) const;

   protected:

   tnlTuple< 1, IndexType > dimensions;

   tnlTuple< 1, RealType > origin, proportions;

   IndexType dofs;

};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
class tnlGrid< 2, Real, Device, Index, Geometry > : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Geometry GeometryType;
   enum { Dimensions = 2};

   tnlGrid();

   static tnlString getTypeStatic();

   tnlString getType() const;

   void setDimensions( const Index ySize, const Index xSize );

   void setDimensions( const tnlTuple< 2, Index >& );

   const tnlTuple< 2, Index >& getDimensions() const;

   void setOrigin( const tnlTuple< 2, Real >& origin );

   const tnlTuple< 2, Real >& getOrigin() const;

   void setProportions( const tnlTuple< 2, Real >& proportions );

   const tnlTuple< 2, Real >& getProportions() const;

   void setParametricStep( const tnlTuple< 2, Real >& spaceStep );

   tnlTuple< 2, Real > getParametricStep() const;

   Index getElementIndex( const Index j,
                          const Index i ) const;

   Index getElementNeighbour( const Index Element,
                              const Index dy,
                              const Index dx ) const;

   Index getDofs() const;

   Real getElementMeasure( const Index j,
                           const Index i ) const;

   Real getElementsDistance( const Index j,
                             const Index i,
                             const Index dy,
                             const Index dx ) const;

   template< int dy, int dx >
   Real getEdgeLength( const Index j,
                       const Index i ) const;

   template< int dy, int dx >
   tnlTuple< 2, Real > getEdgeNormal( const Index j,
                                      const Index i ) const;


   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   template< typename MeshFunction >
   bool write( const MeshFunction& function,
               const tnlString& fileName,
               const tnlString& format ) const;

   protected:

   tnlTuple< 2, IndexType > dimensions;

   tnlTuple< 2, RealType > origin, proportions;

   GeometryType geometry;

   IndexType dofs;

};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
class tnlGrid< 3, Real, Device, Index, Geometry > : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   enum { Dimensions = 3};

   tnlGrid();

   static tnlString getTypeStatic();

   tnlString getType() const;

   void setDimensions( const Index zSize, const Index ySize, const Index xSize );

   void setDimensions( const tnlTuple< 3, Index >& );

   const tnlTuple< 3, Index >& getDimensions() const;

   void setOrigin( const tnlTuple< 3, Real >& origin );

   const tnlTuple< 3, Real >& getOrigin() const;

   void setProportions( const tnlTuple< 3, Real >& proportions );

   const tnlTuple< 3, Real >& getProportions() const;

   void setSpaceStep( const tnlTuple< 3, Real >& spaceStep );

   tnlTuple< 3, Real > getSpaceStep() const;

   Index getElementIndex( const Index k, const Index j, const Index i ) const;

   Index getDofs() const;

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   template< typename MeshFunction >
   bool write( const MeshFunction& function,
                const tnlString& fileName,
                const tnlString& format ) const;

   protected:

   tnlTuple< 3, IndexType > dimensions;

   tnlTuple< 3, RealType > origin, proportions;

   IndexType dofs;

};

#include <implementation/mesh/tnlGrid1D_impl.h>
#include <implementation/mesh/tnlGrid2D_impl.h>
#include <implementation/mesh/tnlGrid3D_impl.h>


#endif /* TNLGRID_H_ */
