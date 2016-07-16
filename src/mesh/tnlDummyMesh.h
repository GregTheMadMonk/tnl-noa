/***************************************************************************
                          tnlDummyMesh.h  -  description
                             -------------------
    begin                : Apr 17, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLDUMMYMESH_H_
#define TNLDUMMYMESH_H_

template< typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlDummyMesh
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlDummyMesh< Real, Device, Index > ThisType;
 
   static const int meshDimensions = 1;
 
   constexpr static int getMeshDimensions() { return meshDimensions; }
 
 
   const Real& getParametricStep(){ return 0.0; }
 
   tnlString getSerializationType() const { return tnlString( "tnlDummyMesh" ); }

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const { return 0.0; }

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const { return 0.0; }

   bool save( tnlFile& file ) const { return true; }

   //! Method for restoring the object from a file
   bool load( tnlFile& file ) { return true; }

   bool save( const tnlString& fileName ) const { return true; }

   bool load( const tnlString& fileName ) { return true; }

   bool writeMesh( const tnlString& fileName,
                   const tnlString& format ) const { return true; }


   template< typename MeshFunction >
   bool write( const MeshFunction& function,
                const tnlString& fileName,
                const tnlString& format ) const { return true; }
};


#endif /* TNLDUMMYMESH_H_ */
