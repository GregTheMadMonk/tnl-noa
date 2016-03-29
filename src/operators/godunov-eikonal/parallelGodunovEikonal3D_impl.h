/***************************************************************************
                          parallelGodunovEikonal3D_impl.h  -  description
                             -------------------
    begin                : Dec 1 , 2014
    copyright            : (C) 2014 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef PARALLELGODUNOVEIKONAL3D_IMPL_H_
#define PARALLELGODUNOVEIKONAL3D_IMPL_H_


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real parallelGodunovEikonalScheme< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: positivePart(const Real arg) const
{
	if(arg > 0.0)
		return arg;
	return 0.0;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real  parallelGodunovEikonalScheme< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: negativePart(const Real arg) const
{
	if(arg < 0.0)
		return -arg;
	return 0.0;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real parallelGodunovEikonalScheme< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: sign(const Real x, const Real eps) const
{
	if(x > eps)
		return 1.0;
	if (x < -eps)
		return (-1.0);

	if ( x == 0.0)
		return 0.0;

	return sin((M_PI/2.0)*(x/eps));
}




template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool parallelGodunovEikonalScheme< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index > :: init( const tnlParameterContainer& parameters )
{
	   const tnlString& meshFile = parameters.getParameter< tnlString >( "mesh" );
	   if( ! this->originalMesh.load( meshFile ) )
	   {
		   cerr << "I am not able to load the mesh from the file " << meshFile << "." << endl;
		   return false;
	   }


	   hx = originalMesh.template getSpaceStepsProducts< 1, 0, 0 >();
	   hy = originalMesh.template getSpaceStepsProducts< 0, 1, 0 >();
	   hz = originalMesh.template getSpaceStepsProducts< 0, 0, 1 >();
	   ihx = 1.0/hx;
	   ihy = 1.0/hy;
	   ihz = 1.0/hz;

	   epsilon = parameters. getParameter< double >( "epsilon" );

	   if(epsilon != 0.0)
		   epsilon *=sqrt( hx*hx + hy*hy + hz*hz );

	//   dofVector. setSize( this->mesh.getDofs() );

	   return true;

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString parallelGodunovEikonalScheme< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index > :: getType()
{
   return tnlString( "tnlLinearDiffusion< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real parallelGodunovEikonalScheme< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >:: getValue( const MeshType& mesh,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType cellIndex,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const CoordinatesType& coordinates,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Vector& u,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Real& time,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType boundaryCondition,
          	  	          	  	  	  	  	  	  	  	  	  	                     const tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 3, tnlGridEntityNoStencilStorage >,3> neighbourEntities  ) const
{
	if ( ((coordinates.x() == 0 && (boundaryCondition & 4)) or
		 (coordinates.x() == mesh.getDimensions().x() - 1 && (boundaryCondition & 2)) or
		 (coordinates.y() == 0 && (boundaryCondition & 8)) or
		 (coordinates.y() == mesh.getDimensions().y() - 1  && (boundaryCondition & 1)) or
		 (coordinates.z() == 0 && (boundaryCondition & 32)) or
		 (coordinates.z() == mesh.getDimensions().y() - 1  && (boundaryCondition & 16)))

		)
	{
		return 0.0;
	}


	//-----------------------------------

	RealType signui;
	signui = sign(u[cellIndex], this->epsilon);


	RealType xb = u[cellIndex];
	RealType xf = -u[cellIndex];
	RealType yb = u[cellIndex];
	RealType yf = -u[cellIndex];
	RealType zb = u[cellIndex];
	RealType zf = -u[cellIndex];
	RealType a,b,c,d;


	   if(coordinates.x() == mesh.getDimensions().x() - 1)
		   xf += u[neighbourEntities.template getEntityIndex< -1,  0,  0 >()];
	   else
		   xf += u[neighbourEntities.template getEntityIndex< 1,  0,  0 >()];

	   if(coordinates.x() == 0)
		   xb -= u[neighbourEntities.template getEntityIndex< 1,  0,  0 >()];
	   else
		   xb -= u[neighbourEntities.template getEntityIndex< -1,  0,  0 >()];

	   if(coordinates.y() == mesh.getDimensions().y() - 1)
		   yf += u[neighbourEntities.template getEntityIndex< 0,  -1,  0 >()];
	   else
		   yf += u[neighbourEntities.template getEntityIndex< 0,  1,  0 >()];

	   if(coordinates.y() == 0)
		   yb -= u[neighbourEntities.template getEntityIndex< 0,  1,  0 >()];
	   else
		   yb -= u[neighbourEntities.template getEntityIndex< 0,  -1,  0 >()];


	   if(coordinates.z() == mesh.getDimensions().z() - 1)
		   zf += u[neighbourEntities.template getEntityIndex< 0,  0,  -1 >()];
	   else
		   zf += u[neighbourEntities.template getEntityIndex< 0,  0,  1 >()];

	   if(coordinates.z() == 0)
		   zb -= u[neighbourEntities.template getEntityIndex< 0,  0,  1 >()];
	   else
		   zb -= u[neighbourEntities.template getEntityIndex< 0,  0,  -1 >()];

	   if(signui > 0.0)
	   {
		   xf = negativePart(xf);

		   xb = positivePart(xb);

		   yf = negativePart(yf);

		   yb = positivePart(yb);

		   zf = negativePart(zf);

		   zb = positivePart(zb);

	   }
	   else if(signui < 0.0)
	   {

		   xb = negativePart(xb);

		   xf = positivePart(xf);

		   yb = negativePart(yb);

		   yf = positivePart(yf);

		   zb = negativePart(zb);

		   zf = positivePart(zf);
	   }


	   if(xb - xf > 0.0)
		   a = xb;
	   else
		   a = xf;

	   if(yb - yf > 0.0)
		   b = yb;
	   else
		   b = yf;

	   if(zb - zf > 0.0)
		   c = zb;
	   else
		   c = zf;

	   d = ( 1.0 - sqrt(a*a + b*b + c*c)*ihx );

//	   d = 1.0 - sqrt(xf*xf + xb*xb + yf*yf + yb*yb + zf*zf + zb*zb)*ihx; /*upwind*/

	   if(Sign(d) > 0.0 )
		   return Sign(u[cellIndex])*d;
	   else
		   return signui*d;
}



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >

#ifdef HAVE_CUDA
__device__
#endif
Real parallelGodunovEikonalScheme< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >:: getValueDev( const MeshType& mesh,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType cellIndex,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const CoordinatesType& coordinates,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Real* u,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Real& time,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType boundaryCondition,
          	  	          	  	  	  	  	  	  	  	  	  	                     const tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 3, tnlGridEntityNoStencilStorage >,3> neighbourEntities
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 ) const
{
	RealType signui;
//	if(boundaryCondition == 0)
		signui = sign(u[cellIndex], this->epsilon);
//	else
//		signui = Sign(u[cellIndex]);


	RealType xb = u[cellIndex];
	RealType xf = -u[cellIndex];
	RealType yb = u[cellIndex];
	RealType yf = -u[cellIndex];
	RealType zb = u[cellIndex];
	RealType zf = -u[cellIndex];
	RealType a,b,c,d;
//	if(threadIdx.x+threadIdx.y+threadIdx.z == 0)
//		printf("x = %d, y = %d, z = %d\n",mesh.getDimensions().x() - 1,mesh.getDimensions().y() - 1,mesh.getDimensions().z() - 1);


	   if(coordinates.x() == mesh.getDimensions().x() - 1)
		   xf += u[neighbourEntities.template getEntityIndex< -1,  0,  0 >()];
	   else
		   xf += u[neighbourEntities.template getEntityIndex< 1,  0,  0 >()];

	   if(coordinates.x() == 0)
		   xb -= u[neighbourEntities.template getEntityIndex< 1,  0,  0 >()];
	   else
		   xb -= u[neighbourEntities.template getEntityIndex< -1,  0,  0 >()];

	   if(coordinates.y() == mesh.getDimensions().y() - 1)
		   yf += u[neighbourEntities.template getEntityIndex< 0,  -1,  0 >()];
	   else
		   yf += u[neighbourEntities.template getEntityIndex< 0,  1,  0 >()];

	   if(coordinates.y() == 0)
		   yb -= u[neighbourEntities.template getEntityIndex< 0,  1,  0 >()];
	   else
		   yb -= u[neighbourEntities.template getEntityIndex< 0,  -1,  0 >()];


	   if(coordinates.z() == mesh.getDimensions().z() - 1)
		   zf += u[neighbourEntities.template getEntityIndex< 0,  0,  -1 >()];
	   else
		   zf += u[neighbourEntities.template getEntityIndex< 0,  0,  1 >()];

	   if(coordinates.z() == 0)
		   zb -= u[neighbourEntities.template getEntityIndex< 0,  0,  1 >()];
	   else
		   zb -= u[neighbourEntities.template getEntityIndex< 0,  0,  -1 >()];


	   //xb *= ihx;
	   //xf *= ihx;
	  // yb *= ihy;
	   //yf *= ihy;

	   if(signui > 0.0)
	   {
		   xf = negativePart(xf);

		   xb = positivePart(xb);

		   yf = negativePart(yf);

		   yb = positivePart(yb);

		   zf = negativePart(zf);

		   zb = positivePart(zb);

	   }
	   else if(signui < 0.0)
	   {

		   xb = negativePart(xb);

		   xf = positivePart(xf);

		   yb = negativePart(yb);

		   yf = positivePart(yf);

		   zb = negativePart(zb);

		   zf = positivePart(zf);
	   }


	   if(xb - xf > 0.0)
		   a = xb;
	   else
		   a = xf;

	   if(yb - yf > 0.0)
		   b = yb;
	   else
		   b = yf;

	   if(zb - zf > 0.0)
		   c = zb;
	   else
		   c = zf;

	   d = ( 1.0 - sqrt(a*a + b*b + c*c)*ihx );

//	   d = 1.0 - sqrt(xf*xf + xb*xb + yf*yf + yb*yb + zf*zf + zb*zb)*ihx; /*upwind*/

	   if(Sign(d) > 0.0 )
		   return Sign(u[cellIndex])*d;
	   else
		   return signui*d;
}


#endif /* PARALLELGODUNOVEIKONAL3D_IMPL_H_ */
