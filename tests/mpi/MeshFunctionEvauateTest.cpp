/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <iostream>
#include <sstream>

using namespace std;


#ifdef HAVE_MPI

#define USE_MPI
#include <TNL/mpi-supp.h>


#include <TNL/Containers/Array.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedGrid.h>
#include <TNL/Meshes/DistributedGridSynchronizer.h>
#include <TNL/Functions/MeshFunction.h>

#include <TNL/Timer.h>

#define OUTPUT 

#include "../../src/UnitTests/Mpi/Functions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Functions;
using namespace TNL::Devices;
#endif

int main ( int argc, char *argv[])
{
	
#ifdef USE_MPI
  Timer all,setup,eval,sync;
	
  MPI::Init(argc,argv);
  
  //typedef Grid<1,double,Host,int> MeshType;
  //typedef Grid<2,double,Host,int> MeshType;
  typedef Grid<3,double,Host,int> MeshType;
  typedef MeshFunction<MeshType> MeshFunctionType;
  typedef Vector<double,Host,int> DofType;
  typedef typename MeshType::Cell Cell;
  typedef typename MeshType::IndexType IndexType; 
  typedef typename MeshType::PointType PointType; 
  
  int size=9;
  int cycles=1;
  if(argc==3)
  {
	  size=strtol(argv[1],NULL,10);
	  cycles=strtol(argv[2],NULL,10);
	  //cout << "size: "<< size <<"cycles: "<< cycles <<endl;
  }
  
  	all.start();
	setup.start();
  
 PointType globalOrigin;
 globalOrigin.x()=-0.5;
 globalOrigin.y()=-0.5;
 globalOrigin.z()=-0.5;
 
 PointType globalProportions;
 globalProportions.x()=size;
 globalProportions.y()=size;
 globalProportions.z()=size;
 
 
 MeshType globalGrid;
 //globalGrid.setDimensions(9);
 //globalGrid.setDimensions(size,size);
 globalGrid.setDimensions(size,size,size);
 globalGrid.setDomain(globalOrigin,globalProportions);
 
 int distr[3]={0,0,0}; 
 DistributedGrid<MeshType> distrgrid(globalGrid, distr); 
 
 //DistributedGrid<MeshType> distrgrid(globalGrid);
  
 SharedPointer<MeshType> gridptr;
 SharedPointer<MeshFunctionType> meshFunctionptr;
 MeshFunctionEvaluator< MeshFunctionType, LinearFunction<double,3> > linearFunctionEvaluator;
 MeshFunctionEvaluator< MeshFunctionType, ConstFunction<double,3> > constFunctionEvaluator;
 
  distrgrid.SetupGrid(*gridptr);
  
  DofType dof(gridptr->template getEntitiesCount< Cell >());

  int maxx=gridptr->getDimensions().x();
  int maxy=gridptr->getDimensions().y();
  int maxz=gridptr->getDimensions().z();
  for(int k=0;k<maxz;k++)
	for(int j=0;j<maxy;j++)
		for(int i=0;i<maxx;i++)
			dof[k*maxx*maxy+maxx*j+i]=0;
  
  meshFunctionptr->bind(gridptr,dof);  
  
  SharedPointer< LinearFunction<double,3>, Host > linearFunctionPtr;
  SharedPointer< ConstFunction<double,3>, Host > constFunctionPtr; 
   
  DistributedGridSynchronizer<DistributedGrid<MeshType>,MeshFunctionType,3> synchronizer(&distrgrid);
  
  setup.stop();
  
  double sum=0.0;

  for(int i=0;i<cycles;i++)
	{
	    //zero->Number=MPI::COMM_WORLD.Get_rank();
	    
	    eval.start();
		/*zero->Number=-1;
		zeroevaluator.evaluateBoundaryEntities( meshFunctionptr , zero );*/
		constFunctionPtr->Number=MPI::COMM_WORLD.Get_rank();
		constFunctionEvaluator.evaluateAllEntities( meshFunctionptr , constFunctionPtr );
		//zero->Number=-1;
		/*zero->Number=MPI::COMM_WORLD.Get_rank();
		zeroevaluator.evaluateBoundaryEntities( meshFunctionptr , zero );*/
		MPI::COMM_WORLD.Barrier();
		eval.stop();


		sync.start();	
		synchronizer.Synchronize(*meshFunctionptr);
		MPI::COMM_WORLD.Barrier();
		sync.stop();

		sum+=dof[2*gridptr->getDimensions().y()]; //dummy acces to array	
	}
  all.stop();
  
#ifdef OUTPUT	
  //print local dof
  maxx=gridptr->getDimensions().x();
  maxy=gridptr->getDimensions().y();
  maxz=gridptr->getDimensions().z();
  
  stringstream sout;
  distrgrid.printcoords(sout);
  for(int k=0;k<maxz;k++)
  {
	for(int j=0;j<maxy;j++)
	{
		for(int ii=0;ii<k;ii++)
			sout<<"  ";
		for(int i=0;i<maxx;i++)
		{
			sout <<dof[k*maxx*maxy+maxx*j+i]<<"  ";
		}
		sout << endl;
	}
  }
  cout << sout.str()<< endl<<endl;
#endif
  
  
  
  
  
  if(MPI::COMM_WORLD.Get_rank()==0)
  {
	cout << sum <<endl<<endl;  
	  
	cout<<"setup: "<<setup.getRealTime() <<endl;
	cout<<"evalpercycle: "<<eval.getRealTime()/cycles<<endl;
	cout<<"syncpercycle: "<<sync.getRealTime()/cycles<<endl;
	cout <<"eval: "<<eval.getRealTime()<<endl;
	cout <<"sync: "<<sync.getRealTime()<<endl;
	cout<<"all: "<<all.getRealTime()<<endl<<endl;
  }
  

  MPI::Finalize();
#else
  std::cout<<"MPI not Supported." << std::endl;
#endif
  return 0;

}
