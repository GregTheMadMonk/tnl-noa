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

#define DIMENSION 2
//#define OUTPUT 


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
  
  typedef Grid<DIMENSION,double,Host,int> MeshType;
  typedef MeshFunction<MeshType> MeshFunctionType;
  typedef Vector<double,Host,int> DofType;
  typedef typename MeshType::Cell Cell;
  typedef typename MeshType::IndexType IndexType; 
  typedef typename MeshType::PointType PointType; 
  
  typedef DistributedGrid<MeshType> DistributedGridType;
  
  typedef LinearFunction<double,DIMENSION> LinearFunctionType;
  typedef ConstFunction<double,DIMENSION> ConstFunctionType;
  
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
 globalOrigin.setValue(-0.5);
 
 PointType globalProportions;
 globalProportions.setValue(size);
 
 
 MeshType globalGrid;
 globalGrid.setDimensions(globalProportions);
 globalGrid.setDomain(globalOrigin,globalProportions);
 
 
 int distr[DIMENSION];
 for(int i=0;i<DIMENSION;i++) 
	distr[i]=0;
 DistributedGridType distrgrid(globalGrid, distr); 
   
 SharedPointer<MeshType> gridptr;
 SharedPointer<MeshFunctionType> meshFunctionptr;
 MeshFunctionEvaluator< MeshFunctionType, LinearFunctionType > linearFunctionEvaluator;
 MeshFunctionEvaluator< MeshFunctionType, ConstFunctionType > constFunctionEvaluator;
 
  distrgrid.SetupGrid(*gridptr);
  
  DofType dof(gridptr->template getEntitiesCount< Cell >());

  dof.setValue(0);
  
  meshFunctionptr->bind(gridptr,dof);  
  
  SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
  SharedPointer< ConstFunctionType, Host > constFunctionPtr; 
   
  DistributedGridSynchronizer<DistributedGridType,MeshFunctionType> synchronizer(&distrgrid);
  
  setup.stop();
  
  double sum=0.0;

  constFunctionPtr->Number=MPI::COMM_WORLD.Get_rank();
  
  for(int i=0;i<cycles;i++)
	{    
	    eval.start();
		
		//constFunctionEvaluator.evaluateBoundaryEntities( meshFunctionptr , constFunctionPtr );
		linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);
		MPI::COMM_WORLD.Barrier();
		eval.stop();

		sync.start();	
		synchronizer.Synchronize(*meshFunctionptr);
		MPI::COMM_WORLD.Barrier();
		sync.stop();

		sum+=dof[gridptr->getDimensions().x()/2]; //dummy acces to array	
	}
  all.stop();
  
#ifdef OUTPUT	
  //print local dof
  Printer<MeshType,DofType>::print_dof(MPI::COMM_WORLD.Get_rank(),*gridptr, dof);
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
