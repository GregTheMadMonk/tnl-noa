/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <iostream>
#define HAVE_MPI
#include <TNL/mpi-supp.h>

using namespace std;

#include <TNL/Containers/Array.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedGrid.h>
#include <TNL/Meshes/DistributedGridSynchronizer.h>
#include <TNL/Functions/MeshFunction.h>

#include <TNL/Timer.h>

#define OUTPUT 

#include "Functions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Functions;
using namespace TNL::Devices;

int main ( int argc, char *argv[])
{
	Timer all,setup,eval,sync;

#ifdef OUTPUT
  cout << "MeshFunction Evaluate Test for MPI develop by hanouvit" << endl;
#endif
  
#ifdef HAVE_MPI
  MPI::Init(argc,argv);
  
  //typedef Grid<1,double,Host,int> MeshType;
  typedef Grid<2,double,Host,int> MeshType;
  typedef MeshFunction<MeshType> MeshFunctionType;
  typedef Vector<double,Host,int> DofType;
  typedef typename MeshType::Cell Cell;
  typedef typename MeshType::IndexType IndexType; 
  typedef typename MeshType::PointType PointType; 
  
  int size=10;
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
 
 PointType globalProportions;
 globalProportions.x()=size;
 globalProportions.y()=size;
 
 
 MeshType globalGrid;
 //globalGrid.setDimensions(9);
 globalGrid.setDimensions(size,size);
 globalGrid.setDomain(globalOrigin,globalProportions);
 
 DistributedGrid<MeshType> distrgrid(globalGrid); 
  
 SharedPointer<MeshType> gridptr;
 SharedPointer<MeshFunctionType> meshFunctionptr;
 MeshFunctionEvaluator< MeshFunctionType, FunctionToEvaluate<double,2> > evaluator;
 MeshFunctionEvaluator< MeshFunctionType, ZeroFunction<double,2> > zeroevaluator;
 
  distrgrid.SetupGrid(*gridptr);
  
  DofType dof(gridptr->template getEntitiesCount< Cell >());

  meshFunctionptr->bind(gridptr,dof);  
  
  SharedPointer< FunctionToEvaluate<double,2>, Host > functionToEvaluate;
  SharedPointer< ZeroFunction<double,2>, Host > zero; 
   
  setup.stop();
  
  double sum=0.0;

  for(int i=0;i<cycles;i++)
	{
	    //zero->Number=MPI::COMM_WORLD.Get_rank();
	    zero->Number=i;
	    eval.start();
		zeroevaluator.evaluateInteriorEntities( meshFunctionptr , zero );
		//evaluator.evaluateAllEntities( meshFunctionptr , functionToEvaluate );
		zero->Number=-1;
		zeroevaluator.evaluateBoundaryEntities( meshFunctionptr , zero );
		MPI::COMM_WORLD.Barrier();
		eval.stop();


		sync.start();	
		DistributedGridSynchronizer<DistributedGrid<MeshType>,MeshFunctionType,2>::Synchronize(distrgrid,*meshFunctionptr);
		MPI::COMM_WORLD.Barrier();
		sync.stop();

		sum+=dof[2*gridptr->getDimensions().y()]; //dummy acces to array	
	}
  
#ifdef OUTPUT	
  //print local dof
  int maxx=gridptr->getDimensions().x();
  int maxy=gridptr->getDimensions().y();
  distrgrid.printcoords();
  for(int j=0;j<maxy;j++)
  {
	for(int i=0;i<maxx;i++)
	{
		cout <<dof[maxx*j+i]<<"  ";
	}
	cout << endl;
  }
  cout << endl<<endl;
#endif
  
  all.stop();
  
  
  
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
