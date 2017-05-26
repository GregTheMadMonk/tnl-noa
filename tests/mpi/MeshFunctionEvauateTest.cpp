/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <iostream>
#define HAVE_MPI
#include <TNL/mpi-supp.h>

using namespace std;

unsigned int errors=0;
unsigned int success=0;
#define TEST_TEST(a) if((a)){cout << __LINE__ <<":\t OK " <<endl;success++;}else{cout << __LINE__<<":\t FAIL" <<endl;errors++;}
#define TEST_RESULT cout<<"SUCCES: "<<success<<endl<<"ERRRORS: "<<errors<<endl;
inline void Test_Say( const char * message)
{
#ifdef TEST_VERBOSE
	cout << message <<endl;
#endif
}

#include <TNL/Containers/Array.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedGrid.h>
#include <TNL/Meshes/DistributedGridSynchronizer.h>
#include <TNL/Functions/MeshFunction.h>



#include "Functions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Functions;
using namespace TNL::Devices;

int main ( int argc, char *argv[])
{
  
  cout << "MeshFunction Evaluate Test for MPI develop by hanouvit" << endl;
#ifdef HAVE_MPI
  MPI::Init(argc,argv);
  
  //cout << "MPI is inicialized: "<<MPI::Is_initialized() << endl;
  
 // cout << MPI::COMM_WORLD.Get_rank() << "/" << MPI::COMM_WORLD.Get_size() << endl;
  
  //typedef Grid<1,double,Host,int> MeshType;
  typedef Grid<2,double,Host,int> MeshType;
  typedef MeshFunction<MeshType> MeshFunctionType;
  typedef Vector<double,Host,int> DofType;
  typedef typename MeshType::Cell Cell;
  typedef typename MeshType::IndexType IndexType; 
  typedef typename MeshType::PointType PointType; 
  
  
 PointType globalOrigin;
 globalOrigin.x()=-0.5;
 globalOrigin.y()=-0.5;
 
 PointType globalProportions;
 globalProportions.x()=8;
 globalProportions.y()=7;
 
 
 MeshType globalGrid;
 //globalGrid.setDimensions(9);
 globalGrid.setDimensions(8,7);
 globalGrid.setDomain(globalOrigin,globalProportions);
 
 DistributedGrid<MeshType> distrgrid(globalGrid); 
  
 SharedPointer<MeshType> gridptr;
 SharedPointer<MeshFunctionType> meshFunctionptr;
 MeshFunctionEvaluator< MeshFunctionType, FunctionToEvaluate<double,2> > evaluator;
 MeshFunctionEvaluator< MeshFunctionType, ZeroFunction<double,2>> zeroevaluator;
 
  distrgrid.SetupGrid(*gridptr);
  
  DofType dof(gridptr->template getEntitiesCount< Cell >());

  meshFunctionptr->bind(gridptr,dof);  
  
  SharedPointer< FunctionToEvaluate<double,2>, Host > functionToEvaluate;
  SharedPointer< ZeroFunction<double,2>, Host > zero; 
   
  	//zeroevaluator.evaluateBoundaryEntities( meshFunctionptr , zero );
	evaluator.evaluateAllEntities( meshFunctionptr , functionToEvaluate );
    zeroevaluator.evaluateBoundaryEntities( meshFunctionptr , zero );

  int maxx=gridptr->getDimensions().x();
  int maxy=gridptr->getDimensions().y();
  
/*  distrgrid.printcoords();
  for(int j=0;j<maxy;j++)
  {
	for(int i=0;i<maxx;i++)
	{
		cout <<dof[maxx*j+i]<<"  ";
	}
	cout << endl;
  }
  cout << endl;*/
  
  
  DistributedGridSynchronizer<DistributedGrid<MeshType>,MeshFunctionType,2>::Synchronize(distrgrid,*meshFunctionptr);
  
  //print local dof
  
  distrgrid.printcoords();
  for(int j=0;j<maxy;j++)
  {
	for(int i=0;i<maxx;i++)
	{
		cout <<dof[maxx*j+i]<<"  ";
	}
	cout << endl;
  }
  cout << endl;
  
 // int maxx=gridptr->getDimensions().x();
 /* distrgrid.printcoords();
  for(int i=0;i<maxx;i++)
  {
	  cout <<dof[i]<<"   ";
  }
  cout << endl;*/
  

  /*distrgrid.printcoords();
  for(int i=0;i<maxx;i++)
  {
	  cout <<dof[i]<<"   ";
  }
  cout << endl;*/
  
  MPI::Finalize();

#else
  std::cout<<"MPI not Supported." << std::endl;
#endif
  return 0;
}
