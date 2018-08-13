/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <iostream>
#include <sstream>

using namespace std;


#ifdef HAVE_MPI

#include <TNL/Containers/Array.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Functions/MeshFunction.h>

#include <TNL/Timer.h>
#include  <TNL/SharedPointer.h>

//#define DIMENSION 3
//#define OUTPUT 
//#define XDISTR
//#define YDISTR
//#define ZDISTR

#include "../../src/UnitTests/Mpi/Functions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Meshes::DistributedMeshes;
using namespace TNL::Communicators;
using namespace TNL::Functions;
using namespace TNL::Devices;
#endif
 
int main ( int argc, char *argv[])
{
    
#ifdef HAVE_MPI
  Timer all,setup,eval,sync;
    

  typedef MpiCommunicator CommunicatorType;
  //typedef NoDistrCommunicator CommType;
  typedef Grid<DIMENSION, double,Host,int> MeshType;
  typedef MeshFunction<MeshType> MeshFunctionType;
  typedef Vector<double,Host,int> DofType;
  typedef typename MeshType::Cell Cell;
  typedef typename MeshType::IndexType IndexType; 
  typedef typename MeshType::PointType PointType;
  using CoordinatesType = typename MeshType::CoordinatesType;
  
  typedef DistributedMesh<MeshType> DistributedMeshType;
  
  typedef LinearFunction<double,DIMENSION> LinearFunctionType;
  typedef ConstFunction<double,DIMENSION> ConstFunctionType;
  
  CommunicatorType::Init(argc,argv);

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

 
 CoordinatesType distr;
 for(int i=0;i<DIMENSION;i++) 
    distr[i]=1;

 #ifdef XDISTR
     distr[0]=0;
 #endif

 #ifdef YDISTR
    distr[1]=0;
 #endif

 #ifdef ZDISTR
     distr[2]=0;
 #endif

 typename MeshType::CoordinatesType overlap;
 overlap.setValue(1);
 DistributedMeshType distrgrid;
 distrgrid.setDomainDecomposition( distr );
 distrgrid.template setGlobalGrid<CommunicatorType>( globalGrid, overlap,overlap); 
   
 SharedPointer<MeshType> gridptr;
 SharedPointer<MeshFunctionType> meshFunctionptr;
 MeshFunctionEvaluator< MeshFunctionType, LinearFunctionType > linearFunctionEvaluator;
 MeshFunctionEvaluator< MeshFunctionType, ConstFunctionType > constFunctionEvaluator;
 
  distrgrid.setupGrid(*gridptr);
  
  DofType dof(gridptr->template getEntitiesCount< Cell >());

  dof.setValue(0);
  
  meshFunctionptr->bind(gridptr,dof);  
  
  SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
  SharedPointer< ConstFunctionType, Host > constFunctionPtr; 
   
  
  
  setup.stop();
  
  double sum=0.0;

  constFunctionPtr->Number=CommunicatorType::GetRank(CommunicatorType::AllGroup);
  
  for(int i=0;i<cycles;i++)
    {    
        eval.start();
        
        constFunctionEvaluator.evaluateAllEntities( meshFunctionptr , constFunctionPtr );
        //linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);
        CommunicatorType::Barrier(CommunicatorType::AllGroup);
        eval.stop();

        sync.start();    
        meshFunctionptr->template synchronize<CommunicatorType>();
        CommunicatorType::Barrier(CommunicatorType::AllGroup);
        sync.stop();

        sum+=dof[gridptr->getDimensions().x()/2]; //dummy acces to array    
    }
  all.stop();
  
#ifdef OUTPUT    
  //print local dof
  Printer<MeshType,DofType>::print_dof(CommunicatorType::GetRank(CommunicatorType::AllGroup),*gridptr, dof);
#endif
  
  if(CommunicatorType::GetRank(CommunicatorType::AllGroup)==0)
  {
    cout << sum <<endl<<endl;  
    
    cout<<"distr: ";
    cout << distrgrid.printProcessDistr().getString();
    cout << endl;
  
    cout<<"setup: "<<setup.getRealTime() <<endl;
    cout<<"evalpercycle: "<<eval.getRealTime()/cycles<<endl;
    cout<<"syncpercycle: "<<sync.getRealTime()/cycles<<endl;
    cout <<"eval: "<<eval.getRealTime()<<endl;
    cout <<"sync: "<<sync.getRealTime()<<endl;
    cout<<"all: "<<all.getRealTime()<<endl<<endl;
  }
  

  CommunicatorType::Finalize();



#else
  std::cout<<"MPI not Supported." << std::endl;
#endif
  return 0;

}
