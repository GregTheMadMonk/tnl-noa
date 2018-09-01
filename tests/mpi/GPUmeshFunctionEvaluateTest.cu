#include <iostream>


#if defined(HAVE_MPI) && defined(HAVE_CUDA)

#include <TNL/Timer.h>
#include <TNL/SharedPointer.h>
#include <TNL/Containers/Array.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.h>


using namespace std;

#define DIMENSION 3
//#define OUTPUT 
#define XDISTR
//#define YDISTR
//#define ZDISTR

#include "../../src/UnitTests/Functions/Functions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Meshes::DistributedMeshes;
using namespace TNL::Communicators;
using namespace TNL::Functions;
using namespace TNL::Devices;

 
int main ( int argc, char *argv[])
{
    
  Timer all,setup,eval,sync;
  
  typedef Cuda Device;  
  //typedef Host Device;  

  typedef MpiCommunicator CommunicatorType;
  //typedef NoDistrCommunicator CommType;
  typedef Grid<DIMENSION, double,Device,int> MeshType;
  typedef MeshFunction<MeshType> MeshFunctionType;
  typedef Vector<double,Device,int> DofType;
  typedef typename MeshType::Cell Cell;
  typedef typename MeshType::IndexType IndexType; 
  typedef typename MeshType::PointType PointType; 
  using CoordinatesType = MeshType::CoordinatesType;
  
  typedef DistributedMesh<MeshType> DistributedMeshType;
  
  typedef LinearFunction<double,DIMENSION> LinearFunctionType;
  typedef ConstFunction<double,DIMENSION> ConstFunctionType;
  
  CommunicatorType::Init(argc,argv);

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

   DistributedMeshType distrgrid;
   distrgrid.setDomainDecomposition( distr );
   distrgrid.template setGlobalGrid<CommunicatorType>( globalGrid );
   typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< MeshType, CommunicatorType >::getOverlaps( &distrgrid, lowerOverlap, upperOverlap, 1 );
   distrgrid.setOverlaps( lowerOverlap, upperOverlap );
   
   SharedPointer<MeshType> gridptr;
   SharedPointer<MeshFunctionType> meshFunctionptr;
   MeshFunctionEvaluator< MeshFunctionType, LinearFunctionType > linearFunctionEvaluator;
   MeshFunctionEvaluator< MeshFunctionType, ConstFunctionType > constFunctionEvaluator;
 
   distrgrid.setupGrid(*gridptr);
  
   DofType dof(gridptr->template getEntitiesCount< Cell >());

   dof.setValue(0);
  
  meshFunctionptr->bind(gridptr,dof);  
  
  SharedPointer< LinearFunctionType, Device > linearFunctionPtr;
  SharedPointer< ConstFunctionType, Device > constFunctionPtr; 
   
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

        ///sum+=dof[gridptr->getDimensions().x()/2]; //dummy acces to array    
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

  return 0;

}

#else

using namespace std;

int main(void)
{
    cout << "MPI or Cuda missing...." <<endl;
}
#endif


