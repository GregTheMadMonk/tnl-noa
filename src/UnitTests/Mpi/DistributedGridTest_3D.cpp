#ifdef HAVE_GTEST 
#include <gtest/gtest.h>

#ifdef HAVE_MPI    

#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Communicators/MpiCommunicator.h>

#include "Functions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Functions;
using namespace TNL::Devices;
using namespace TNL::Communicators;
using namespace TNL::Meshes::DistributedMeshes; 

template<typename DofType>
void setDof_3D(DofType &dof, typename DofType::RealType value)
{
    for(int i=0;i<dof.getSize();i++)
        dof[i]=value;
}

template<typename GridType>
int getAdd(GridType &grid,bool bottom, bool nord, bool west )
{
    int maxx=grid.getDimensions().x();
    int maxy=grid.getDimensions().y();
    int maxz=grid.getDimensions().z();

    int add=0;
    if(!west)
        add+=maxx-1;
    if(!nord)
        add+=(maxy-1)*maxx;
    if(!bottom)
        add+=(maxz-1)*maxx*maxy;

    return add;
}

template<typename DofType,typename GridType>
void checkConner(GridType &grid, DofType &dof,bool bottom, bool nord, bool west, typename DofType::RealType expectedValue )
{
    int i=getAdd(grid,bottom,nord,west);
    EXPECT_EQ( dof[i], expectedValue) << "Conner test failed";
    
}

template<typename DofType,typename GridType>
void checkXDirectionEdge(GridType &grid, DofType &dof, bool bottom, bool nord, typename DofType::RealType expectedValue)
{
    int add=getAdd(grid,bottom,nord,true);        
    for(int i=1;i<grid.getDimensions().x()-1;i++) 
            EXPECT_EQ( dof[i+add], expectedValue) << "X direction Edge test failed " << i;
}


template<typename DofType,typename GridType>
void checkYDirectionEdge(GridType &grid, DofType &dof, bool bottom, bool west, typename DofType::RealType expectedValue)
{
    int add=getAdd(grid,bottom,true,west);
    for(int i=1;i<grid.getDimensions().y()-1;i++) 
            EXPECT_EQ( dof[grid.getDimensions().x()*i+add], expectedValue) << "Y direction Edge test failed " << i;
}

template<typename DofType,typename GridType>
void checkZDirectionEdge(GridType &grid, DofType &dof, bool nord, bool west, typename DofType::RealType expectedValue)
{
    int add=getAdd(grid,true,nord,west);
    for(int i=1;i<grid.getDimensions().z()-1;i++) 
            EXPECT_EQ( dof[grid.getDimensions().y()*grid.getDimensions().x()*i+add], expectedValue) << "Z direction Edge test failed " << i;
}

template<typename DofType,typename GridType>
void checkZFace(GridType &grid, DofType &dof, bool bottom, typename DofType::RealType expectedValue)
{
    int add=getAdd(grid,bottom,true,true);
    for(int i=1;i<grid.getDimensions().y()-1;i++)
        for(int j=1; j<grid.getDimensions().x()-1;j++)
        {
            EXPECT_EQ( dof[grid.getDimensions().x()*i+j+add], expectedValue) << "Z Face test failed "<<i<< " " << j;
        }
}

template<typename DofType,typename GridType>
void checkYFace(GridType &grid, DofType &dof, bool nord, typename DofType::RealType expectedValue)
{
    int add=getAdd(grid,true,nord,true);
    for(int i=1;i<grid.getDimensions().z()-1;i++)
        for(int j=1; j<grid.getDimensions().x()-1;j++)
        {
            EXPECT_EQ( dof[grid.getDimensions().y()*grid.getDimensions().x()*i+j+add], expectedValue) << "Y Face test failed "<<i<< " " << j;
        }
}

template<typename DofType,typename GridType>
void checkXFace(GridType &grid, DofType &dof, bool west, typename DofType::RealType expectedValue)
{
    int add=getAdd(grid,true,true,west);
    for(int i=1;i<grid.getDimensions().z()-1;i++)
        for(int j=1; j<grid.getDimensions().y()-1;j++)
        {
            EXPECT_EQ( dof[grid.getDimensions().y()*grid.getDimensions().x()*i+grid.getDimensions().x()*j+add], expectedValue) << "X Face test failed "<<i<< " " << j;
        }
}

/*
Expected 27 process
*/
template<typename DofType,typename GridType>
void check_Boundary_3D(int rank, GridType &grid, DofType &dof, typename DofType::RealType expectedValue)
{
    if(rank==0)//Bottom North West
    {
        checkConner(grid,dof,true, true, true, expectedValue );
        checkXDirectionEdge(grid,dof,true,true,expectedValue);
        checkYDirectionEdge(grid,dof,true,true,expectedValue);
        checkZDirectionEdge(grid,dof,true,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }    

    if(rank==1)//Bottom North Center
    {
        checkXDirectionEdge(grid,dof,true,true,expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }

    if(rank==2)//Bottom North East
    {
        checkConner(grid,dof,true, true, false, expectedValue );
        checkXDirectionEdge(grid,dof,true,true,expectedValue);
        checkYDirectionEdge(grid,dof,true,false,expectedValue);
        checkZDirectionEdge(grid,dof,true,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }

    if(rank==3)//Bottom Center West
    {
        checkYDirectionEdge(grid,dof,true,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }

    if(rank==4)//Bottom Center Center
    {
        checkZFace(grid, dof, true, expectedValue);
    }


    if(rank==5)//Bottom Center East
    {
        checkYDirectionEdge(grid,dof,true,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }

    if(rank==6)//Bottom South West
    {
        checkConner(grid,dof,true, false, true, expectedValue );
        checkXDirectionEdge(grid,dof,true,false,expectedValue);
        checkYDirectionEdge(grid,dof,true,true,expectedValue);
        checkZDirectionEdge(grid,dof,false,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }

    if(rank==7)//Bottom South Center
    {
        checkXDirectionEdge(grid,dof,true,false,expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }

    if(rank==8)//Bottom South East
    {
        checkConner(grid,dof,true, false, false, expectedValue );
        checkXDirectionEdge(grid,dof,true,false,expectedValue);
        checkYDirectionEdge(grid,dof,true,false,expectedValue);
        checkZDirectionEdge(grid,dof,false,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }

    if(rank==9)//Center North West
    {
        checkZDirectionEdge(grid,dof,true,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
    }    

    if(rank==10)//Center North Center
    {
        checkYFace(grid, dof, true, expectedValue);
    }

    if(rank==11)//Center North East
    {
        checkZDirectionEdge(grid,dof,true,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
    }

    if(rank==12)//Center Center West
    {
        checkXFace(grid, dof, true, expectedValue);
    }

    if(rank==13)//Center Center Center
    {
        //no Boundary
    }


    if(rank==14)//Center Center East
    {
        checkXFace(grid, dof, false, expectedValue);
    }

    if(rank==15)//Center South West
    {
        checkZDirectionEdge(grid,dof,false,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
    }

    if(rank==16)//Center South Center
    {
        checkYFace(grid, dof, false, expectedValue);
    }

    if(rank==17)//Center South East
    {
        checkZDirectionEdge(grid,dof,false,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
    }

    if(rank==18)//Top North West
    {
        checkConner(grid,dof,false, true, true, expectedValue );
        checkXDirectionEdge(grid,dof,false,true,expectedValue);
        checkYDirectionEdge(grid,dof,false,true,expectedValue);
        checkZDirectionEdge(grid,dof,true,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }    

    if(rank==19)//Top North Center
    {
        checkXDirectionEdge(grid,dof,false,true,expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }

    if(rank==20)//Top North East
    {
        checkConner(grid,dof,false, true, false, expectedValue );
        checkXDirectionEdge(grid,dof,false,true,expectedValue);
        checkYDirectionEdge(grid,dof,false,false,expectedValue);
        checkZDirectionEdge(grid,dof,true,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }

    if(rank==21)//Top Center West
    {
        checkYDirectionEdge(grid,dof,false,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }

    if(rank==22)//Top Center Center
    {
        checkZFace(grid, dof, false, expectedValue);
    }


    if(rank==23)//Top Center East
    {
        checkYDirectionEdge(grid,dof,false,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }

    if(rank==24)//Top South West
    {
        checkConner(grid,dof,false, false, true, expectedValue );
        checkXDirectionEdge(grid,dof,false,false,expectedValue);
        checkYDirectionEdge(grid,dof,false,true,expectedValue);
        checkZDirectionEdge(grid,dof,false,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }

    if(rank==25)//Top South Center
    {
        checkXDirectionEdge(grid,dof,false,false,expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }

    if(rank==26)//Top South East
    {
        checkConner(grid,dof,false, false, false, expectedValue );
        checkXDirectionEdge(grid,dof,false,false,expectedValue);
        checkYDirectionEdge(grid,dof,false,false,expectedValue);
        checkZDirectionEdge(grid,dof,false,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }

}

template<typename DofType,typename GridType>
void CheckConnerNode_Overlap(GridType &grid, DofType &dof,bool bottom, bool nord, bool west, typename DofType::RealType expectedValue)
{
    checkConner(grid,dof,!bottom, !nord, !west, expectedValue );
    checkXDirectionEdge(grid,dof,!bottom,!nord,expectedValue);
    checkYDirectionEdge(grid,dof,!bottom,!west,expectedValue);
    checkZDirectionEdge(grid,dof,!nord,!west,expectedValue);
    checkXFace(grid, dof, !west, expectedValue);
    checkYFace(grid, dof, !nord, expectedValue);
    checkZFace(grid, dof, !bottom, expectedValue);
}

template<typename DofType,typename GridType>
void CheckXEdgeNode_Overlap(GridType &grid, DofType &dof,bool bottom, bool nord, typename DofType::RealType expectedValue)
{
    checkConner(grid,dof,!bottom, !nord, false, expectedValue );
    checkConner(grid,dof,!bottom, !nord, true, expectedValue );
    checkXDirectionEdge(grid,dof,!bottom, !nord,expectedValue);
    checkYDirectionEdge(grid,dof,!bottom,false,expectedValue);
    checkYDirectionEdge(grid,dof,!bottom,true,expectedValue);
    checkZDirectionEdge(grid,dof,!nord,false,expectedValue);
    checkZDirectionEdge(grid,dof,!nord,true,expectedValue);
    checkXFace(grid, dof, false, expectedValue);
    checkXFace(grid, dof, true, expectedValue);
    checkYFace(grid, dof, !nord, expectedValue);
    checkZFace(grid, dof, !bottom, expectedValue);
}

template<typename DofType,typename GridType>
void CheckYEdgeNode_Overlap(GridType &grid, DofType &dof,bool bottom, bool west, typename DofType::RealType expectedValue)
{
        checkConner(grid,dof,!bottom, false, !west, expectedValue );
        checkConner(grid,dof,!bottom, true, !west, expectedValue );
        checkXDirectionEdge(grid,dof,!bottom,false,expectedValue);
        checkXDirectionEdge(grid,dof,!bottom,true,expectedValue);
        checkYDirectionEdge(grid,dof,!bottom,!west,expectedValue);
        checkZDirectionEdge(grid,dof,false,!west,expectedValue);
        checkZDirectionEdge(grid,dof,true,!west,expectedValue);
        checkXFace(grid, dof, !west, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, !bottom, expectedValue);
}

template<typename DofType,typename GridType>
void CheckZEdgeNode_Overlap(GridType &grid, DofType &dof,bool nord, bool west, typename DofType::RealType expectedValue)
{
        checkConner(grid,dof,false, !nord, !west, expectedValue );
        checkConner(grid,dof,true, !nord, !west, expectedValue );
        checkXDirectionEdge(grid,dof,false,!nord,expectedValue);
        checkXDirectionEdge(grid,dof,true,!nord,expectedValue);
        checkYDirectionEdge(grid,dof,false,!west,expectedValue);
        checkYDirectionEdge(grid,dof,true,!west,expectedValue);
        checkZDirectionEdge(grid,dof,!nord,!west,expectedValue);
        checkXFace(grid, dof, !west, expectedValue);
        checkYFace(grid, dof, !nord, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
}

template<typename DofType,typename GridType>
void CheckXFaceNode_Overlap(GridType &grid, DofType &dof,bool west, typename DofType::RealType expectedValue)
{
        checkConner(grid,dof,false, false, !west, expectedValue );
        checkConner(grid,dof,false, true, !west, expectedValue );
        checkConner(grid,dof,true, false, !west, expectedValue );
        checkConner(grid,dof,true, true, !west, expectedValue );
        checkXDirectionEdge(grid,dof,false,false,expectedValue);
        checkXDirectionEdge(grid,dof,false,true,expectedValue);
        checkXDirectionEdge(grid,dof,true,false,expectedValue);
        checkXDirectionEdge(grid,dof,true,true,expectedValue);
        checkYDirectionEdge(grid,dof,false,!west,expectedValue);
        checkYDirectionEdge(grid,dof,true,!west,expectedValue);
        checkZDirectionEdge(grid,dof,false,!west,expectedValue);
        checkZDirectionEdge(grid,dof,true,!west,expectedValue);
        checkXFace(grid, dof, !west, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, false, expectedValue);    
        checkZFace(grid, dof, true, expectedValue);        
}

template<typename DofType,typename GridType>
void CheckYFaceNode_Overlap(GridType &grid, DofType &dof,bool nord, typename DofType::RealType expectedValue)
{
        checkConner(grid,dof, false,!nord, false, expectedValue );
        checkConner(grid,dof, false,!nord, true, expectedValue );
        checkConner(grid,dof, true, !nord, false, expectedValue );
        checkConner(grid,dof, true, !nord, true, expectedValue );
        checkXDirectionEdge(grid,dof,false,!nord,expectedValue);
        checkXDirectionEdge(grid,dof,true,!nord,expectedValue);
        checkYDirectionEdge(grid,dof,false,false,expectedValue);
        checkYDirectionEdge(grid,dof,false,true,expectedValue);
        checkYDirectionEdge(grid,dof,true,false,expectedValue);
        checkYDirectionEdge(grid,dof,true,true,expectedValue);
        checkZDirectionEdge(grid,dof,!nord,false,expectedValue);
        checkZDirectionEdge(grid,dof,!nord,true,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkYFace(grid, dof, !nord, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, true, expectedValue);    
}

template<typename DofType,typename GridType>
void CheckZFaceNode_Overlap(GridType &grid, DofType &dof,bool bottom, typename DofType::RealType expectedValue)
{
        checkConner(grid,dof,!bottom, false, false, expectedValue );
        checkConner(grid,dof,!bottom, false, true, expectedValue );
        checkConner(grid,dof,!bottom, true, false, expectedValue );
        checkConner(grid,dof,!bottom, true, true, expectedValue );
        checkXDirectionEdge(grid,dof,!bottom,false,expectedValue);
        checkXDirectionEdge(grid,dof,!bottom,true,expectedValue);
        checkYDirectionEdge(grid,dof,!bottom,false,expectedValue);
        checkYDirectionEdge(grid,dof,!bottom,true,expectedValue);
        checkZDirectionEdge(grid,dof,false,false,expectedValue);
        checkZDirectionEdge(grid,dof,false,true,expectedValue);
        checkZDirectionEdge(grid,dof,true,false,expectedValue);
        checkZDirectionEdge(grid,dof,true,true,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, !bottom, expectedValue);    
}

template<typename DofType,typename GridType>
void CheckCentralNode_Overlap(GridType &grid, DofType &dof,typename DofType::RealType expectedValue)
{
        checkConner(grid,dof,false, false, false, expectedValue );
        checkConner(grid,dof,false, false, true, expectedValue );
        checkConner(grid,dof,false, true, false, expectedValue );
        checkConner(grid,dof,false, true, true, expectedValue );
        checkConner(grid,dof,true, false, false, expectedValue );
        checkConner(grid,dof,true, false, true, expectedValue );
        checkConner(grid,dof,true, true, false, expectedValue );
        checkConner(grid,dof,true, true, true, expectedValue );

        checkXDirectionEdge(grid,dof,false,false,expectedValue);
        checkXDirectionEdge(grid,dof,false,true,expectedValue);
        checkXDirectionEdge(grid,dof,true,false,expectedValue);
        checkXDirectionEdge(grid,dof,true,true,expectedValue);
        checkYDirectionEdge(grid,dof,false,false,expectedValue);
        checkYDirectionEdge(grid,dof,false,true,expectedValue);
        checkYDirectionEdge(grid,dof,true,false,expectedValue);
        checkYDirectionEdge(grid,dof,true,true,expectedValue);
        checkZDirectionEdge(grid,dof,false,false,expectedValue);
        checkZDirectionEdge(grid,dof,false,true,expectedValue);
        checkZDirectionEdge(grid,dof,true,false,expectedValue);
        checkZDirectionEdge(grid,dof,true,true,expectedValue);

        checkXFace(grid, dof, false, expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, true, expectedValue);    
}

/*
* Expected 27 procs. 
*/
template<typename DofType,typename GridType>
void check_Overlap_3D(int rank, GridType &grid, DofType &dof, typename DofType::RealType expectedValue)
{
    if(rank==0)
        CheckConnerNode_Overlap(grid,dof,true,true,true,expectedValue);

    if(rank==1)
        CheckXEdgeNode_Overlap(grid,dof,true,true,expectedValue);

    if(rank==2)    
        CheckConnerNode_Overlap(grid,dof,true,true,false,expectedValue);
    
    if(rank==3)
        CheckYEdgeNode_Overlap(grid,dof,true,true,expectedValue);

    if(rank==4)
        CheckZFaceNode_Overlap(grid,dof,true,expectedValue);
        
    if(rank==5)
        CheckYEdgeNode_Overlap(grid,dof,true,false,expectedValue);
    
    if(rank==6)
        CheckConnerNode_Overlap(grid,dof,true,false,true,expectedValue);

    if(rank==7)
        CheckXEdgeNode_Overlap(grid,dof,true,false,expectedValue);

    if(rank==8)
        CheckConnerNode_Overlap(grid,dof,true,false,false,expectedValue);

    if(rank==9)
        CheckZEdgeNode_Overlap(grid,dof,true,true,expectedValue);

    if(rank==10)
        CheckYFaceNode_Overlap(grid,dof,true,expectedValue);

    if(rank==11)
        CheckZEdgeNode_Overlap(grid,dof,true,false,expectedValue);

    if(rank==12)
        CheckXFaceNode_Overlap(grid,dof,true,expectedValue);

    if(rank==13)
        CheckCentralNode_Overlap(grid,dof,expectedValue);

    if(rank==14)
        CheckXFaceNode_Overlap(grid,dof,false,expectedValue);

    if(rank==15)
        CheckZEdgeNode_Overlap(grid,dof,false,true,expectedValue);

    if(rank==16)
        CheckYFaceNode_Overlap(grid,dof,false,expectedValue);

    if(rank==17)
        CheckZEdgeNode_Overlap(grid,dof,false,false,expectedValue);
    
    if(rank==18)
        CheckConnerNode_Overlap(grid,dof,false,true,true,expectedValue);

    if(rank==19)
        CheckXEdgeNode_Overlap(grid,dof,false,true,expectedValue);

    if(rank==20)    
        CheckConnerNode_Overlap(grid,dof,false,true,false,expectedValue);
    
    if(rank==21)
        CheckYEdgeNode_Overlap(grid,dof,false,true,expectedValue);

    if(rank==22)
        CheckZFaceNode_Overlap(grid,dof,false,expectedValue);
        
    if(rank==23)
        CheckYEdgeNode_Overlap(grid,dof,false,false,expectedValue);
    
    if(rank==24)
        CheckConnerNode_Overlap(grid,dof,false,false,true,expectedValue);

    if(rank==25)
        CheckXEdgeNode_Overlap(grid,dof,false,false,expectedValue);

    if(rank==26)
        CheckConnerNode_Overlap(grid,dof,false,false,false,expectedValue);

}

template<typename DofType,typename GridType>
void check_Inner_3D(int rank, GridType grid, DofType dof, typename DofType::RealType expectedValue)
{
    int maxx=grid.getDimensions().x();
    int maxy=grid.getDimensions().y();
    int maxz=grid.getDimensions().z();
    for(int k=1;k<maxz-1;k++)
        for(int j=1;j<maxy-1;j++)//prvni a posledni jsou buď hranice, nebo overlap
            for(int i=1;i<maxx-1;i++) //buď je vlevo hranice, nebo overlap
                EXPECT_EQ( dof[k*maxx*maxy+j*maxx+i], expectedValue) <<" "<<k <<" "<< j<<" "<<i << " " << maxx << " " << maxy<< " " << maxz;
}


/*
 * Light check of 3D distributed grid and its synchronization. 
 * expected 27 processors
 */
typedef MpiCommunicator CommunicatorType;
typedef Grid<3,double,Host,int> MeshType;
typedef MeshFunction<MeshType> MeshFunctionType;
typedef Vector<double,Host,int> DofType;
typedef typename MeshType::Cell Cell;
typedef typename MeshType::IndexType IndexType; 
typedef typename MeshType::PointType PointType; 
typedef DistributedMesh<MeshType> DistributedGridType;
     
class DistributedGirdTest_3D : public ::testing::Test {
 protected:

    static DistributedGridType *distrgrid;
    static DofType *dof;

    static SharedPointer<MeshType> gridptr;
    static SharedPointer<MeshFunctionType> meshFunctionptr;

    static MeshFunctionEvaluator< MeshFunctionType, ConstFunction<double,3> > constFunctionEvaluator;
    static SharedPointer< ConstFunction<double,3>, Host > constFunctionPtr;

    static MeshFunctionEvaluator< MeshFunctionType, LinearFunction<double,3> > linearFunctionEvaluator;
    static SharedPointer< LinearFunction<double,3>, Host > linearFunctionPtr;

    static int rank;
    static int nproc;    
     
  // Per-test-case set-up.
  // Called before the first test in this test case.
  // Can be omitted if not needed.
  static void SetUpTestCase() {
      
    int size=10;
    rank=MPI::COMM_WORLD.Get_rank();
    nproc=MPI::COMM_WORLD.Get_size();
    
    PointType globalOrigin;
    PointType globalProportions;
    MeshType globalGrid;
    
    globalOrigin.x()=-0.5;
    globalOrigin.y()=-0.5;
    globalOrigin.z()=-0.5;    
    globalProportions.x()=size;
    globalProportions.y()=size;
    globalProportions.z()=size;
        
    globalGrid.setDimensions(size,size,size);
    globalGrid.setDomain(globalOrigin,globalProportions);
    
    typename DistributedGridType::CoordinatesType overlap;
    overlap.setValue(1);
    distrgrid=new DistributedGridType();
    distrgrid->setDomainDecomposition( typename DistributedGridType::CoordinatesType( 3, 3, 3 ) );
    distrgrid->template setGlobalGrid<CommunicatorType>( globalGrid, overlap );
    
    distrgrid->setupGrid(*gridptr);
    dof=new DofType(gridptr->template getEntitiesCount< Cell >());
    
    meshFunctionptr->bind(gridptr,*dof);   
    constFunctionPtr->Number=rank;
    
  }

  // Per-test-case tear-down.
  // Called after the last test in this test case.
  // Can be omitted if not needed.
  static void TearDownTestCase() {
      delete dof;
      delete distrgrid;

  }

};

DistributedGridType *DistributedGirdTest_3D::distrgrid=NULL;
DofType *DistributedGirdTest_3D::dof=NULL;
SharedPointer<MeshType> DistributedGirdTest_3D::gridptr;
SharedPointer<MeshFunctionType> DistributedGirdTest_3D::meshFunctionptr;
MeshFunctionEvaluator< MeshFunctionType, ConstFunction<double,3> > DistributedGirdTest_3D::constFunctionEvaluator;
SharedPointer< ConstFunction<double,3>, Host > DistributedGirdTest_3D::constFunctionPtr;
MeshFunctionEvaluator< MeshFunctionType, LinearFunction<double,3> > DistributedGirdTest_3D::linearFunctionEvaluator;
SharedPointer< LinearFunction<double,3>, Host > DistributedGirdTest_3D::linearFunctionPtr;
int DistributedGirdTest_3D::rank;
int DistributedGirdTest_3D::nproc;    

TEST_F(DistributedGirdTest_3D, evaluateAllEntities)
{

    //Check Traversars
    //All entities, witout overlap
    setDof_3D(*dof,-1);
    constFunctionEvaluator.evaluateAllEntities( meshFunctionptr , constFunctionPtr );
    //Printer<MeshType,DofType>::print_dof(rank,*gridptr,*dof);
    check_Boundary_3D(rank, *gridptr, *dof, rank);
    check_Overlap_3D(rank, *gridptr, *dof, -1);
    check_Inner_3D(rank, *gridptr, *dof, rank);
}

TEST_F(DistributedGirdTest_3D, evaluateBoundaryEntities)
{
    //Boundary entities, witout overlap
    setDof_3D(*dof,-1);
    constFunctionEvaluator.evaluateBoundaryEntities( meshFunctionptr , constFunctionPtr );
    check_Boundary_3D(rank, *gridptr, *dof, rank);
    check_Overlap_3D(rank, *gridptr, *dof, -1);
    check_Inner_3D(rank, *gridptr, *dof, -1);
}

TEST_F(DistributedGirdTest_3D, evaluateInteriorEntities)
{
    //Inner entities, witout overlap
    setDof_3D(*dof,-1);
    constFunctionEvaluator.evaluateInteriorEntities( meshFunctionptr , constFunctionPtr );
    check_Boundary_3D(rank, *gridptr, *dof, -1);
    check_Overlap_3D(rank, *gridptr, *dof, -1);
    check_Inner_3D(rank, *gridptr, *dof, rank);
}   

TEST_F(DistributedGirdTest_3D, LinearFunctionTest)
{
    //fill meshfunction with linear function (physical center of cell corresponds with its coordinates in grid) 
    setDof_3D(*dof,-1);
    linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr, linearFunctionPtr);
    meshFunctionptr->template synchronize<CommunicatorType>();
    
    int count =gridptr->template getEntitiesCount< Cell >();
    for(int i=0;i<count;i++)
    {
            auto entity= gridptr->template getEntity< Cell >(i);
            entity.refresh();
            EXPECT_EQ(meshFunctionptr->getValue(entity), (*linearFunctionPtr)(entity)) << "Linear function doesnt fit recievd data. " << entity.getCoordinates().x() << " "<<entity.getCoordinates().y() << " "<< gridptr->getDimensions().x() <<" "<<gridptr->getDimensions().y();
    }
}

/* not implemented
TEST_F(DistributedGirdTest_3D, SynchronizerNeighborTest)
{

}
*/

#else
TEST(NoMPI, NoTest)
{
    ASSERT_TRUE(true) << ":-(";
}
#endif

#endif


#if (defined(HAVE_GTEST) && defined(HAVE_MPI))
#include <sstream>

  class MinimalistBuffredPrinter : public ::testing::EmptyTestEventListener {
      
  private:
      std::stringstream sout;
      
  public:
      
    // Called before a test starts.
    virtual void OnTestStart(const ::testing::TestInfo& test_info) {
      sout<< test_info.test_case_name() <<"." << test_info.name() << " Start." <<std::endl;
    }

    // Called after a failed assertion or a SUCCEED() invocation.
    virtual void OnTestPartResult(
        const ::testing::TestPartResult& test_part_result) {
      sout << (test_part_result.failed() ? "====Failure=== " : "===Success=== ") 
              << test_part_result.file_name() << " "
              << test_part_result.line_number() <<std::endl
              << test_part_result.summary() <<std::endl;
    }

    // Called after a test ends.
    virtual void OnTestEnd(const ::testing::TestInfo& test_info) 
    {
        int rank=CommunicatorType::GetRank(CommunicatorType::AllGroup);
        sout<< test_info.test_case_name() <<"." << test_info.name() << " End." <<std::endl;
        std::cout << rank << ":" << std::endl << sout.str()<< std::endl;
        sout.str( std::string() );
        sout.clear();
    }
  };
#endif

#include "../../src/UnitTests/GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );

    #ifdef HAVE_MPI
       ::testing::TestEventListeners& listeners =
          ::testing::UnitTest::GetInstance()->listeners();

       delete listeners.Release(listeners.default_result_printer());
       listeners.Append(new MinimalistBuffredPrinter);

       CommunicatorType::Init(argc,argv);
    #endif
       int result= RUN_ALL_TESTS();

    #ifdef HAVE_MPI
       CommunicatorType::Finalize();
    #endif
       return result;
#else
   
   throw GtestMissingError();
#endif
}




