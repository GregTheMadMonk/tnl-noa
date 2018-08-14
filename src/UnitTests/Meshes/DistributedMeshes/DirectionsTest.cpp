#ifdef HAVE_GTEST  
#include <gtest/gtest.h>

#include <TNL/Meshes/DistributedMeshes/Directions.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/StaticVectorFor.h>

using namespace TNL::Meshes::DistributedMeshes;
using namespace TNL::Containers;
using namespace TNL;

TEST(Direction1D, Conners)
{
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(-1)),Left) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(1)),Right) << "Failed";
}

TEST(Direction2D, Edge)
{
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(-1)),Left) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(1)),Right) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(-2)),Up) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(2)),Down) << "Failed";

    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-1,0)),Left) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(1,0)),Right) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(0,-2)),Up) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(0,2)),Down) << "Failed";
}

TEST(Direction2D, Conners)
{
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-2,-1)),UpLeft) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-2, 1)),UpRight) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(2,-1)),DownLeft) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(2,1)),DownRight) << "Failed";
}

TEST(Direction3D, Faces)
{
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(-1)),West) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(1)),East) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(-2)),North) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(2)),South) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(-3)),Bottom) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(3)),Top) << "Failed";
}

TEST(Direction3D, Edges)
{
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-2,-1)),NorthWest) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-2, 1)),NorthEast) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>( 2,-1)),SouthWest) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>( 2, 1)),SouthEast) << "Failed";
    
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-3,-1)),BottomWest) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-3, 1)),BottomEast) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-3,-2)),BottomNorth) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-3, 2)),BottomSouth) << "Failed";

    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>( 3,-1)),TopWest) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>( 3, 1)),TopEast) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>( 3,-2)),TopNorth) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>( 3, 2)),TopSouth) << "Failed";
}

TEST(Direction3D, Conners)
{
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(-3,-2,-1)),BottomNorthWest) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(-3,-2, 1)),BottomNorthEast) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(-3, 2,-1)),BottomSouthWest) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(-3, 2, 1)),BottomSouthEast) << "Failed";
    
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(3,-2,-1)),TopNorthWest) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(3,-2, 1)),TopNorthEast) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(3, 2,-1)),TopSouthWest) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(3, 2, 1)),TopSouthEast) << "Failed";
}

TEST(XYZ, 2D )
{
    EXPECT_EQ( Directions::template getXYZ<2>(Left), (StaticVector<2,int>(-1,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<2>(Right), (StaticVector<2,int>(1,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<2>(Up), (StaticVector<2,int>(0,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<2>(Down), (StaticVector<2,int>(0,1)) ) << "Failed";

    EXPECT_EQ( Directions::template getXYZ<2>(UpLeft), (StaticVector<2,int>(-1,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<2>(UpRight), (StaticVector<2,int>(1,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<2>(DownLeft), (StaticVector<2,int>(-1,1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<2>(DownRight), (StaticVector<2,int>(1,1)) ) << "Failed";

}

TEST(XYZ, 3D )
{
    EXPECT_EQ( Directions::template getXYZ<3>(West), (StaticVector<3,int>(-1,0,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(East), (StaticVector<3,int>(1,0,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(North), (StaticVector<3,int>(0,-1,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(South), (StaticVector<3,int>(0,1,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(Bottom), (StaticVector<3,int>(0,0,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(Top), (StaticVector<3,int>(0,0,1)) ) << "Failed";    

    EXPECT_EQ( Directions::template getXYZ<3>(NorthWest), (StaticVector<3,int>(-1,-1,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(NorthEast), (StaticVector<3,int>(1,-1,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(SouthWest), (StaticVector<3,int>(-1,1,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(SouthEast), (StaticVector<3,int>(1,1,0)) ) << "Failed";

    EXPECT_EQ( Directions::template getXYZ<3>(BottomWest), (StaticVector<3,int>(-1,0,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(BottomEast), (StaticVector<3,int>(1,0,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(TopWest), (StaticVector<3,int>(-1,0,1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(TopEast), (StaticVector<3,int>(1,0,1)) ) << "Failed";

    EXPECT_EQ( Directions::template getXYZ<3>(BottomNorth), (StaticVector<3,int>(0,-1,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(BottomSouth), (StaticVector<3,int>(0,1,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(TopNorth), (StaticVector<3,int>(0,-1,1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(TopSouth), (StaticVector<3,int>(0,1,1)) ) << "Failed";

    EXPECT_EQ( Directions::template getXYZ<3>(TopSouthWest), (StaticVector<3,int>(-1,1,1)) ) << "Failed";

}

#endif

#include "../../src/UnitTests/GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
       int result= RUN_ALL_TESTS();
       return result;
#else
   
   throw GtestMissingError();
#endif
}
