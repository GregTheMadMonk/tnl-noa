#pragma once

#include <TNL/Assert.h>
#include <TNL/Containers/StaticVector.h>

using namespace std;

namespace TNL {
namespace Meshes { 
namespace DistributedMeshes {

//index of direction can be written as number in 3-base system 
//  -> 1 order x axis, 2 order y axis, 3 order z axis
//  -> 0 - not used, 1 negative direction, 2 positive direction
//finaly we subtrackt 1 because we dont need (0,0,0) aka 0 aka no direction

enum Directions2D { Left = 0 , Right = 1 , Up = 2, UpLeft =3, UpRight=4, Down=5, DownLeft=6, DownRight=7 }; 

/*MEH - osa zed je zdola nahoru, asi---
enum Directions3D { West = 0 , East = 1 , 
                    North = 2, NorthWest = 3, NorthEast = 4,
                    South = 5, SouthWest = 6, SouthEast = 7,
                    Top = 8, TopWest = 9, TopEast =10,
                    TopNorth = 11, TopNorthWest = 12, TopNorthEast = 13,
                    TopSouth = 14, TopSouthWest = 15,TopSouthEast = 16,
                    Bottom = 17 ,BottomWest = 18 , BottomEast = 19 , 
                    BottomNorth = 20, BottomNorthWest = 21, BottomNorthEast = 22,
                    BottomSouth = 23, BottomSouthWest = 24, BottomSouthEast = 25,
                  };*/

enum Directions3D { West = 0 , East = 1 , 
                    North = 2, NorthWest = 3, NorthEast = 4,
                    South = 5, SouthWest = 6, SouthEast = 7,
                    Bottom = 8 ,BottomWest = 9 , BottomEast = 10 , 
                    BottomNorth = 11, BottomNorthWest = 12, BottomNorthEast = 13,
                    BottomSouth = 14, BottomSouthWest = 15, BottomSouthEast = 16,
                    Top = 17, TopWest = 18, TopEast =19,
                    TopNorth = 20, TopNorthWest = 21, TopNorthEast = 22,
                    TopSouth = 23, TopSouthWest = 24,TopSouthEast = 25,
                  };

class Directions {

public:
    template<int numerofDriection>
    static int getDirection(Containers::StaticVector<numerofDriection,int> directions) //takes +/- nuber of ax (i.e. (-2,+3))
    {
        int result=0;
        for(int i=0;i<directions.size;i++)
            result+=add(directions[i]);
        return result-1;
    }

    template<int dim>
    static Containers::StaticVector<dim,int> getXYZ(int neighbor)// return neighbor as direction like (0,-1,1)
    {
        Containers::StaticVector<dim,int> res;
        int number=neighbor+1;
        for(int i=0;i<dim;i++)
        {
            int direction=number%3;
            if(direction==0)
                res[i]=0;
            if(direction==1)
                res[i]=-1;
            if(direction==2)
                res[i]=1;
            number=number/3;
        }
        return res;
    }
    

 /*   static int getDirection(int direction)
    {
        int result=0;
        result+=add(direction);
        return result-1;
    }

    static int getDirection(int direction1,int direction2)
    {
        int result=0;
        result+=add(direction1);
        result+=add(direction2);
        return result-1;
    }

    static int getDirection(int direction1,int direction2, int direction3)
    {
        int result=0;
        result+=add(direction1);
        result+=add(direction2);
        result+=add(direction3);
        return result-1;
    }*/
    
    static int add(int direction)
    {
        if(direction==0)
            return 0;

        if(direction>0)
            return 2*pow3(direction-1); //positive direction has higer index
        else
            return pow3(-direction-1);


    }

    static int pow3(int exp)
    {
        int ret=1;
        for(int i=0;i<exp;i++)
            ret*=3;
        return ret;
    }
};

//for c++11 -- in c++14 simply 3^dim-1
template<int dim>
class DirectionCount
{
public:
    static constexpr int get(){return 0;}
};

template <>
class DirectionCount<1>
{
public:
    static constexpr int get(){return 2;}
};

template <>
class DirectionCount<2>
{
public:
    static constexpr int get(){return 8;}
};

template <>
class DirectionCount<3>
{
public:
    static constexpr int get(){return 26;}
};


} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

