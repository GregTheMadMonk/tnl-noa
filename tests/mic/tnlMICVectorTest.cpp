/***************************************************************************
                          tnlMICVectorTest.cpp  -  
                application testing Vector implemntation on MIC KNC
                              by hanouvit 
                             -------------------
    copyright            : (C) 2004 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <iostream>
#include <omp.h>
#include <stdint.h>

#include <TNL/Devices/MIC.h>
#include <TNL/Containers/Vector.h>


using namespace std;
using namespace TNL;
using namespace TNL::Containers;

//TUNE MACROS FOR YOUR FUNKY OUTPUT
#define TEST_VERBOSE

unsigned int errors=0;
unsigned int success=0;
#define TEST_TEST(a) if((a)){cout << __LINE__ <<":\t OK" <<endl;success++;}else{cout << __LINE__<<":\t FAIL" <<endl;errors++;}
#define TEST_RESULT cout<<"SUCCES: "<<success<<endl<<"ERRRORS: "<<errors<<endl;
inline void Test_Say( const char * message)
{
#ifdef TEST_VERBOSE
	cout << message <<endl;
#endif
}

int main(void)
{
    cout << "Vector on MIC test by hanouvit:" <<endl;
	
    #ifdef HAVE_ICPC
            cout << "ICPC in USE" <<endl; 
    #endif

    #ifdef HAVE_MIC
            cout << "MIC in USE" <<endl;
    #endif

#ifdef HAVE_MIC
        Vector<double,Devices::MIC,int> aa(10);
        Vector<double,Devices::MIC,int> bb(10);
        Vector<double,Devices::MIC,int> cc(10);

        Vector<double,Devices::Host,int> aaa(10);
        Vector<double,Devices::Host,int> bbb(10);
        Vector<double,Devices::Host,int> ccc(10);

        for(int i=0;i<10;i++)
        {
            aa.setElement(i,i-5);
            aaa.setElement(i,i-5);
            bb.setElement(i,5-i);
            bbb.setElement(i,5-i);
            cc.setElement(i,10+i);
            ccc.setElement(i,10+i);
        }

        Test_Say("Is filled correctly?:");
        for(int i=0;i<10;i++)
        {
            TEST_TEST(aa.getElement(i)==aaa.getElement(i));
            TEST_TEST(bb.getElement(i)==bbb.getElement(i));
            TEST_TEST(cc.getElement(i)==ccc.getElement(i));
        }
        
        Test_Say("min():");
           TEST_TEST(bb.min()==bbb.min());        
        Test_Say("absMin():");
           TEST_TEST(bb.absMin()==bbb.absMin());
        Test_Say("max():");
           TEST_TEST(bb.max()==bbb.max());
        Test_Say("absMax():");
           TEST_TEST(bb.absMax()==bbb.absMax());
           
        Test_Say("lpNorm( N ):");
           TEST_TEST(aa.lpNorm(1)==aaa.lpNorm(1));
           TEST_TEST(aa.lpNorm(2)==aaa.lpNorm(2));
           TEST_TEST(aa.lpNorm(0.5)==aaa.lpNorm(0.5));
           TEST_TEST(aa.lpNorm(3)==aaa.lpNorm(3));
        Test_Say("sum():");
           TEST_TEST(aa.sum()==aaa.sum());

        Test_Say("differenceMax():");
           TEST_TEST(aa.differenceMax(bb)==aaa.differenceMax(bbb));
        Test_Say("differenceMin():");
           TEST_TEST(aa.differenceMin(bb)==aaa.differenceMin(bbb));
        Test_Say("differenceAbsMax():");
           TEST_TEST(aa.differenceAbsMax(bb)==aaa.differenceAbsMax(bbb));
        Test_Say("differenceAbsMin():");
           TEST_TEST(aa.differenceAbsMin(bb)==aaa.differenceAbsMin(bbb));
        Test_Say("differenceSum():");
           TEST_TEST(aa.differenceSum(bb)==aaa.differenceSum(bbb));
           
        ////
        Test_Say("differenceLpNorm( N ):");
           TEST_TEST(aa.differenceLpNorm(bb,1)==aaa.differenceLpNorm(bbb,1));
           TEST_TEST(aa.differenceLpNorm(bb,2)==aaa.differenceLpNorm(bbb,2));
           TEST_TEST(aa.differenceLpNorm(bb,0.5)==aaa.differenceLpNorm(bbb,0.5));
           TEST_TEST(aa.differenceLpNorm(bb,3)==aaa.differenceLpNorm(bbb,3));
        
        ////
        Test_Say("== :");
            TEST_TEST(aa==aaa);
        Test_Say("vct*0.5 :");
        aa*=0.5;
        aaa*=0.5;
            TEST_TEST(aa==aaa);
        
        Test_Say("scalarProduct :");
            TEST_TEST(aa.scalarProduct(bb) == aaa.scalarProduct(bbb));

        Test_Say("addVector :");
        aa.addVector(bb,2.0,3.0);
        aaa.addVector(bbb,2.0,3.0);
            TEST_TEST(aa==aaa);            
        aa.addVectors(bb,2.0,cc,1.0,-3.0);
        aaa.addVectors(bbb,2.0,ccc,1.0,-3.0);
            TEST_TEST(aa==aaa); 
            
        Test_Say("computeExclusivePrefixSum :");    
        aa.computeExclusivePrefixSum();
        aaa.computeExclusivePrefixSum();
            TEST_TEST(aa==aaa);             
        bb.computeExclusivePrefixSum(2,4);
        bbb.computeExclusivePrefixSum(2,4);
            TEST_TEST(bb==bbb); 
            
        Test_Say("computePrefixSum :");    
        cc.computePrefixSum();
        ccc.computePrefixSum();
            TEST_TEST(cc==ccc); 
        cc.computePrefixSum(2,4);
        ccc.computePrefixSum(2,4);
            TEST_TEST(cc==ccc); 	
#endif
	TEST_RESULT;	
		
    return 0;
}
