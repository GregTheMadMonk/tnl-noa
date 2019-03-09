#include <TNL/Containers/Multimaps/StaticEllpackIndexMultimap.h>

using namespace TNL;
using namespace TNL::Containers::Multimaps;

using IndexType = int;
using Device = Devices::Host;
using LocalIndexType = short;

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>

TEST( MultimapTest, TestTypedefs )
{
   using MultimapType = StaticEllpackIndexMultimap< 4, IndexType, Device, LocalIndexType >;
   const bool same_index = std::is_same< typename MultimapType::IndexType, IndexType >::value;
   ASSERT_TRUE( same_index );
   const bool same_device = std::is_same< typename MultimapType::DeviceType, Device >::value;
   ASSERT_TRUE( same_device );
   const bool same_localindex = std::is_same< typename MultimapType::LocalIndexType, LocalIndexType >::value;
   ASSERT_TRUE( same_localindex );
}

TEST( MultimapTest, TestSettingValues )
{
   using MultimapType = StaticEllpackIndexMultimap< 4, IndexType, Device, LocalIndexType >;

   const IndexType inputs = 10;
   const LocalIndexType allocatedValues = 4;

   MultimapType map;
   map.setKeysRange( inputs );
   ASSERT_EQ( map.getKeysRange(), inputs );
   map.allocate();

   for( IndexType i = 0; i < inputs; i++ ) {
      auto values = map.getValues( i );
      const auto constValues = ( (const MultimapType) map ).getValues( i );

      for( LocalIndexType o = 0; o < allocatedValues; o++ )
         values.setValue( o, i + o );

      for( LocalIndexType o = 0; o < allocatedValues; o++ ) {
         ASSERT_EQ( values.getValue( o ), i + o );
         ASSERT_EQ( values[ o ], i + o );
         ASSERT_EQ( constValues.getValue( o ), i + o );
         ASSERT_EQ( constValues[ o ], i + o );
      }

      for( LocalIndexType o = 0; o < allocatedValues; o++ )
         values[ o ] = i * o;

      for( LocalIndexType o = 0; o < allocatedValues; o++ ) {
         ASSERT_EQ( values.getValue( o ), i * o );
         ASSERT_EQ( values[ o ], i * o );
         ASSERT_EQ( constValues.getValue( o ), i * o );
         ASSERT_EQ( constValues[ o ], i * o );
      }
   }
}

TEST( MultimapTest, TestSaveAndLoad )
{
   using MultimapType = StaticEllpackIndexMultimap< 4, IndexType, Device, LocalIndexType >;

   const IndexType inputs = 10;
   const LocalIndexType allocatedValues = 4;

   MultimapType map, map2;
   map.setKeysRange( inputs );
   ASSERT_EQ( map.getKeysRange(), inputs );
   map.allocate();

   for( IndexType i = 0; i < inputs; i++ ) {
      auto values = map.getValues( i );
      for( LocalIndexType o = 0; o < allocatedValues; o++ )
         values.setValue( o, i + o );
   }

   map.save( "multimap-test.tnl" );
   map2.load( "multimap-test.tnl" );

   EXPECT_EQ( map, map2 );
   EXPECT_EQ( map.getKeysRange(), map2.getKeysRange() );

   for( IndexType i = 0; i < inputs; i++ ) {
      auto values = map.getValues( i );
      auto values2 = map2.getValues( i );

      for( LocalIndexType o = 0; o < allocatedValues; o++ ) {
         ASSERT_EQ( values[ o ], i + o );
         ASSERT_EQ( values2[ o ], i + o );
      }
   }
}
#endif

int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   return EXIT_FAILURE;
#endif
}
