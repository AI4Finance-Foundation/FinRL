/* TA-LIB Copyright (c) 1999-2007, Mario Fortier
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in
 *   the documentation and/or other materials provided with the
 *   distribution.
 *
 * - Neither name of author nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* List of contributors:
 *
 *  Initial  Name/description
 *  -------------------------------------------------------------------
 *  MF       Mario Fortier
 *
 *
 * Change history:
 *
 *  MMDDYY BY   Description
 *  -------------------------------------------------------------------
 *  112400 MF   First version.
 *
 */

/* Description:
 *     Regression test of Bollinger Bands (BBANDS).
 */

/**** Headers ****/
#include <stdio.h>
#include <string.h>

#include "ta_test_priv.h"
#include "ta_test_func.h"
#include "ta_utility.h"
#include "ta_memory.h"

/**** External functions declarations. ****/
/* None */

/**** External variables declarations. ****/
/* None */

/**** Global variables definitions.    ****/
/* None */

/**** Local declarations.              ****/
typedef struct
{
   TA_Integer doRangeTestFlag;

   TA_Integer startIdx;
   TA_Integer endIdx;

   TA_Integer    optInTimePeriod;
   TA_Real       optInNbDevUp;
   TA_Real       optInNbDevDn;
   TA_Integer    optInMethod_3;
   TA_Integer    compatibility;

   TA_RetCode expectedRetCode;

   TA_Integer expectedBegIdx;
   TA_Integer expectedNbElement;
   
   TA_Integer oneOfTheExpectedOutRealIndex0;
   TA_Real    oneOfTheExpectedOutReal0;

   TA_Integer oneOfTheExpectedOutRealIndex1;
   TA_Real    oneOfTheExpectedOutReal1;

   TA_Integer oneOfTheExpectedOutRealIndex2;
   TA_Real    oneOfTheExpectedOutReal2;

} TA_Test;

typedef struct
{
   const TA_Test *test;
   const TA_Real *close;
} TA_RangeTestParam;                                                                      

/**** Local functions declarations.    ****/
static ErrorNumber do_test( const TA_History *history,
                            const TA_Test *test );

/**** Local variables definitions.     ****/
static TA_Test tableTest[] =
{

   /****************************/
   /*   BBANDS - CLASSIC - EMA */
   /****************************/

   /* No multiplier */
   /* With upper band multiplier only. */
   /* With lower band multiplier only. */
   /* With identical upper/lower multiplier. */
   { 0, 0,  251, 20, 2.0, 2.0, TA_MAType_EMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,
     19, 252-19,
     13, 93.674,   /* Upper */
     13, 87.679,   /* Middle */
     13, 81.685 }, /* Lower */

   { 0, 0,  251, 20, 2.0, 2.0, TA_MAType_EMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,
     19, 252-19,
     0, 98.0734,   /* Upper */
     0, 92.8910,   /* Middle */
     0, 87.7086 }, /* Lower */
   /* With distinctive upper/lower multiplier. */

   /****************************/
   /*   BBANDS - CLASSIC - SMA */
   /****************************/
   /* No multiplier */
   /* With upper band multiplier only. */
   /* With lower band multiplier only. */
   /* With identical upper/lower multiplier. */
   { 1, 0,  251, 20, 2.0, 2.0, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,
     19, 252-19,
     0, 98.0734,   /* Upper */
     0, 92.8910,   /* Middle */
     0, 87.7086 }, /* Lower */
   /* With distinctive upper/lower multiplier. */

   
   /******************************/
   /*   BBANDS - METASTOCK - SMA */
   /******************************/

   /* No multiplier */
   /* With upper band multiplier only. */
   /* With lower band multiplier only. */

   /* With identical upper/lower multiplier. */
   { 1, 0,  251, 20, 2.0, 2.0, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     0, 98.0734,    /* Upper */
     0, 92.8910,    /* Middle */
     0, 87.7086  }, /* Lower */

   /* With distinctive upper/lower multiplier. */

   /******************************/
   /*   BBANDS - METASTOCK - EMA */
   /******************************/

   /* No multiplier */
   { 1, 0,  251, 20, 1.0, 1.0, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     0, 94.6914,   /* Upper  */
     0, 92.1002,   /* Middle */
     0, 89.5090 }, /* Lower  */
   { 0, 0,  251, 20, 1.0, 1.0, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     3, 94.0477,   /* Upper  */
     3, 90.7270,   /* Middle */
     3, 87.4063 }, /* Lower  */
   { 0, 0,  251, 20, 1.0, 1.0, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     252-20, 111.5415,   /* Upper  */
     252-20, 108.5265,   /* Middle */
     252-20, 105.5115 }, /* Lower  */

   /* With upper band multiplier only. */
   { 0, 0,  251, 20, 1.5, 1.0, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     0, 95.9870,   /* Upper */
     0, 92.1002,   /* Middle */
     0, 89.5090},  /* Lower */
   { 0, 0,  251, 20, 1.5, 1.0, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     3, 95.7080,  /* Upper */
     3, 90.7270,  /* Middle */
     3, 87.4063}, /* Lower */
   { 0, 0,  251, 20, 1.5, 1.0, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     252-20, 113.0490,   /* Upper */
     252-20, 108.5265,   /* Middle */
     252-20, 105.5115 }, /* Lower */

   /* With lower band multiplier only. */
   { 1, 0,  251, 20, 1.0, 1.5, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     0, 94.6914,   /* Upper */
     0, 92.1002,   /* Middle */
     0, 88.2134 }, /* Lower */
   { 0, 0,  251, 20, 1.0, 1.5, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     3, 94.0477,  /* Upper */
     3, 90.7270,  /* Middle */
     3, 85.7460}, /* Lower */
   { 0, 0,  251, 20, 1.0, 1.5, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     252-20, 111.5415,   /* Upper */
     252-20, 108.5265,   /* Middle */
     252-20, 104.0040},  /* Lower */

   /* With identical upper/lower multiplier. */
   { 0, 0,  251, 20, 2.0, 2.0, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     0, 97.2826,  /* Upper */
     0, 92.1002,  /* Middle */
     0, 86.9178}, /* Lower */
   { 0, 0,  251, 20, 2.0, 2.0, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     1, 97.2637,    /* Upper */
     1, 91.7454,    /* Middle */
     1, 86.2271}, /* Lower */
   { 0, 0,  251, 20, 2.0, 2.0, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     252-20, 114.5564,  /* Upper */
     252-20, 108.5265,  /* Middle */
     252-20, 102.4965}, /* Lower */
  
   /* With distinctive upper/lower multiplier. */
   { 0, 0,  251, 20, 2.0, 1.5, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     0, 97.2826,   /* Upper */
     0, 92.1002,   /* Middle */
     0, 88.2134 }, /* Lower */
   { 0, 0,  251, 20, 2.0, 1.5, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     3, 97.3684,    /* Upper */
     3, 90.7270,    /* Middle */
     3, 85.7460}, /* Lower */
   { 0, 0,  251, 20, 2.0, 1.5, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,
     19, 252-19,
     252-20, 114.5564, /* Upper */
     252-20, 108.5265, /* Middle */
     252-20, 104.0040} /* Lower */

};

#define NB_TEST (sizeof(tableTest)/sizeof(TA_Test))

/**** Global functions definitions.   ****/
ErrorNumber test_func_bbands( TA_History *history )
{
   unsigned int i;
   ErrorNumber retValue;

   for( i=0; i < NB_TEST; i++ )
   {

      if( (int)tableTest[i].expectedNbElement > (int)history->nbBars )
      {
         printf( "%s Failed Bad Parameter for Test #%d (%d,%d)\n", __FILE__,
                 i,
                 tableTest[i].expectedNbElement,
                 history->nbBars );
         return TA_TESTUTIL_TFRR_BAD_PARAM;
      }

      retValue = do_test( history, &tableTest[i] );
      if( retValue != 0 )
      {
         printf( "%s Failed Test #%d (Code=%d)\n", __FILE__, i, retValue );
         return retValue;
      }
   }

   /* All test succeed. */
   return TA_TEST_PASS; 
}

/**** Local functions definitions.     ****/
static TA_RetCode rangeTestFunction( TA_Integer    startIdx,
                                     TA_Integer    endIdx,
                                     TA_Real      *outputBuffer,
                                     TA_Integer   *outputBufferInt,
                                     TA_Integer   *outBegIdx,
                                     TA_Integer   *outNbElement,
                                     TA_Integer   *lookback,
                                     void         *opaqueData,
                                     unsigned int  outputNb,
                                     unsigned int *isOutputInteger )
{
  TA_RetCode retCode;
  TA_RangeTestParam *testParam;
  TA_Real *dummyBuffer1, *dummyBuffer2;
  TA_Real *out1, *out2, *out3;

  (void)outputBufferInt;

  *isOutputInteger = 0;

  testParam = (TA_RangeTestParam *)opaqueData;   

  dummyBuffer1 = TA_Malloc( ((endIdx-startIdx)+1)*sizeof(TA_Real));
  if( !dummyBuffer1 )
     return TA_ALLOC_ERR;

  dummyBuffer2 = TA_Malloc( ((endIdx-startIdx)+1)*sizeof(TA_Real));
  if( !dummyBuffer2 )
  {
     TA_Free(  dummyBuffer1 );
     return TA_ALLOC_ERR;
  }

  switch( outputNb )
  {
  case 0:
     out1 = outputBuffer;
     out2 = dummyBuffer1;
     out3 = dummyBuffer2;
     break;
  case 1:
     out2 = outputBuffer;
     out1 = dummyBuffer1;
     out3 = dummyBuffer2;
     break;
  case 2:
     out3 = outputBuffer;
     out2 = dummyBuffer1;
     out1 = dummyBuffer2;
     break;
  default:
     TA_Free(  dummyBuffer1 );
     TA_Free(  dummyBuffer2 );
     return TA_BAD_PARAM;
  }

   retCode = TA_BBANDS( startIdx,
                        endIdx,
                        testParam->close,
                        testParam->test->optInTimePeriod,
                        testParam->test->optInNbDevUp,
                        testParam->test->optInNbDevDn,
                        (TA_MAType)testParam->test->optInMethod_3,
                        outBegIdx, outNbElement,
                        out1, out2, out3 );

   *lookback = TA_BBANDS_Lookback( testParam->test->optInTimePeriod,
                                   testParam->test->optInNbDevUp,
                                   testParam->test->optInNbDevDn,
                                   (TA_MAType)testParam->test->optInMethod_3 );

   TA_Free(  dummyBuffer1 );
   TA_Free(  dummyBuffer2 );

   return retCode;
}


static ErrorNumber do_test( const TA_History *history,
                            const TA_Test *test )
{
   TA_RetCode retCode;
   ErrorNumber errNb;
   TA_Integer outBegIdx;
   TA_Integer outNbElement;
   TA_RangeTestParam testParam;

   retCode = TA_SetUnstablePeriod( TA_FUNC_UNST_EMA, 0 );
   if( retCode != TA_SUCCESS )
      return TA_TEST_TFRR_SETUNSTABLE_PERIOD_FAIL;

   /* Set to NAN all the elements of the gBuffers.  */
   clearAllBuffers();

   /* Build the input. */
   setInputBuffer( 0, history->close, history->nbBars );
   setInputBuffer( 1, history->close, history->nbBars );
   setInputBuffer( 2, history->close, history->nbBars );
   setInputBuffer( 3, history->close, history->nbBars );

   TA_SetCompatibility( (TA_Compatibility)test->compatibility );

   /* Make a simple first call. */
   retCode = TA_BBANDS( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        test->optInTimePeriod,
                        test->optInNbDevUp,
                        test->optInNbDevDn,
                        (TA_MAType)test->optInMethod_3,

                        &outBegIdx, &outNbElement,
                        gBuffer[0].out0, 
                        gBuffer[0].out1, 
                        gBuffer[0].out2 );

   errNb = checkDataSame( gBuffer[0].in, history->close, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[0].out0, 0 );
   CHECK_EXPECTED_VALUE( gBuffer[0].out1, 1 );
   CHECK_EXPECTED_VALUE( gBuffer[0].out2, 2 );

   outBegIdx = outNbElement = 0;

   /* Make another call where the input and the output are the
    * same buffer.
    */
   retCode = TA_BBANDS( test->startIdx,
                        test->endIdx,
                        gBuffer[1].in,
                        test->optInTimePeriod,
                        test->optInNbDevUp,
                        test->optInNbDevDn,
                        (TA_MAType)test->optInMethod_3,
                        &outBegIdx, &outNbElement,
                        gBuffer[1].in, gBuffer[1].out1, gBuffer[1].out2 );                        

   /* The previous call should have the same output
    * as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[0].out0, gBuffer[1].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[1].in,   0 );
   CHECK_EXPECTED_VALUE( gBuffer[1].out1, 1 );
   CHECK_EXPECTED_VALUE( gBuffer[1].out2, 2 );

   outBegIdx = outNbElement = 0;

   /* Make another call where the input and the output are the
    * same buffer.
    */
   retCode = TA_BBANDS( test->startIdx,
                        test->endIdx,
                        gBuffer[2].in,
                        test->optInTimePeriod,
                        test->optInNbDevUp,
                        test->optInNbDevDn,
                        (TA_MAType)test->optInMethod_3,
                        &outBegIdx, &outNbElement,
                        gBuffer[2].out1, 
                        gBuffer[2].in,
                        gBuffer[2].out2 );

   /* The previous call should have the same output
    * as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[1].out1, gBuffer[2].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[2].out1, 0 );
   CHECK_EXPECTED_VALUE( gBuffer[2].in,   1 );
   CHECK_EXPECTED_VALUE( gBuffer[2].out2, 2 );

   outBegIdx = outNbElement = 0;

   /* Make another call where the input and the output are the
    * same buffer.
    */
   retCode = TA_BBANDS( test->startIdx,
                        test->endIdx,
                        gBuffer[3].in,
                        test->optInTimePeriod,
                        test->optInNbDevUp,
                        test->optInNbDevDn,
                        (TA_MAType)test->optInMethod_3,
                        &outBegIdx, &outNbElement,
                        gBuffer[3].out0, 
                        gBuffer[3].out1,
                        gBuffer[3].in );

   /* The previous call should have the same output
    * as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[2].out2, gBuffer[3].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[3].out0, 0 );
   CHECK_EXPECTED_VALUE( gBuffer[3].out1, 1 );
   CHECK_EXPECTED_VALUE( gBuffer[3].in,   2 );

   /* Do a systematic test of most of the
    * possible startIdx/endIdx range.
    */
   testParam.test  = test;
   testParam.close = history->close;

   if( test->doRangeTestFlag )
   {
      if( test->optInMethod_3 == TA_MAType_EMA )
      {
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_EMA,
                              (void *)&testParam, 3, 0 );
      }
      else
      {
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_NONE,
                              (void *)&testParam, 3, 0 );
      }

      if( errNb != TA_TEST_PASS )
         return errNb;
   }

   return TA_TEST_PASS;
}
