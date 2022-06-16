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
 *  AA       Andrew Atkinson
 *
 * Change history:
 *
 *  MMDDYY BY   Description
 *  -------------------------------------------------------------------
 *  112400 MF   First version.
 *  020605 MF   Add regression test with inverted slow/fast period.
 *  020805 AA   Fix one of the TA_PPO call (wrong buffer was pass).
 */

/* Description:
 *     Regression test of APO(Absolute Price Oscillator).
 *     Regression test of PPO (Percentage Price Oscillator).
 */

/**** Headers ****/
#include <stdio.h>
#include <string.h>

#include "ta_test_priv.h"
#include "ta_test_func.h"
#include "ta_utility.h"

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

   TA_Integer doPercentage;

   TA_Integer startIdx;
   TA_Integer endIdx;

   TA_Integer optInFastPeriod; /* From 1 to 200 */
   TA_Integer optInSlowPeriod; /* From 1 to 200 */
   TA_Integer optInMethod_2;
   TA_Integer compatibility;

   TA_RetCode expectedRetCode;

   TA_Integer oneOfTheExpectedOutRealIndex;
   TA_Real    oneOfTheExpectedOutReal;


   TA_Integer expectedBegIdx;
   TA_Integer expectedNbElement;
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
   /**********************************/
   /*    APO TEST - SIMPLE - CLASSIC */
   /**********************************/
   { 1, 0, 0, 251, 26, 12, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,      0, -3.3124, 25,  252-25 }, /* First Value */
   { 1, 0, 0, 251, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,      0, -3.3124, 25,  252-25 }, /* First Value */
   { 0, 0, 0, 251, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,      1, -3.5876, 25,  252-25 },
   { 0, 0, 0, 251, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 252-26, -0.1667, 25,  252-25 }, /* Last Value */

   { 0, 0, 0,   1, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   0,        0,    0,  0 }, /* Out of range value */
   { 0, 0, 1,   1, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   0,        0,    0,  0 }, /* Out of range value */
   { 0, 0, 25,  25, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   0,  -3.3124,   25,  1 }, /* First/Last Value */
   { 0, 0, 250, 251, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   1,  -0.1667,  250,  2 }, /* Last  Value */

   /************************************/
   /*    APO TEST - SIMPLE - METASTOCK */
   /************************************/
   { 0, 0, 0, 251, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,      0, -3.3124, 25,  252-25 }, /* First Value */
   { 0, 0, 0, 251, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,      1, -3.5876, 25,  252-25 },
   { 0, 0, 0, 251, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 252-26, -0.1667, 25,  252-25 }, /* Last Value */

   { 0, 0, 0,   1, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,        0,    0,  0 }, /* Out of range value */
   { 0, 0, 1,   1, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,        0,    0,  0 }, /* Out of range value */
   { 0, 0, 25,  25, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  -3.3124,   25,  1 }, /* First/Last Value */
   { 0, 0, 250, 251, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1,  -0.1667,  250,  2 }, /* Last  Value */


   /***************************************/
   /*    APO TEST - EXPONENTIAL - CLASSIC */
   /***************************************/
   /* !!! To be done. */

   /*****************************************/
   /*    APO TEST - EXPONENTIAL - METASTOCK */
   /*****************************************/
   { 1, 0, 0, 251, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,      0, -2.4193, 25,  252-25 }, /* First Value */
   { 0, 0, 0, 251, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,      1, -2.4367, 25,  252-25 },
   { 0, 0, 0, 251, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 252-26, 0.90401, 25,  252-25 }, /* Last Value */

   { 0, 0, 0,   1, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,        0,    0,  0 }, /* Out of range value */
   { 0, 0, 1,   1, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,        0,    0,  0 }, /* Out of range value */
   { 0, 0, 25,  25, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  -2.4193,   25,  1 },
   { 0, 0, 250, 251, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1,  0.90401,  250,  2 }, /* Last  Value */

   { 0, 0, 251, 251, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  0.90401,  251,  1 },  /* Last  Value */
   { 0, 0, 25,  25, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  -2.4193,   25,  1 },  /* Just enough to calculate first. */
   { 0, 0, 26,  26, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  -2.4367,   26,  1 },  /* Just enough to calculate second. */

   /**********************************/
   /*    PPO TEST - SIMPLE - CLASSIC */
   /**********************************/
   { 1, 1, 0, 251, 2, 3, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   0,  1.10264, 2,  252-2 }, /* First Value */
   { 0, 1, 0, 251, 2, 3, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   1, -0.02813, 2,  252-2 },
   { 0, 1, 0, 251, 2, 3, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 249, -0.21191, 2,  252-2 }, /* Last Value */

   { 0, 1, 0,   1, 2, 3, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   0,        0,   0,  0 }, /* Out of range value */
   { 0, 1, 1,   1, 2, 3, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   0,        0,   0,  0 }, /* Out of range value */
   { 0, 1, 2,   2, 2, 3, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   0,  1.10264,   2,  1 }, /* First/Last Value */
   { 0, 1, 250, 251, 2, 3, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   1, -0.21191, 250,  2 }, /* Last  Value */

   /************************************/
   /*    PPO TEST - SIMPLE - METASTOCK */
   /************************************/
   { 0, 1, 0, 251, 3, 2, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  1.10264, 2,  252-2 }, /* First Value */
   { 0, 1, 0, 251, 2, 3, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1, -0.02813, 2,  252-2 },
   { 0, 1, 0, 251, 3, 2, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 249, -0.21191, 2,  252-2 }, /* Last Value */

   { 0, 1, 0,   1, 2, 3, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,        0,   0,  0 }, /* Out of range value */
   { 1, 1, 1,   1, 3, 2, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,        0,   0,  0 }, /* Out of range value */
   { 1, 1, 2,   2, 2, 3, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  1.10264,   2,  1 }, /* First/Last Value */
   { 0, 1, 250, 251, 3, 2, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1, -0.21191, 250,  2 }, /* Last  Value */

   { 0, 1, 0, 251, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,      0, -3.6393, 25,  252-25 }, /* First Value */
   { 0, 1, 0, 251, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,      1, -3.9534, 25,  252-25 },
   { 0, 1, 0, 251, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 252-26, -0.15281, 25,  252-25 }, /* Last Value */

   { 0, 1, 0,   1, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,        0,   0,  0 }, /* Out of range value */
   { 0, 1, 1,   1, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,        0,   0,  0 }, /* Out of range value */
   { 0, 1, 25,  25, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0, -3.6393,   25,  1 }, /* First/Last Value */
   { 0, 1, 250, 251, 12, 26, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1, -0.15281, 250,  2 }, /* Last  Value */

   /***************************************/
   /*    PPO TEST - EXPONENTIAL - CLASSIC */
   /***************************************/
   /* !!! To be done. */

   /*****************************************/
   /*    PPO TEST - EXPONENTIAL - METASTOCK */
   /*****************************************/
   { 1, 1, 0, 251, 26, 12, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,      0, -2.7083, 25,  252-25 }, /* First Value */
   { 0, 1, 0, 251, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,      1, -2.7390, 25,  252-25 },
   { 0, 1, 0, 251, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 252-26, 0.83644, 25,  252-25 }, /* Last Value */

   { 0, 1, 0,   1, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,        0,    0,  0 }, /* Out of range value */
   { 0, 1, 1,   1, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,        0,    0,  0 }, /* Out of range value */
   { 0, 1, 25,  25, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,   -2.7083,   25,  1 },
   { 0, 1, 250, 251, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1,   0.83644,  250,  2 }, /* Last  Value */

   { 0, 1, 251, 251, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  0.83644,  251,  1 },  /* Last  Value */
   { 0, 1, 25,  25, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  -2.7083,   25,  1 },  /* Just enough to calculate first. */
   { 0, 1, 26,  26, 12, 26, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  -2.7390,   26,  1 },  /* Just enough to calculate second. */
};

#define NB_TEST (sizeof(tableTest)/sizeof(TA_Test))

/**** Global functions definitions.   ****/
ErrorNumber test_func_po( TA_History *history )
{
   unsigned int i;
   ErrorNumber retValue;

   for( i=0; i < NB_TEST; i++ )
   {

      if( (int)tableTest[i].expectedNbElement > (int)history->nbBars )
      {
         printf( "TA_APO/TA_PPO Failed Bad Parameter for Test #%d (%d,%d)\n",
                 i, tableTest[i].expectedNbElement, history->nbBars );
         return TA_TESTUTIL_TFRR_BAD_PARAM;
      }

      retValue = do_test( history, &tableTest[i] );
      if( retValue != 0 )
      {
         printf( "TA_APO/TA_PPO Failed Test #%d (Code=%d)\n", i, retValue );
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
                                     unsigned  int outputNb,
                                     unsigned int *isOutputInteger )      
{
   TA_RetCode retCode;
   TA_RangeTestParam *testParam;

   (void)outputNb;
   (void)outputBufferInt;
  
   *isOutputInteger = 0;

   testParam = (TA_RangeTestParam *)opaqueData;   

   if( testParam->test->doPercentage )
   {
      retCode = TA_PPO( startIdx,
                        endIdx,
                        testParam->close,
                        testParam->test->optInFastPeriod,
                        testParam->test->optInSlowPeriod,
                        (TA_MAType)testParam->test->optInMethod_2,
                        outBegIdx,
                        outNbElement,
                        outputBuffer );

     *lookback = TA_PPO_Lookback( testParam->test->optInFastPeriod,
                      testParam->test->optInSlowPeriod,
                      (TA_MAType)testParam->test->optInMethod_2 );                      
   }
   else
   {
      retCode = TA_APO( startIdx,
                        endIdx,
                        testParam->close,
                        testParam->test->optInFastPeriod,
                        testParam->test->optInSlowPeriod,
                        (TA_MAType)testParam->test->optInMethod_2,
                        outBegIdx,
                        outNbElement,
                        outputBuffer );


     *lookback = TA_APO_Lookback( testParam->test->optInFastPeriod,
                      testParam->test->optInSlowPeriod,
                      (TA_MAType)testParam->test->optInMethod_2 );
   }

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

   /* Set to NAN all the elements of the gBuffers.  */
   clearAllBuffers();

   TA_SetCompatibility( (TA_Compatibility)test->compatibility );

   /* Build the input. */
   setInputBuffer( 0, history->close, history->nbBars );
   setInputBuffer( 1, history->close, history->nbBars );
   
   TA_SetUnstablePeriod( TA_FUNC_UNST_EMA, 0 );

   /* Make a simple first call. */
   if( test->doPercentage )
   {
      retCode = TA_PPO( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        test->optInFastPeriod,
                        test->optInSlowPeriod,
                        (TA_MAType)test->optInMethod_2,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[0].out0 );
   }
   else
   {
      retCode = TA_APO( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        test->optInFastPeriod,
                        test->optInSlowPeriod,
                        (TA_MAType)test->optInMethod_2,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[0].out0 );
   }

   errNb = checkDataSame( gBuffer[0].in, history->close, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   errNb = checkExpectedValue( gBuffer[0].out0, 
                               retCode, test->expectedRetCode,
                               outBegIdx, test->expectedBegIdx,
                               outNbElement, test->expectedNbElement,
                               test->oneOfTheExpectedOutReal,
                               test->oneOfTheExpectedOutRealIndex );   
   if( errNb != TA_TEST_PASS )
      return errNb;

   outBegIdx = outNbElement = 0;

   /* Make another call where the input and the output are the
    * same buffer.
    */
   if( test->doPercentage )
   {
      retCode = TA_PPO( test->startIdx,
                        test->endIdx,
                        gBuffer[1].in,
                        test->optInFastPeriod,
                        test->optInSlowPeriod,
                        (TA_MAType)test->optInMethod_2,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[1].in );
   }
   else
   {
      retCode = TA_APO( test->startIdx,
                        test->endIdx,
                        gBuffer[1].in,
                        test->optInFastPeriod,
                        test->optInSlowPeriod,
                        (TA_MAType)test->optInMethod_2,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[1].in );
   }

   /* The previous call should have the same output
    * as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[0].out0, gBuffer[1].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   errNb = checkExpectedValue( gBuffer[1].in, 
                               retCode, test->expectedRetCode,
                               outBegIdx, test->expectedBegIdx,
                               outNbElement, test->expectedNbElement,
                               test->oneOfTheExpectedOutReal,
                               test->oneOfTheExpectedOutRealIndex );   
   if( errNb != TA_TEST_PASS )
      return errNb;

   /* Do a systematic test of most of the
    * possible startIdx/endIdx range.
    */
   testParam.test  = test;
   testParam.close = history->close;

   if( test->doRangeTestFlag )
   {

      if( test->optInMethod_2 == TA_MAType_EMA )
      {
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_EMA,
                              (void *)&testParam, 1, 0 );
         if( errNb != TA_TEST_PASS )
            return errNb;
      }
      else
      {
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_NONE,
                              (void *)&testParam, 1, 0 );
         if( errNb != TA_TEST_PASS )
            return errNb;
      }
   }

   return TA_TEST_PASS;
}

