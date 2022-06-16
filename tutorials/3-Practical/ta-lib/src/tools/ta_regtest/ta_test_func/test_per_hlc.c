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
 *  061904 MF   Add test to detect cummulative errors in CCI algorithm 
 *              when some values were close to zero (epsilon).
 *  021106 MF   Add tests for ULTOSC.
 *  042206 MF   Add tests for NATR
 *              
 */

/* Description:
 *
 *     Test functions which have the following
 *     characterisic: 
 *      - the input is high,low and close.
 *     
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
typedef enum {
TA_CCI_TEST,
TA_WILLR_TEST,
TA_ULTOSC_TEST,
TA_NATR_TEST
} TA_TestId;

typedef struct
{
   TA_Integer doRangeTestFlag;

   TA_TestId  theFunction;

   TA_Integer startIdx;
   TA_Integer endIdx;

   TA_Integer optInTimePeriod1;
   TA_Integer optInTimePeriod2;
   TA_Integer optInTimePeriod3;
   
   TA_RetCode expectedRetCode;

   TA_Integer oneOfTheExpectedOutRealIndex0;
   TA_Real    oneOfTheExpectedOutReal0;

   TA_Integer expectedBegIdx;
   TA_Integer expectedNbElement;
} TA_Test;

typedef struct
{
   const TA_Test *test;
   const TA_Real *high;
   const TA_Real *low;
   const TA_Real *close;
} TA_RangeTestParam;

/**** Local functions declarations.    ****/
static ErrorNumber do_test( const TA_History *history,
                            const TA_Test *test );

/**** Local variables definitions.     ****/

static TA_Test tableTest[] =
{
   /****************/
   /* NATR TEST    */
   /****************/
   /* TODO Analyze further why NATR requires a very large unstable period.
    * for now, just disable range testing.
    */
   { 0, TA_NATR_TEST, 0, 251, 14, 0, 0, TA_SUCCESS,       0,  3.9321, 14,  252-14 },
   { 0, TA_NATR_TEST, 0, 251, 14, 0, 0, TA_SUCCESS,       1,  3.7576, 14,  252-14 },
   { 0, TA_NATR_TEST, 0, 251, 14, 0, 0, TA_SUCCESS,  252-15,  3.0229, 14,  252-14 },

   /****************/
   /* ULTOSC TEST  */
   /****************/
   { 0, TA_ULTOSC_TEST, 0, 251, 7, 14, 28, TA_SUCCESS,       0,   47.1713, 28,  252-28 },
   { 0, TA_ULTOSC_TEST, 0, 251, 7, 14, 28, TA_SUCCESS,       1,   46.2802, 28,  252-28 },
   { 1, TA_ULTOSC_TEST, 0, 251, 7, 14, 28, TA_SUCCESS,  252-29,   40.0854, 28,  252-28 },


   /****************/
   /* WILLR TEST   */
   /****************/
   { 0, TA_WILLR_TEST, 13, 251, 14, 0, 0, TA_SUCCESS,   1,   -66.9903,  13,  252-13 }, /* First Value */
   { 1, TA_WILLR_TEST,  0, 251, 14, 0, 0, TA_SUCCESS,   0,   -90.1943,  13,  252-13 },
   { 0, TA_WILLR_TEST,  0, 251, 14, 0, 0, TA_SUCCESS, 112,        0.0,  13,  252-13 },

   { 0, TA_WILLR_TEST,  24, 24, 14, 0, 0, TA_SUCCESS, 0,    -89.2857,  24,  1 },
   { 0, TA_WILLR_TEST,  25, 25, 14, 0, 0, TA_SUCCESS, 0,    -97.2602,  25,  1 },
   { 0, TA_WILLR_TEST,  26, 26, 14, 0, 0, TA_SUCCESS, 0,    -71.5482,  26,  1 },

   { 0, TA_WILLR_TEST, 251, 251, 14, 0, 0, TA_SUCCESS,      0,    -59.1515, 251,  1 },
   { 0, TA_WILLR_TEST,  14,  251, 14, 0, 0, TA_SUCCESS, 252-15,   -59.1515, 14,  252-14 },

   /****************/
   /*   CCI TEST  */
   /****************/

   /* The following two should always be identical. */
   { 0, TA_CCI_TEST, 186,187,  2, 0, 0, TA_SUCCESS,   1, 0.0, 186,  2 },
   { 0, TA_CCI_TEST, 187,187,  2, 0, 0, TA_SUCCESS,   0, 0.0, 187,  1 },
 
   /* Test period 2, 5 and 11 */
   { 0, TA_CCI_TEST, 0, 251,  2, 0, 0, TA_SUCCESS,  0, 66.666, 1,  252-1 },
   { 1, TA_CCI_TEST, 0, 251,  5, 0, 0, TA_SUCCESS,  0, 18.857, 4,  252-4 },

   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS,  0,   87.927,  10,  252-10 }, /* First Value */
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS,  1,   180.005, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS,  2,  143.5190963, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS,  3,  -113.8669783, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS,  4,  -111.064497, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS,  5,  -26.77393309, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS,  6,  -70.77933765, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS,  7,  -83.15662884, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS,  8,  -41.14421073, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS,  9,  -49.63059589, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 10,  -86.45142995, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 11,  -105.6275799, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 12,  -157.698269, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 13,  -190.5251436, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 14,  -142.8364298, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 15,  -122.4448056, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 16,  -79.95100041, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 17,  22.03829204, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 18,  7.765575065, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 19,  32.38905945, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 20,  -0.005587727, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 21,  43.84607294, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 22,  40.35152301, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 23,  92.89237535, 10,  252-10 },
   { 0, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 24,  113.4778681, 10,  252-10 },
   { 1, TA_CCI_TEST, 0, 251, 11, 0, 0, TA_SUCCESS, 252-11,  -169.65514, 10,  252-10 }, /* Last Value */
};

#define NB_TEST (sizeof(tableTest)/sizeof(TA_Test))

/**** Global functions definitions.   ****/
ErrorNumber test_func_per_hlc( TA_History *history )
{
   unsigned int i;
   ErrorNumber retValue;

   /* Re-initialize all the unstable period to zero. */
   TA_SetUnstablePeriod( TA_FUNC_UNST_ALL, 0 );

   for( i=0; i < NB_TEST; i++ )
   {
      if( (int)tableTest[i].expectedNbElement > (int)history->nbBars )
      {
         printf( "Failed Bad Parameter for Test #%d (%d,%d)\n",
                 i, tableTest[i].expectedNbElement, history->nbBars );
         return TA_TESTUTIL_TFRR_BAD_PARAM;
      }

      retValue = do_test( history, &tableTest[i] );
      if( retValue != 0 )
      {
         printf( "Failed Test #%d (Code=%d)\n", i, retValue );
         return retValue;
      }
   }

   /* Re-initialize all the unstable period to zero. */
   TA_SetUnstablePeriod( TA_FUNC_UNST_ALL, 0 );

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

   (void)outputNb;
   (void)outputBufferInt;
  
   *isOutputInteger = 0;

   testParam = (TA_RangeTestParam *)opaqueData;   

   switch( testParam->test->theFunction )
   {
   case TA_NATR_TEST:
      retCode = TA_NATR( startIdx,
                         endIdx,
                         testParam->high,
                         testParam->low,
                         testParam->close,
                         testParam->test->optInTimePeriod1,
                         outBegIdx,
                         outNbElement,
                         outputBuffer );
      *lookback = TA_NATR_Lookback( testParam->test->optInTimePeriod1 );
      break;

   case TA_CCI_TEST:
      retCode = TA_CCI( startIdx,
                        endIdx,
                        testParam->high,
                        testParam->low,
                        testParam->close,
                        testParam->test->optInTimePeriod1,
                        outBegIdx,
                        outNbElement,
                        outputBuffer );
      *lookback = TA_CCI_Lookback( testParam->test->optInTimePeriod1 );
      break;
   case TA_WILLR_TEST:
      retCode = TA_WILLR( startIdx,
                          endIdx,
                          testParam->high,
                          testParam->low,
                          testParam->close,
                          testParam->test->optInTimePeriod1,
                          outBegIdx,
                          outNbElement,
                          outputBuffer );
      *lookback = TA_WILLR_Lookback( testParam->test->optInTimePeriod1 );
      break;

   case TA_ULTOSC_TEST:
      retCode = TA_ULTOSC( startIdx,
                           endIdx,
                           testParam->high,
                           testParam->low,
                           testParam->close,
                           testParam->test->optInTimePeriod1,
                           testParam->test->optInTimePeriod2,
                           testParam->test->optInTimePeriod3,
                           outBegIdx,
                           outNbElement,
                           outputBuffer );
      *lookback = TA_ULTOSC_Lookback( testParam->test->optInTimePeriod1,
                                      testParam->test->optInTimePeriod2,
                                      testParam->test->optInTimePeriod3 );
      break;

   default:
      retCode = TA_INTERNAL_ERROR(132);
   }

   return retCode;
}

static TA_RetCode do_call( const TA_Test *test,
                            const double high[],
                            const double low[],
                            const double close[],
                            int *outBegIdx,
                            int *outNbElement,
                            double output[] )
{
   TA_RetCode retCode;

   switch( test->theFunction )
   {
   case TA_NATR_TEST:
      retCode = TA_NATR( test->startIdx,
                         test->endIdx,
                         high, low, close,
                         test->optInTimePeriod1,
                         outBegIdx,
                         outNbElement,
                         output );
      break;

   case TA_CCI_TEST:
      retCode = TA_CCI( test->startIdx,
                        test->endIdx,
                        high, low, close,
                        test->optInTimePeriod1,
                        outBegIdx,
                        outNbElement,
                        output );
      break;

   case TA_WILLR_TEST:
      retCode = TA_WILLR( test->startIdx,
                          test->endIdx,
                          high, low, close,
                          test->optInTimePeriod1,
                          outBegIdx,
                          outNbElement,
                          output );
      break;

   case TA_ULTOSC_TEST:
      retCode = TA_ULTOSC( test->startIdx,
                           test->endIdx,
                           high, low, close,
                           test->optInTimePeriod1,
                           test->optInTimePeriod2,
                           test->optInTimePeriod3,
                           outBegIdx,
                           outNbElement,
                           output );
      break;

   default:
      retCode = TA_INTERNAL_ERROR(133);
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

   /* Build the input. */
   setInputBuffer( 0, history->high,  history->nbBars );
   setInputBuffer( 1, history->low,   history->nbBars );
   setInputBuffer( 2, history->close, history->nbBars );
      
   /* Make a simple first call. */
   retCode = do_call( test,
                      gBuffer[0].in,
                      gBuffer[1].in,
                      gBuffer[2].in,                  
                      &outBegIdx,
                      &outNbElement,
                      gBuffer[0].out0 );

   /* Check that the input were preserved. */
   errNb = checkDataSame( gBuffer[0].in, history->high,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[1].in, history->low, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[2].in, history->close,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[0].out0, 0 );

   outBegIdx = outNbElement = 0;

   /* Make another call where the input and the output are the
    * same buffer.
    */
   retCode = do_call( test,
                      gBuffer[0].in,
                      gBuffer[1].in,
                      gBuffer[2].in,                  
                      &outBegIdx,
                      &outNbElement,
                      gBuffer[0].in );

   /* Check that the input were preserved. */
   errNb = checkDataSame( gBuffer[1].in, history->low, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[2].in, history->close,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   /* The previous call should have the same output as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[0].out0, gBuffer[0].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[0].in, 0 );
   setInputBuffer( 0, history->high,  history->nbBars );

   /* Make another call where the input and the output are the
    * same buffer.
    */
   retCode = do_call( test,
                      gBuffer[0].in,
                      gBuffer[1].in,
                      gBuffer[2].in,                  
                      &outBegIdx,
                      &outNbElement,
                      gBuffer[1].in );

   /* Check that the input were preserved. */
   errNb = checkDataSame( gBuffer[0].in, history->high,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[2].in, history->close,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   /* The previous call should have the same output as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[0].out0, gBuffer[1].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[1].in, 0 );
   setInputBuffer( 1, history->low,   history->nbBars );

   /* Make another call where the input and the output are the
    * same buffer.
    */
   retCode = do_call( test,
                      gBuffer[0].in,
                      gBuffer[1].in,
                      gBuffer[2].in,                  
                      &outBegIdx,
                      &outNbElement,
                      gBuffer[2].in );

   /* Check that the input were preserved. */
   errNb = checkDataSame( gBuffer[0].in, history->high,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[1].in, history->low, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   /* The previous call should have the same output as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[0].out0, gBuffer[2].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[2].in, 0 );
   setInputBuffer( 2, history->close, history->nbBars );

   /* Do a systematic test of most of the
    * possible startIdx/endIdx range.
    */
   testParam.test  = test;
   testParam.high  = history->high;
   testParam.low   = history->low;
   testParam.close = history->close;

   if( test->doRangeTestFlag )
   {
      switch( test->theFunction )
      {
      case TA_NATR_TEST:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_NATR,
                              (void *)&testParam, 1, 0 );
         break;
      default:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_NONE,
                              (void *)&testParam, 1, 0 );
         break;
      }

      if( errNb != TA_TEST_PASS )
         return errNb;
   }

   return TA_TEST_PASS;
}

