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
 *  020203 MF   First version.
 *  122506 MF   Add TA_BETA tests.
 */

/* Description:
 *
 *     Test functions which have the following characterisic: 
 *      - two inputs are needed (high and low are used here).
 *      - has zero or one parameter being a period.
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
typedef enum {
TA_AROON_UP_TEST,
TA_AROON_DOWN_TEST,
TA_AROONOSC_TEST,
TA_CORREL_TEST,
TA_BETA_TEST
} TA_TestId;

typedef struct
{
   TA_Integer doRangeTestFlag;

   TA_TestId  theFunction;

   TA_Integer startIdx;
   TA_Integer endIdx;
   TA_Integer optInTimePeriod;
   
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
} TA_RangeTestParam;

/**** Local functions declarations.    ****/
static ErrorNumber do_test( const TA_History *history,
                            const TA_Test *test );

/**** Local variables definitions.     ****/

static TA_Test tableTest[] =
{
   /*****************/
   /* BETA TEST     */
   /*****************/

   /* Uncomment following to enable tons of tests. Replace 999.99 with the first
    * value you are expecting.
    */

   { 1, TA_BETA_TEST,  0, 251, 5, TA_SUCCESS,      0, 0.62907,  5,  252-5 },
   { 0, TA_BETA_TEST,  0, 251, 5, TA_SUCCESS,      1, 0.83604,  5,  252-5 },

   /*****************/
   /* CORREL TEST   */
   /*****************/
   { 1, TA_CORREL_TEST,  0, 251, 20, TA_SUCCESS,      0, 0.9401569,  19,  252-19 }, /* First Value */
   { 0, TA_CORREL_TEST,  0, 251, 20, TA_SUCCESS,      1, 0.9471812,  19,  252-19 },
   { 0, TA_CORREL_TEST,  0, 251, 20, TA_SUCCESS, 252-20, 0.8866901,  19,  252-19 }, /* Last Value */

   
   /*******************/
   /* AROON UP TEST   */
   /*******************/
   { 1, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS,  0, 78.571,  14,  252-14 }, /* First Value */
   { 0, TA_AROON_UP_TEST,  0, 251, 14, TA_SUCCESS,  1, 71.429,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS,  2, 64.2857,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS,  3, 57.143,  14,  252-14 },
   { 0, TA_AROON_UP_TEST,  0, 251, 14, TA_SUCCESS,  4, 50.000,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS,  5, 42.857,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS,  6, 35.714,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS,  7, 28.571,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS,  8, 21.429,  14,  252-14 },
   { 0, TA_AROON_UP_TEST,  0, 251, 14, TA_SUCCESS,  9, 14.286,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS, 10, 7.1429,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS, 11, 0.0000,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS, 12, 0.0000,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS, 13, 21.429,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS, 14, 14.286,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS, 15, 7.1429,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS, 16, 0.0000,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS, 17, 14.286,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS, 20, 0.00,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS, 21, 92.857,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS, 27, 50.000,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS, 28, 42.857,  14,  252-14 },
   { 0, TA_AROON_UP_TEST, 13, 251, 14, TA_SUCCESS, 29,100.000,  14,  252-14 },
   { 0, TA_AROON_UP_TEST,  1, 251, 14, TA_SUCCESS, 252-16, 0.0,  14,  252-14 },
   { 0, TA_AROON_UP_TEST,  0, 251, 14, TA_SUCCESS, 252-15, 7.1429,  14,  252-14 }, /* Last Value */

   /*******************/
   /* AROON DOWN TEST */
   /*******************/
   { 1, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS,  0, 100.0,  14,  252-14 }, /* First Value */
   { 0, TA_AROON_DOWN_TEST,  0, 251, 14, TA_SUCCESS,  1, 92.857,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS,  2, 85.714,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS,  3, 78.571,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST,  0, 251, 14, TA_SUCCESS,  4, 71.429,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS,  5, 64.286,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS,  6, 57.143,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS,  7,100.000,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS,  8,100.000,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST,  0, 251, 14, TA_SUCCESS,  9,100.000,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 10,100.000,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 11,100.000,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 12, 92.857,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 13, 85.714,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 14, 78.571,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 15, 71.429,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 16, 64.286,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 17, 57.143,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 18, 50.000,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 19, 42.857,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 20, 35.714,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST,  0, 251, 14, TA_SUCCESS, 21, 28.571,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST,  0, 251, 14, TA_SUCCESS, 22, 21.429,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST,  0, 251, 14, TA_SUCCESS, 23, 14.286,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 24, 7.1429,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 25, 0.0,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 26, 0.0,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 27, 92.857,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST, 13, 251, 14, TA_SUCCESS, 28, 85.714,  14,  252-14 },

   { 0, TA_AROON_DOWN_TEST,  4, 251, 14, TA_SUCCESS, 252-16, 28.571,  14,  252-14 },
   { 0, TA_AROON_DOWN_TEST,  0, 251, 14, TA_SUCCESS, 252-15, 21.429,  14,  252-14 }, /* Last Value */

   /******************/
   /* AROON OSC TEST */
   /******************/
   { 0, TA_AROONOSC_TEST, 13, 251, 14, TA_SUCCESS,  0, -21.4285,  14,  252-14 }, /* First Value */
   { 0, TA_AROONOSC_TEST, 13, 251, 14, TA_SUCCESS,  6, -21.4285,  14,  252-14 },
   { 0, TA_AROONOSC_TEST, 13, 251, 14, TA_SUCCESS,  7, -71.4285,  14,  252-14 },
   { 0, TA_AROONOSC_TEST, 0, 251, 14, TA_SUCCESS, 252-16, -28.5714,  14,  252-14 },
   { 0, TA_AROONOSC_TEST, 0, 251, 14, TA_SUCCESS, 252-15, -14.28571,  14,  252-14 }, /* Last Value */

};

#define NB_TEST (sizeof(tableTest)/sizeof(TA_Test))

/**** Global functions definitions.   ****/
ErrorNumber test_func_per_hl( TA_History *history )
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
   double *dummyBuffer;

   (void)outputNb;
   (void)outputBufferInt;
  
   *isOutputInteger = 0;

   testParam = (TA_RangeTestParam *)opaqueData;   

   /* Allocate a buffer for the output who is going
    * to be ignored (make it slightly larger to play
    * safe)
    */
   dummyBuffer = TA_Malloc( sizeof(double) * (endIdx-startIdx+100) );
   switch( testParam->test->theFunction )
   {
   case TA_AROON_UP_TEST:
      retCode = TA_AROON( startIdx,
                          endIdx,
                          testParam->high,
                          testParam->low,
                          testParam->test->optInTimePeriod,
                          outBegIdx,
                          outNbElement,                          
                          &dummyBuffer[20],
                          outputBuffer );

      *lookback = TA_AROON_Lookback( testParam->test->optInTimePeriod );
      break;
   case TA_AROON_DOWN_TEST:
      retCode = TA_AROON( startIdx,
                          endIdx,
                          testParam->high,
                          testParam->low,
                          testParam->test->optInTimePeriod,
                          outBegIdx,
                          outNbElement,
                          outputBuffer,
                          &dummyBuffer[20]
                        );
      *lookback = TA_AROON_Lookback( testParam->test->optInTimePeriod );
      break;
   case TA_AROONOSC_TEST:
      retCode = TA_AROONOSC( startIdx,
                             endIdx,
                             testParam->high,
                             testParam->low,
                             testParam->test->optInTimePeriod,
                             outBegIdx,
                             outNbElement,
                             outputBuffer );
      *lookback = TA_AROONOSC_Lookback( testParam->test->optInTimePeriod );
      break;
   case TA_CORREL_TEST:
      retCode = TA_CORREL( startIdx,
                           endIdx,
                           testParam->high,
                           testParam->low,
                           testParam->test->optInTimePeriod,
                           outBegIdx,
                           outNbElement,
                           outputBuffer );
      *lookback = TA_CORREL_Lookback( testParam->test->optInTimePeriod );
      break;

   case TA_BETA_TEST:
      retCode = TA_BETA( startIdx,
                         endIdx,
                         testParam->high,
                         testParam->low,
                         testParam->test->optInTimePeriod, /* time period */
                         outBegIdx,
                         outNbElement,
                         outputBuffer );
      *lookback = TA_BETA_Lookback(testParam->test->optInTimePeriod);
      break;

   default:
      retCode = TA_INTERNAL_ERROR(132);
   }

   TA_Free( dummyBuffer );

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

   /* Make a simple first call. */
   switch( test->theFunction )
   {
   case TA_AROON_UP_TEST:
      retCode = TA_AROON( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          test->optInTimePeriod,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[1].out0,
                          gBuffer[0].out0
                        );
      break;

   case TA_AROON_DOWN_TEST:      
      retCode = TA_AROON( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          test->optInTimePeriod,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[0].out0,
                          gBuffer[1].out0
                        );
      break;

   case TA_AROONOSC_TEST:
      retCode = TA_AROONOSC( test->startIdx,
                             test->endIdx,
                             gBuffer[0].in,
                             gBuffer[1].in,
                             test->optInTimePeriod,
                             &outBegIdx,
                             &outNbElement,
                             gBuffer[0].out0
                           );
      break;

   case TA_CORREL_TEST:
      retCode = TA_CORREL( test->startIdx,
                           test->endIdx,
                           gBuffer[0].in,
                           gBuffer[1].in,
                           test->optInTimePeriod,
                           &outBegIdx,
                           &outNbElement,
                           gBuffer[0].out0
                         );
      break;

   case TA_BETA_TEST:
      retCode = TA_BETA( test->startIdx,
                         test->endIdx,
                         gBuffer[0].in,
                         gBuffer[1].in,
                         test->optInTimePeriod,
                         &outBegIdx,
                         &outNbElement,
                         gBuffer[0].out0
                         );
      break;

   default:
      retCode = TA_INTERNAL_ERROR(133);
   }

   /* Check that the input were preserved. */
   errNb = checkDataSame( gBuffer[0].in, history->high,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[1].in, history->low, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[0].out0, 0 );

   outBegIdx = outNbElement = 0;

   /* Make another call where one of the input and one of the output 
    * are the same buffer.
    */
   switch( test->theFunction )
   {
   case TA_AROON_UP_TEST:      
      retCode = TA_AROON( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          test->optInTimePeriod,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[1].out1,
                          gBuffer[0].in
                        );
      break;

   case TA_AROON_DOWN_TEST:      
      retCode = TA_AROON( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          test->optInTimePeriod,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[0].in,
                          gBuffer[1].out1
                        );
      break;

   case TA_AROONOSC_TEST:
      retCode = TA_AROONOSC( test->startIdx,
                             test->endIdx,
                             gBuffer[0].in,
                             gBuffer[1].in,
                             test->optInTimePeriod,
                             &outBegIdx,
                             &outNbElement,
                             gBuffer[0].in
                           );
      break;

   case TA_CORREL_TEST:
      retCode = TA_CORREL( test->startIdx,
                           test->endIdx,
                           gBuffer[0].in,
                           gBuffer[1].in,
                           test->optInTimePeriod,
                           &outBegIdx,
                           &outNbElement,
                           gBuffer[0].in
                         );
      break;

   case TA_BETA_TEST:
      retCode = TA_BETA( test->startIdx,
                         test->endIdx,
                         gBuffer[0].in,
                         gBuffer[1].in,
                         test->optInTimePeriod,
                         &outBegIdx,
                         &outNbElement,
                         gBuffer[0].in
                         );
      break;

   default:
      retCode = TA_INTERNAL_ERROR(134);
   }

   /* Check that the other input was preserved. */
   errNb = checkDataSame( gBuffer[1].in, history->low, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   /* The previous call should have the same output
    * as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[0].out0, gBuffer[0].in );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkSameContent( gBuffer[1].out1, gBuffer[1].out0 );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[0].in, 0 );
   setInputBuffer( 0, history->high,  history->nbBars );
   setInputBuffer( 1, history->low,   history->nbBars );

   /* Make another call where the input and the output are the
    * same buffer.
    */
   switch( test->theFunction )
   {
   case TA_AROON_UP_TEST:      
      retCode = TA_AROON( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          test->optInTimePeriod,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[1].out2,
                          gBuffer[1].in
                        );
      break;

   case TA_AROON_DOWN_TEST:      
      retCode = TA_AROON( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          test->optInTimePeriod,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[1].in,
                          gBuffer[1].out2
                        );
      break;

   case TA_AROONOSC_TEST:
      retCode = TA_AROONOSC( test->startIdx,
                             test->endIdx,
                             gBuffer[0].in,
                             gBuffer[1].in,
                             test->optInTimePeriod,
                             &outBegIdx,
                             &outNbElement,
                             gBuffer[1].in
                           );
      break;

   case TA_CORREL_TEST:
      retCode = TA_CORREL( test->startIdx,
                           test->endIdx,
                           gBuffer[0].in,
                           gBuffer[1].in,
                           test->optInTimePeriod,
                           &outBegIdx,
                           &outNbElement,
                           gBuffer[1].in
                         );
      break;

   case TA_BETA_TEST:
      retCode = TA_BETA( test->startIdx,
                         test->endIdx,
                         gBuffer[0].in,
                         gBuffer[1].in,
                         test->optInTimePeriod,
                         &outBegIdx,
                         &outNbElement,
                         gBuffer[1].in
                         );
      break;

   default:
      retCode = TA_INTERNAL_ERROR(135);
   }

   /* Check that the other input were preserved. */
   errNb = checkDataSame( gBuffer[0].in, history->high, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   /* The previous call should have the same output as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent(  gBuffer[0].out0, gBuffer[1].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   /* Do a systematic test of most of the
    * possible startIdx/endIdx range.
    */
   testParam.test  = test;
   testParam.high  = history->high;
   testParam.low   = history->low;

   if( test->doRangeTestFlag )
   {
      errNb = doRangeTest( rangeTestFunction, 
                           TA_FUNC_UNST_NONE,
                           (void *)&testParam, 1, 0 );
      if( errNb != TA_TEST_PASS )
         return errNb;
   }

   return TA_TEST_PASS;
}

