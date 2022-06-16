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
 *  112405 MF   First version.
 *              
 */

/* Description:
 *
 *     Test functions which have the following
 *     characterisic: 
 *      - have one output 
 *      - zero or one parameter being a period.
 *      - the input is open, high, low and close.
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
TA_AVGPRICE_TEST,
TA_BOP_TEST
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
   const TA_Real *open;
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
   /*******************/
   /* AVGPRICE TEST   */
   /*******************/
   { 0, TA_AVGPRICE_TEST,  0, 251, 0, TA_SUCCESS,   0,  92.0,  0,  252 },
   { 1, TA_AVGPRICE_TEST,  0, 251, 0, TA_SUCCESS,   1,  93.17, 0,  252 }, 

   /*************/
   /* BOP TEST  */
   /*************/

   /* The following two should always be identical. */
   { 0, TA_BOP_TEST,  0, 251, 0, TA_SUCCESS,   0,  -0.40,  0,  252 },
   { 1, TA_BOP_TEST,  0, 251, 0, TA_SUCCESS,   1,   0.94,  0,  252 }

};

#define NB_TEST (sizeof(tableTest)/sizeof(TA_Test))

/**** Global functions definitions.   ****/
ErrorNumber test_func_per_ohlc( TA_History *history )
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
   case TA_BOP_TEST:
      retCode = TA_BOP( startIdx,
                        endIdx,
                        testParam->open,
                        testParam->high,
                        testParam->low,
                        testParam->close,                        
                        outBegIdx,
                        outNbElement,
                        outputBuffer );
      *lookback = TA_BOP_Lookback();
      break;
   case TA_AVGPRICE_TEST:
      retCode = TA_AVGPRICE( startIdx,
                          endIdx,
                          testParam->open,
                          testParam->high,
                          testParam->low,
                          testParam->close,
                          outBegIdx,
                          outNbElement,
                          outputBuffer );
      *lookback = TA_AVGPRICE_Lookback();
      break;
   default:
      retCode = TA_INTERNAL_ERROR(171);
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
   setInputBuffer( 0, history->open,  history->nbBars );
   setInputBuffer( 1, history->high,   history->nbBars );
   setInputBuffer( 2, history->low, history->nbBars );
   setInputBuffer( 3, history->close, history->nbBars );
      
   /* Make a simple first call. */
   switch( test->theFunction )
   {
   case TA_BOP_TEST:
      retCode = TA_BOP( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        gBuffer[3].in,                        
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[0].out0 );
      break;

   case TA_AVGPRICE_TEST:
      retCode = TA_AVGPRICE( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          gBuffer[2].in,
                          gBuffer[3].in,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[0].out0 );
      break;

   default:
      retCode = TA_INTERNAL_ERROR(172);
   }

   /* Check that the input were preserved. */
   errNb = checkDataSame( gBuffer[0].in, history->open,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[1].in, history->high, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[2].in, history->low,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[3].in, history->close,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[0].out0, 0 );

   outBegIdx = outNbElement = 0;

   /* Make another call where the input and the output are the
    * same buffer.
    */
   switch( test->theFunction )
   {
   case TA_BOP_TEST:
      retCode = TA_BOP( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        gBuffer[3].in,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[0].in );
      break;
   case TA_AVGPRICE_TEST:
      retCode = TA_AVGPRICE( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          gBuffer[2].in,
                          gBuffer[3].in,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[0].in );
      break;
   default:
      retCode = TA_INTERNAL_ERROR(173);
   }

   /* Check that the input were preserved. */
   errNb = checkDataSame( gBuffer[1].in, history->high, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[2].in, history->low,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[3].in, history->close,history->nbBars );
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
   setInputBuffer( 0, history->open,  history->nbBars );

   /* Make another call where the input and the output are the
    * same buffer.
    */
   switch( test->theFunction )
   {
   case TA_BOP_TEST:
      retCode = TA_BOP( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        gBuffer[3].in,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[1].in );
      break;
   case TA_AVGPRICE_TEST:
      retCode = TA_AVGPRICE( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          gBuffer[2].in,
                          gBuffer[3].in,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[1].in );
      break;
   default:
      retCode = TA_INTERNAL_ERROR(174);
   }

   /* Check that the input were preserved. */
   errNb = checkDataSame( gBuffer[0].in, history->open,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[2].in, history->low,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[3].in, history->close,history->nbBars );
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
   setInputBuffer( 1, history->high,   history->nbBars );

   /* Make another call where the input and the output are the
    * same buffer.
    */
   switch( test->theFunction )
   {
   case TA_BOP_TEST:
      retCode = TA_BOP( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        gBuffer[3].in,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[2].in );
      break;
   case TA_AVGPRICE_TEST:
      retCode = TA_AVGPRICE( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          gBuffer[2].in,
                          gBuffer[3].in,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[2].in );
      break;
   default:
      retCode = TA_INTERNAL_ERROR(175);
   }

   /* Check that the input were preserved. */
   errNb = checkDataSame( gBuffer[0].in, history->open,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[1].in, history->high, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[3].in, history->close, history->nbBars );
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
   setInputBuffer( 2, history->low, history->nbBars );

   /* Make another call where the input and the output are the
    * same buffer.
    */
   switch( test->theFunction )
   {
   case TA_BOP_TEST:
      retCode = TA_BOP( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        gBuffer[3].in,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[3].in );
      break;
   case TA_AVGPRICE_TEST:
      retCode = TA_AVGPRICE( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          gBuffer[2].in,
                          gBuffer[3].in,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[3].in );
      break;
   default:
      retCode = TA_INTERNAL_ERROR(176);
   }

   /* Check that the input were preserved. */
   errNb = checkDataSame( gBuffer[0].in, history->open,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[1].in, history->high, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[2].in, history->low, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   /* The previous call should have the same output as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[0].out0, gBuffer[3].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[3].in, 0 );
   setInputBuffer( 3, history->close, history->nbBars );

   /* Do a systematic test of most of the
    * possible startIdx/endIdx range.
    */
   testParam.test  = test;
   testParam.open  = history->open;
   testParam.high  = history->high;
   testParam.low   = history->low;
   testParam.close = history->close;

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

