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
 *     Test TRANGE and ATR function.
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
   TA_Integer doRangeTestFlag; /* One will do a call to doRangeTest */

   TA_Integer unstablePeriod;

   TA_Integer startIdx;
   TA_Integer endIdx;

   TA_Integer doAverage;          /* 1 indicate ATR, else TRANGE. */
   TA_Integer optInTimePeriod;  /* Meaningful only for ATR. */

   TA_RetCode expectedRetCode;

   TA_Integer oneOfTheExpectedOutRealIndex;
   TA_Real    oneOfTheExpectedOutReal;

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
   /* TRANGE TEST */
   { 1, 0, 0, 251, 0,  0, TA_SUCCESS,   0,  3.535,  1,  251 }, /* First Value */
   { 0, 0, 0, 251, 0,  0, TA_SUCCESS,  12,  9.685,  1,  251 },
   { 0, 0, 0, 251, 0,  0, TA_SUCCESS,  40,  5.125,  1,  251 },
   { 0, 0, 0, 251, 0,  0, TA_SUCCESS, 250,  2.88,   1,  251 }, /* Last Value */

   /* ATR TEST */
   { 1, 0, 0, 251, 1,  1, TA_SUCCESS,   0,  3.535,  1,  251 }, /* First Value */
   { 0, 0, 0, 251, 1,  1, TA_SUCCESS,  12,  9.685,  1,  251 },
   { 0, 0, 0, 251, 1,  1, TA_SUCCESS,  40,  5.125,  1,  251 },
   { 0, 0, 0, 251, 1,  1, TA_SUCCESS, 250,  2.88,   1,  251 }, /* Last Value */

   { 0, 1, 14, 15, 1, 14, TA_SUCCESS,   0, 3.4876, 15, 1 },
   { 0, 1, 15, 16, 1, 14, TA_SUCCESS,   0, 3.4876, 15, 2 },

   { 1, 0, 0, 251, 1, 14, TA_SUCCESS,   0,  3.578, 14,  252-14 }, /* First Value */
   { 0, 0, 0, 251, 1, 14, TA_SUCCESS,   1,  3.4876, 14, 252-14 },
   { 0, 0, 0, 251, 1, 14, TA_SUCCESS,   2,  3.55, 14,  252-14 },
   { 0, 0, 0, 251, 1, 14, TA_SUCCESS,  12,  3.245, 14,  252-14 },
   { 0, 0, 0, 251, 1, 14, TA_SUCCESS,  13,  3.394, 14,  252-14 },
   { 0, 0, 0, 251, 1, 14, TA_SUCCESS,  14,  3.413, 14,  252-14 },
   { 0, 0, 0, 251, 1, 14, TA_SUCCESS, 237,  3.26, 14,  252-14 }, /* Last Value */

};

#define NB_TEST (sizeof(tableTest)/sizeof(TA_Test))

/**** Global functions definitions.   ****/
ErrorNumber test_func_trange( TA_History *history )
{
   unsigned int i;
   ErrorNumber retValue;

   for( i=0; i < NB_TEST; i++ )
   {
      if( (int)tableTest[i].expectedNbElement > (int)history->nbBars )
      {
         printf( "%s Failed Bad Parameter for Test #%d (%d,%d)\n",
                 tableTest[i].doAverage? "TA_ATR":"TA_TRANGE",
                 i, tableTest[i].expectedNbElement, history->nbBars );
         return TA_TESTUTIL_TFRR_BAD_PARAM;
      }

      retValue = do_test( history, &tableTest[i] );
      if( retValue != 0 )
      {
         printf( "%s Failed Test #%d (Code=%d)\n",
                 tableTest[i].doAverage? "TA_ATR":"TA_TRANGE",              
                 i, retValue );
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

   (void)outputNb;
   (void)outputBufferInt;
  
   *isOutputInteger = 0;

   testParam = (TA_RangeTestParam *)opaqueData;   


   if( testParam->test->doAverage )
   {
      retCode = TA_ATR(
                        startIdx,
                        endIdx,
                        testParam->high,
                        testParam->low,
                        testParam->close,
                        testParam->test->optInTimePeriod,                        
                        outBegIdx,
                        outNbElement,
                        outputBuffer );
     *lookback = TA_ATR_Lookback( testParam->test->optInTimePeriod );
   }
   else
   {
      retCode = TA_TRANGE(
                        startIdx,
                        endIdx,
                        testParam->high,
                        testParam->low,
                        testParam->close,                        
                        outBegIdx,
                        outNbElement,
                        outputBuffer );

     *lookback = TA_TRANGE_Lookback();
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

   if( test->doAverage )
   {
      TA_SetUnstablePeriod( TA_FUNC_UNST_ATR, test->unstablePeriod );
      retCode = TA_ATR(    test->startIdx,
                           test->endIdx,
                           gBuffer[0].in,
                           gBuffer[1].in,
                           gBuffer[2].in,
                           test->optInTimePeriod,                           
                           &outBegIdx,
                           &outNbElement,
                           gBuffer[0].out0 );
   }
   else
   {
      retCode = TA_TRANGE( test->startIdx,
                           test->endIdx,
                           gBuffer[0].in,
                           gBuffer[1].in,
                           gBuffer[2].in,                           
                           &outBegIdx,
                           &outNbElement,
                           gBuffer[0].out0 );
   }

   errNb = checkDataSame( gBuffer[0].in, history->high,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[1].in, history->low, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[2].in, history->close, history->nbBars );
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
   if( test->doAverage )
   {
      TA_SetUnstablePeriod( TA_FUNC_UNST_ATR, test->unstablePeriod );
      retCode = TA_ATR( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        test->optInTimePeriod,                           
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[0].in );
   }
   else
   {
      retCode = TA_TRANGE( test->startIdx,
                           test->endIdx,
                           gBuffer[0].in,
                           gBuffer[1].in,
                           gBuffer[2].in,                           
                           &outBegIdx,
                           &outNbElement,
                           gBuffer[0].in );
   }

   /* The previous call to TA_MA should have the same output
    * as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[0].out0, gBuffer[0].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   errNb = checkExpectedValue( gBuffer[0].in, 
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
   testParam.high  = history->high;
   testParam.low   = history->low;
   testParam.close = history->close;

   if( test->doRangeTestFlag )
   {
      errNb = doRangeTest( rangeTestFunction, 
                           TA_FUNC_UNST_ATR,
                           (void *)&testParam, 1, 0 );
      if( errNb != TA_TEST_PASS )
         return errNb;
   }

   return TA_TEST_PASS;
}
