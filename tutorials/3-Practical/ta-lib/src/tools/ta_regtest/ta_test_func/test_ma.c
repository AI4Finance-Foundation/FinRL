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
 *  031707 MF   Add TA_MAVP tests.
 */

/* Description:
 *     Test all MA (Moving Average) functions.
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
  TA_ANY_MA_TEST,
  TA_MAMA_TEST,
  TA_FAMA_TEST
} TA_TestId;


typedef struct
{
   TA_Integer doRangeTestFlag;
   TA_TestId  id;

   TA_Integer unstablePeriod;

   TA_Integer startIdx;
   TA_Integer endIdx;
   TA_Integer optInTimePeriod;
   TA_Integer optInMAType_1;
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
   const TA_Real *mavpPeriod;
   int   testMAVP; /* Boolean */
} TA_RangeTestParam;

/**** Local functions declarations.    ****/
static ErrorNumber do_test_ma( const TA_History *history,
                               const TA_Test *test,
							   int testMAVP /* Boolean */ );

/**** Local variables definitions.     ****/

static TA_Test tableTest[] =
{
   /************/
   /*  T3 TEST */
   /************/
   { 1, TA_ANY_MA_TEST, 0, 0, 251, 5, TA_MAType_T3, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,      0,  85.73, 24,  252-24  }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 5, TA_MAType_T3, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,      1,  84.37, 24,  252-24  },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 5, TA_MAType_T3, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 252-26, 109.03, 24,  252-24  },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 5, TA_MAType_T3, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 252-25, 108.88, 24,  252-24  }, /* Last Value */

   /***************/
   /*  TRIMA TEST */
   /***************/
   { 1, TA_ANY_MA_TEST, 0, 0, 251, 10, TA_MAType_TRIMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,      0,  93.6043, 9,  252-9  }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 10, TA_MAType_TRIMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,      1,  93.4252, 9,  252-9  },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 10, TA_MAType_TRIMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 252-11, 109.1850, 9,  252-9  },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 10, TA_MAType_TRIMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 252-10, 109.1407, 9,  252-9  }, /* Last Value */

   { 1, TA_ANY_MA_TEST, 0, 0, 251,  9, TA_MAType_TRIMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,     0,   93.8176,  8,  252-8  }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  9, TA_MAType_TRIMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 252-9,  109.1312,  8,  252-8  }, /* Last Value */

   { 1, TA_ANY_MA_TEST, 0, 0, 251, 12, TA_MAType_TRIMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,      0,  93.5329, 11,  252-11  }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 12, TA_MAType_TRIMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 252-12, 109.1157, 11,  252-11  }, /* Last Value */



   /*************
    * MAMA TEST *
    *************/
   { 1,   TA_MAMA_TEST, 0, 0, 251, 10, TA_MAType_MAMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 0,       85.3643, 32, 252-32 }, /* First Value */
   { 0,   TA_MAMA_TEST, 0, 0, 251, 10, TA_MAType_MAMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 252-33, 110.1116, 32, 252-32 }, /* Last Value */

   { 0,   TA_FAMA_TEST, 0, 0, 251, 10, TA_MAType_MAMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 0,         81.88, 32, 252-32 }, /* First Value */
   { 0,   TA_FAMA_TEST, 0, 0, 251, 10, TA_MAType_MAMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 252-33,   108.82, 32, 252-32 }, /* Last Value */

   { 0, TA_ANY_MA_TEST, 0, 0, 251, 10, TA_MAType_MAMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 0,       85.3643, 32, 252-32 }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 10, TA_MAType_MAMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 252-33, 110.1116, 32, 252-32 }, /* Last Value */

   /***************************/
   /*  KAMA TEST - Classic    */
   /***************************/

   /* No output value. */
   { 0, TA_ANY_MA_TEST, 0, 1, 1,  14, TA_MAType_KAMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 0, 0, 0, 0},
#ifndef TA_FUNC_NO_RANGE_CHECK
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  0, TA_MAType_KAMA, TA_COMPATIBILITY_DEFAULT, TA_BAD_PARAM, 0, 0, 0, 0 },
#endif

   /* Test with period 10 */
   { 1, TA_ANY_MA_TEST, 0, 0, 251, 10, TA_MAType_KAMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   0,  92.6575, 10, 252-10 }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 10, TA_MAType_KAMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   1,  92.7783, 10, 252-10 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 10, TA_MAType_KAMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 252-11, 109.294, 10, 252-10 }, /* Last Value */


   /*****************************************/
   /*   SMA TEST - CLASSIC/METASTOCK        */
   /*****************************************/

#ifndef TA_FUNC_NO_RANGE_CHECK
   /* Test with invalid parameters */
   { 0, TA_ANY_MA_TEST, 0, 0, 251, -1, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_BAD_PARAM,  0,   0,  0,  0 },
#endif

   /* Test suppose to succeed. */
   { 1, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   0,   93.15,  1,  252-1  }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   1,   94.59,  1,  252-1  },
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   2,   94.73,  1,  252-1  },
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 250,  108.31,  1,  252-1  }, /* Last Value */

   { 1, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   0,  90.42,  29,  252-29 }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   1,  90.21,  29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   2,  89.96,  29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,  29,  87.12,  29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 221, 107.95,  29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_SMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 222, 108.42,  29,  252-29 }, /* Last Value */

   /* Same test and result as TA_COMPATIBILITY_DEFAULT */
   { 1, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,   93.15,  1,  252-1  }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1,   94.59,  1,  252-1  },
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   2,   94.73,  1,  252-1  },
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 250,  108.31,  1,  252-1  }, /* Last Value */

   { 1, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  90.42,  29,  252-29 }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1,  90.21,  29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   2,  89.96,  29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,  29,  87.12,  29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 221, 107.95,  29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_SMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 222, 108.42,  29,  252-29 }, /* Last Value */


   /*******************************/
   /*   WMA TEST  - CLASSIC       */
   /*******************************/

#ifndef TA_FUNC_NO_RANGE_CHECK
   /* No output value. */
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  0, TA_MAType_WMA, TA_COMPATIBILITY_DEFAULT, TA_BAD_PARAM, 0, 0, 0, 0 },
#endif

   /* One value tests. */
   { 0, TA_ANY_MA_TEST, 0, 2,   2,  2, TA_MAType_WMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   0,  94.52,   2, 1 },

   /* Misc tests: period 2, 30 */

   { 1, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_WMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   0,   93.71,  1,  252-1  }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_WMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   1,   94.52,  1,  252-1  },
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_WMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   2,   94.85,  1,  252-1  },
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_WMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 250,  108.16,  1,  252-1  }, /* Last Value */

   { 1, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_WMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   0,  88.567,  29,  252-29 }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_WMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   1,  88.233,  29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_WMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   2,  88.034,  29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_WMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,  29,  87.191,  29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_WMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 221, 109.3413, 29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_WMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 222, 109.3466, 29,  252-29 }, /* Last Value */

   /*******************************/
   /*   WMA TEST  - METASTOCK     */
   /*******************************/

   /* No output value. */
   { 0, TA_ANY_MA_TEST, 0, 1, 1,  14, TA_MAType_WMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 0, 0, 0, 0},
#ifndef TA_FUNC_NO_RANGE_CHECK
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  0, TA_MAType_WMA, TA_COMPATIBILITY_METASTOCK, TA_BAD_PARAM, 0, 0, 0, 0 },
#endif

   /* One value tests. */
   { 0, TA_ANY_MA_TEST, 0, 2,   2,  2, TA_MAType_WMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  94.52,   2, 1 },

   /* Misc tests: period 2, 30 */
   { 1, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_WMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,   93.71,  1,  252-1  }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_WMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1,   94.52,  1,  252-1  },
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_WMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   2,   94.85,  1,  252-1  },
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_WMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 250,  108.16,  1,  252-1  }, /* Last Value */

   { 1, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_WMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  88.567,  29,  252-29 }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_WMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1,  88.233,  29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_WMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   2,  88.034,  29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_WMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,  29,  87.191,  29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_WMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 221, 109.3413, 29,  252-29 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 30, TA_MAType_WMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 222, 109.3466, 29,  252-29 }, /* Last Value */

   /*******************************/
   /*   EMA TEST - Classic        */
   /*******************************/

   /* No output value. */
   { 0, TA_ANY_MA_TEST, 0, 1, 1,  14, TA_MAType_EMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 0, 0, 0, 0},
#ifndef TA_FUNC_NO_RANGE_CHECK
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  0, TA_MAType_EMA, TA_COMPATIBILITY_DEFAULT, TA_BAD_PARAM, 0, 0, 0, 0 },
#endif

   /* Misc tests: period 2, 10 */
   { 1, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_EMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   0,  93.15, 1, 251 }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_EMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   1,  93.96, 1, 251 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  2, TA_MAType_EMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS, 250, 108.21, 1, 251 }, /* Last Value */

   { 1, TA_ANY_MA_TEST, 0, 0, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,    0,  93.22,  9, 243 }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,    1,  93.75,  9, 243 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,   20,  86.46,  9, 243 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_DEFAULT, TA_SUCCESS,  242, 108.97,  9, 243 }, /* Last Value */

   /*******************************/
   /*   EMA TEST - Metastock      */
   /*******************************/


   /* No output value. */
   { 0, TA_ANY_MA_TEST, 0, 1, 1,  14, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 0, 0, 0, 0},
#ifndef TA_FUNC_NO_RANGE_CHECK
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  0, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_BAD_PARAM, 0, 0, 0, 0 },
#endif

   /* Test with 1 unstable price bar. Test for period 2, 10 */
   { 1, TA_ANY_MA_TEST, 1, 0, 251,  2, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  94.15, 1+1, 251-1 }, /* First Value */
   { 0, TA_ANY_MA_TEST, 1, 0, 251,  2, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1,  94.78, 1+1, 251-1 },
   { 0, TA_ANY_MA_TEST, 1, 0, 251,  2, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 250-1, 108.21, 1+1, 251-1 }, /* Last Value */

   { 1, TA_ANY_MA_TEST, 1, 0, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,    0,  93.24,  9+1, 243-1 }, /* First Value */
   { 0, TA_ANY_MA_TEST, 1, 0, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,    1,  93.97,  9+1, 243-1 },
   { 0, TA_ANY_MA_TEST, 1, 0, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   20,  86.23,  9+1, 243-1 },
   { 0, TA_ANY_MA_TEST, 1, 0, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 242-1, 108.97,  9+1, 243-1 }, /* Last Value */

   /* Test with 2 unstable price bar. Test for period 2, 10 */
   { 0, TA_ANY_MA_TEST, 2, 0, 251,  2, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  94.78, 1+2, 251-2 }, /* First Value */
   { 0, TA_ANY_MA_TEST, 2, 0, 251,  2, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1,  94.11, 1+2, 251-2 },
   { 0, TA_ANY_MA_TEST, 2, 0, 251,  2, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 250-2, 108.21, 1+2, 251-2 }, /* Last Value */

   { 0, TA_ANY_MA_TEST, 2, 0, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,    0,  93.97,  9+2, 243-2 }, /* First Value */
   { 0, TA_ANY_MA_TEST, 2, 0, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,    1,  94.79,  9+2, 243-2 },
   { 0, TA_ANY_MA_TEST, 2, 0, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   20,  86.39,  9+2, 243-2 },
   { 0, TA_ANY_MA_TEST, 2, 0, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,  242-2, 108.97,  9+2, 243-2 }, /* Last Value */

   /* Last 3 value with 1 unstable, period 10 */
   { 0, TA_ANY_MA_TEST, 1, 249, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1, 109.22, 249, 3 },
   { 0, TA_ANY_MA_TEST, 1, 249, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   2, 108.97, 249, 3 },

   /* Last 3 value with 2 unstable, period 10 */
   { 0, TA_ANY_MA_TEST, 2, 249, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   2, 108.97, 249, 3 },

   /* Last 3 value with 3 unstable, period 10 */
   { 0, TA_ANY_MA_TEST, 3, 249, 251,  10, TA_MAType_EMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   2, 108.97, 249, 3 },

   /*******************************/
   /*  DEMA TEST - Metastock      */
   /*******************************/

   /* No output value. */
   { 0, TA_ANY_MA_TEST, 0, 1, 1,  14, TA_MAType_DEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 0, 0, 0, 0},
#ifndef TA_FUNC_NO_RANGE_CHECK
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  0, TA_MAType_DEMA, TA_COMPATIBILITY_METASTOCK, TA_BAD_PARAM, 0, 0, 0, 0 },
#endif

   /* Test with period 14 */
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 14, TA_MAType_DEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  83.785, 26, 252-26 }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 14, TA_MAType_DEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1,  84.768, 26, 252-26 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 14, TA_MAType_DEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 252-27, 109.467, 26, 252-26 }, /* Last Value */

   /* Test with 1 unstable price bar. Test for period 2, 14 */
   { 1, TA_ANY_MA_TEST, 1, 0, 251,  2, TA_MAType_DEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  93.960, 4, 252-4 }, /* First Value */
   { 0, TA_ANY_MA_TEST, 1, 0, 251,  2, TA_MAType_DEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1,  94.522, 4, 252-4 },
   { 0, TA_ANY_MA_TEST, 1, 0, 251,  2, TA_MAType_DEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 252-5, 107.94, 4, 252-4 }, /* Last Value */

   { 1, TA_ANY_MA_TEST, 1, 0, 251,  14, TA_MAType_DEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,    0,  84.91,  (13*2)+2, 252-((13*2)+2) }, /* First Value */
   { 0, TA_ANY_MA_TEST, 1, 0, 251,  14, TA_MAType_DEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,    1,  84.97,  (13*2)+2, 252-((13*2)+2) },
   { 0, TA_ANY_MA_TEST, 1, 0, 251,  14, TA_MAType_DEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,    2,  84.80,  (13*2)+2, 252-((13*2)+2) },
   { 0, TA_ANY_MA_TEST, 1, 0, 251,  14, TA_MAType_DEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,    3,  85.14,  (13*2)+2, 252-((13*2)+2) },
   { 0, TA_ANY_MA_TEST, 1, 0, 251,  14, TA_MAType_DEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   20,  89.83,  (13*2)+2, 252-((13*2)+2) },
   { 0, TA_ANY_MA_TEST, 1, 0, 251,  14, TA_MAType_DEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 252-((13*2)+2+1), 109.4676, (13*2)+2, 252-((13*2)+2) }, /* Last Value */

   /*******************************/
   /*  TEMA TEST - Metastock      */
   /*******************************/
   /* No output value. */
   { 0, TA_ANY_MA_TEST, 0, 1, 1,  14, TA_MAType_TEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 0, 0, 0, 0},
#ifndef TA_FUNC_NO_RANGE_CHECK
   { 0, TA_ANY_MA_TEST, 0, 0, 251,  0, TA_MAType_TEMA, TA_COMPATIBILITY_METASTOCK, TA_BAD_PARAM, 0, 0, 0, 0 },
#endif

   /* Test with period 14 */
   { 1, TA_ANY_MA_TEST, 0, 0, 251, 14, TA_MAType_TEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   0,  84.721, 39, 252-39 }, /* First Value */
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 14, TA_MAType_TEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS,   1,  84.089, 39, 252-39 },
   { 0, TA_ANY_MA_TEST, 0, 0, 251, 14, TA_MAType_TEMA, TA_COMPATIBILITY_METASTOCK, TA_SUCCESS, 252-40, 108.418, 39, 252-39 }, /* Last Value */
};

#define NB_TEST (sizeof(tableTest)/sizeof(TA_Test))

/**** Global functions definitions.   ****/
ErrorNumber test_func_ma( TA_History *history )
{
   unsigned int i;
   ErrorNumber retValue;

   /* Re-initialize all the unstable period to zero. */
   TA_SetUnstablePeriod( TA_FUNC_UNST_ALL, 0 );

   for( i=0; i < NB_TEST; i++ )
   {

      if( (int)tableTest[i].expectedNbElement > (int)history->nbBars )
      {
         printf( "TA_MA Failed Bad Parameter for Test #%d (%d,%d)\n",
                 i, tableTest[i].expectedNbElement, history->nbBars );
         return TA_TESTUTIL_TFRR_BAD_PARAM;
      }

      retValue = do_test_ma( history, &tableTest[i], 0 );
      if( retValue != 0 )
      {
         printf( "TA_MA Failed Test #%d (Code=%d)\n", i, retValue );
         return retValue;
      }

	  /* If TA_ANY_MA_TEST. repeat test with TA_MAVP */
	  if( tableTest[i].id == TA_ANY_MA_TEST )
	  {
         retValue = do_test_ma( history, &tableTest[i], 1 );
         if( retValue != 0 )
         {
            printf( "TA_MAVP Failed Test #%d (Code=%d)\n", i, retValue );
            return retValue;
	     }
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
  TA_Real *dummyBuffer;
    
  (void)outputBufferInt;

  *isOutputInteger = 0;

  testParam = (TA_RangeTestParam *)opaqueData;   

  switch( testParam->test->optInMAType_1 )
  {
  case TA_MAType_MAMA:
     dummyBuffer = TA_Malloc( sizeof(TA_Real)*(endIdx-startIdx+600) );
     if( outputNb == 0 )
     {
        retCode = TA_MAMA( startIdx,
                           endIdx,
                           testParam->close,
                           0.5, 0.05,                      
                           outBegIdx,
                           outNbElement,
                           outputBuffer,
                           &dummyBuffer[300] );
     }
     else
     {
        retCode = TA_MAMA( startIdx,
                           endIdx,
                           testParam->close,
                           0.5, 0.05,                      
                           outBegIdx,
                           outNbElement,
                           &dummyBuffer[300],
                           outputBuffer );
     }
     TA_Free( dummyBuffer );
     *lookback = TA_MAMA_Lookback( 0.5, 0.05 );
     break;
  default:
	  if( testParam->testMAVP )
	  {
     retCode = TA_MAVP( startIdx,
                      endIdx,
                      testParam->close,
					  testParam->mavpPeriod,
                      2,testParam->test->optInTimePeriod,
                      (TA_MAType)testParam->test->optInMAType_1,
                      outBegIdx,
                      outNbElement,
                      outputBuffer );

     *lookback = TA_MAVP_Lookback( 2, testParam->test->optInTimePeriod,
                                 (TA_MAType)testParam->test->optInMAType_1 );
	  }
	  else
	  {
     /* Test for the TA_MA function. All the MA can be done
      * through that function.
      */
     retCode = TA_MA( startIdx,
                      endIdx,
                      testParam->close,
                      testParam->test->optInTimePeriod,
                      (TA_MAType)testParam->test->optInMAType_1,
                      outBegIdx,
                      outNbElement,
                      outputBuffer );

     *lookback = TA_MA_Lookback( testParam->test->optInTimePeriod,
                                 (TA_MAType)testParam->test->optInMAType_1 );
	  }
     break;
  }

  return retCode;
}

static ErrorNumber do_test_ma( const TA_History *history,
                               const TA_Test *test,
							   int testMAVP )
{
   TA_RetCode retCode;
   ErrorNumber errNb;
   TA_Integer outBegIdx;
   TA_Integer outNbElement;
   TA_RangeTestParam testParam;
   TA_Integer temp, temp2;
   const TA_Real *referenceInput;

   /* TA_MAVP is tested only for TA_ANY_MA_TEST */
   if( testMAVP && (test->id != TA_ANY_MA_TEST) )
   {
      return TA_TEST_PASS;
   }

   TA_SetCompatibility( (TA_Compatibility)test->compatibility );

   /* Set to NAN all the elements of the gBuffers.  */
   clearAllBuffers();

   /* Build the input. */
   setInputBuffer( 0, history->close, history->nbBars );
   setInputBuffer( 1, history->close, history->nbBars );
   if( testMAVP )
   {
      setInputBufferValue( 2, test->optInTimePeriod, history->nbBars );
   }

   /* Re-initialize all the unstable period to zero. */
   TA_SetUnstablePeriod( TA_FUNC_UNST_ALL, 0 );
   
   /* Set the unstable period requested for that test. */
   switch( test->optInMAType_1 )
   {
   case TA_MAType_TEMA:
   case TA_MAType_DEMA:
   case TA_MAType_EMA:
      retCode = TA_SetUnstablePeriod( TA_FUNC_UNST_EMA, test->unstablePeriod );
      break;
   case TA_MAType_KAMA:
      retCode = TA_SetUnstablePeriod( TA_FUNC_UNST_KAMA, test->unstablePeriod );
      break;
   case TA_MAType_MAMA:
      retCode = TA_SetUnstablePeriod( TA_FUNC_UNST_MAMA, test->unstablePeriod );
      break;
   case TA_MAType_T3:
      retCode = TA_SetUnstablePeriod( TA_FUNC_UNST_T3, test->unstablePeriod );
      break;
   default:
      retCode = TA_SUCCESS;
      break;
   }

   if( retCode != TA_SUCCESS )
      return TA_TEST_TFRR_SETUNSTABLE_PERIOD_FAIL;

   /* Transform the inputs for MAMA (it is an AVGPRICE in John Ehlers book). */
   if( test->optInMAType_1 == TA_MAType_MAMA )
   {
      TA_MEDPRICE( 0, history->nbBars-1, history->high, history->low,
                   &outBegIdx, &outNbElement, gBuffer[0].in );

      TA_MEDPRICE( 0, history->nbBars-1, history->high, history->low,
                   &outBegIdx, &outNbElement, gBuffer[1].in );

      /* Will be use as reference */
      TA_MEDPRICE( 0, history->nbBars-1, history->high, history->low,
                   &outBegIdx, &outNbElement, gBuffer[2].in );
      referenceInput = gBuffer[2].in;
   }
   else
      referenceInput = history->close;

   

   /* Make a simple first call. */
   switch( test->id )
   {
   case TA_ANY_MA_TEST:
	  if(testMAVP)
	  {
         retCode = TA_MAVP( test->startIdx,
                            test->endIdx,
                            gBuffer[0].in,
							gBuffer[2].in,
							2, test->optInTimePeriod,
                            (TA_MAType)test->optInMAType_1,
                            &outBegIdx,
                            &outNbElement,
                            gBuffer[0].out0 );
	  }
	  else
	  {
         retCode = TA_MA( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          test->optInTimePeriod,
                          (TA_MAType)test->optInMAType_1,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[0].out0 );
	  }
      break;
   case TA_MAMA_TEST:
      retCode = TA_MAMA( test->startIdx,
                         test->endIdx,
                         gBuffer[0].in,
                         0.5, 0.05,                       
                         &outBegIdx,
                         &outNbElement,
                         gBuffer[0].out0,
                         gBuffer[0].out2 );

     break;
   case TA_FAMA_TEST:
      retCode = TA_MAMA( test->startIdx,
                         test->endIdx,
                         gBuffer[0].in,
                         0.5, 0.05,                       
                         &outBegIdx,
                         &outNbElement,
                         gBuffer[0].out2,
                         gBuffer[0].out0 );

     break;
   }

   errNb = checkDataSame( gBuffer[0].in, referenceInput, history->nbBars );
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
   switch( test->id )
   {
   case TA_ANY_MA_TEST:
	  if(testMAVP)
   	  {
      retCode = TA_MAVP( test->startIdx,
                       test->endIdx,
                       gBuffer[1].in,
					   gBuffer[2].in,
                       2,test->optInTimePeriod,
                       (TA_MAType)test->optInMAType_1,
                       &outBegIdx,
                       &outNbElement,
                       gBuffer[1].in );
	  }
	  else
	  {
      retCode = TA_MA( test->startIdx,
                       test->endIdx,
                       gBuffer[1].in,
                       test->optInTimePeriod,
                       (TA_MAType)test->optInMAType_1,
                       &outBegIdx,
                       &outNbElement,
                       gBuffer[1].in );
	  }
      break;
   case TA_MAMA_TEST:
      retCode = TA_MAMA( test->startIdx,
                         test->endIdx,
                         gBuffer[1].in,
                         0.5, 0.05,                       
                         &outBegIdx,
                         &outNbElement,
                         gBuffer[1].in,
                         gBuffer[0].out2 );
     break;
   case TA_FAMA_TEST:
      retCode = TA_MAMA( test->startIdx,
                         test->endIdx,
                         gBuffer[1].in,
                         0.5, 0.05,                       
                         &outBegIdx,
                         &outNbElement,
                         gBuffer[0].out2,
                         gBuffer[1].in );
     break;
   }

   /* The previous call to TA_MA should have the same output
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

  /* Verify that the "all-purpose" TA_MA_Lookback is consistent
   * with the corresponding moving average lookback function.
   */
   if( test->optInTimePeriod >= 2 )
   {
      switch( test->optInMAType_1 )
      {
      case TA_MAType_WMA:
         temp = TA_WMA_Lookback( test->optInTimePeriod );
         break;
   
      case TA_MAType_SMA:
         temp = TA_SMA_Lookback( test->optInTimePeriod );
         break;
   
      case TA_MAType_EMA:
         temp = TA_EMA_Lookback( test->optInTimePeriod );
         break;
   
      case TA_MAType_DEMA:
         temp = TA_DEMA_Lookback( test->optInTimePeriod );
         break;
   
      case TA_MAType_TEMA:
         temp = TA_TEMA_Lookback( test->optInTimePeriod );
         break;
   
      case TA_MAType_KAMA:
         temp = TA_KAMA_Lookback( test->optInTimePeriod );
         break;
   
      case TA_MAType_MAMA:
         temp = TA_MAMA_Lookback( 0.5, 0.05 );
         break;
   
      case TA_MAType_TRIMA:
         temp = TA_TRIMA_Lookback( test->optInTimePeriod );
         break;
   
      case TA_MAType_T3:
         temp = TA_T3_Lookback( test->optInTimePeriod, 0.7 );
         break;
   
      default:
         return TA_TEST_TFRR_BAD_MA_TYPE;
      }
   
      temp2 = TA_MA_Lookback( test->optInTimePeriod, (TA_MAType)test->optInMAType_1 );
   
      if( temp != temp2 )
      {
         printf( "\nFailed for MA Type #%d for period %d\n", test->optInMAType_1, test->optInTimePeriod );
         return TA_TEST_TFFR_BAD_MA_LOOKBACK;
      } 
   }

   /* Do a systematic test of most of the
    * possible startIdx/endIdx range.
    */
   testParam.test  = test;
   testParam.close = referenceInput;
   testParam.testMAVP = testMAVP;
   testParam.mavpPeriod = gBuffer[2].in;

   if( test->doRangeTestFlag )
   {
      switch( test->optInMAType_1 )
      {
      case TA_MAType_TEMA:
      case TA_MAType_DEMA:
      case TA_MAType_EMA:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_EMA,
                              (void *)&testParam, 1, 0 );
         break;
      case TA_MAType_T3:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_T3,
                              (void *)&testParam, 1, 0 );
         break;
      case TA_MAType_KAMA:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_KAMA,
                              (void *)&testParam, 1, 0 );
         break;
      case TA_MAType_MAMA:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_MAMA,
                              (void *)&testParam, 2, 0 );
         break;
      default:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_NONE,
                              (void *)&testParam, 1, 0 );
      }

      if( errNb != TA_TEST_PASS )
         return errNb;
   }

   return TA_TEST_PASS;
}

