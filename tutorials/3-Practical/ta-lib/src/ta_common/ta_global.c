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
 *  AC       Angelo Ciceri
 *
 *
 * Change history:
 *
 *  MMDDYY BY   Description
 *  -------------------------------------------------------------------
 *  112400 MF   First version.
 *  082004 AC   Add TA_SetCandleSettings, TA_RestoreCandleDefaultSettings
 *              and call to TA_RestoreCandleDefaultSettings in TA_Initialize
 *  041106 MF   Add prefix to theGlobals to avoid clash with other libs.
 *  040707 MF   Change global initialization to eliminate Mac OS X link error.
 */

/* Description:
 *   Provides initialization / shutdown functionality for all modules.
 */

/**** Headers ****/
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ta_common.h"
#include "ta_magic_nb.h"
#include "ta_global.h"
#include "ta_func.h"

/**** External functions declarations. ****/
/* None */

/**** External variables declarations. ****/
/* None */

/**** Global variables definitions.    ****/

/* The entry point for all globals */
TA_LibcPriv ta_theGlobals = {0,{{0,0,0}},0,0,0,0,(TA_Compatibility)0,{0},{{(TA_CandleSettingType)0,(TA_RangeType)0,0,0}}};

TA_LibcPriv *TA_Globals = &ta_theGlobals;

/**** Local declarations.              ****/
/* None */

/**** Local functions declarations.    ****/
/* None */

/**** Local variables definitions.     ****/
/* None */

/**** Global functions definitions.   ****/
TA_RetCode TA_Initialize( void )
{
   /* Initialize the "global variable" used to manage the global
    * variables of all other modules...
    */
   memset( TA_Globals, 0, sizeof( TA_LibcPriv ) );
   TA_Globals->magicNb = TA_LIBC_PRIV_MAGIC_NB;	  

   /*** At this point, TA_Shutdown can be called to clean-up. ***/

   /* Set the default value to global variables */
   TA_RestoreCandleDefaultSettings( TA_AllCandleSettings );

   return TA_SUCCESS;
}

TA_RetCode TA_Shutdown( void )
{
   if( TA_Globals->magicNb != TA_LIBC_PRIV_MAGIC_NB )
      return TA_LIB_NOT_INITIALIZE;

   /* Initialize to all zero to make sure we invalidate that object. */
   memset( TA_Globals, 0, sizeof( TA_LibcPriv ) );

   return TA_SUCCESS;
}

TA_RetCode TA_SetCandleSettings( TA_CandleSettingType settingType, 
                                 TA_RangeType rangeType, 
                                 int avgPeriod, 
                                 double factor )
{
    /*printf("setcdlset:%d  ",settingType);*/
    if( settingType >= TA_AllCandleSettings )
        return TA_BAD_PARAM;
    TA_Globals->candleSettings[settingType].settingType = settingType;
    TA_Globals->candleSettings[settingType].rangeType = rangeType;
    TA_Globals->candleSettings[settingType].avgPeriod = avgPeriod;
    TA_Globals->candleSettings[settingType].factor = factor;
    /*printf("cdlset: %d %d %d %f\n",TA_Globals->candleSettings[settingType].settingType,TA_Globals->candleSettings[settingType].rangeType,
        TA_Globals->candleSettings[settingType].avgPeriod,TA_Globals->candleSettings[settingType].factor);*/
    return TA_SUCCESS;
}

TA_RetCode TA_RestoreCandleDefaultSettings( TA_CandleSettingType settingType )
{
    const TA_CandleSetting TA_CandleDefaultSettings[] = {
        /* real body is long when it's longer than the average of the 10 previous candles' real body */
        { TA_BodyLong, TA_RangeType_RealBody, 10, 1.0 },
        /* real body is very long when it's longer than 3 times the average of the 10 previous candles' real body */
        { TA_BodyVeryLong, TA_RangeType_RealBody, 10, 3.0 },
        /* real body is short when it's shorter than the average of the 10 previous candles' real bodies */
        { TA_BodyShort, TA_RangeType_RealBody, 10, 1.0 },
        /* real body is like doji's body when it's shorter than 10% the average of the 10 previous candles' high-low range */
        { TA_BodyDoji, TA_RangeType_HighLow, 10, 0.1 },
        /* shadow is long when it's longer than the real body */
        { TA_ShadowLong, TA_RangeType_RealBody, 0, 1.0 },
        /* shadow is very long when it's longer than 2 times the real body */
        { TA_ShadowVeryLong, TA_RangeType_RealBody, 0, 2.0 },
        /* shadow is short when it's shorter than half the average of the 10 previous candles' sum of shadows */
        { TA_ShadowShort, TA_RangeType_Shadows, 10, 1.0 },
        /* shadow is very short when it's shorter than 10% the average of the 10 previous candles' high-low range */
        { TA_ShadowVeryShort, TA_RangeType_HighLow, 10, 0.1 },
        /* when measuring distance between parts of candles or width of gaps */
        /* "near" means "<= 20% of the average of the 5 previous candles' high-low range" */
        { TA_Near, TA_RangeType_HighLow, 5, 0.2 },
        /* when measuring distance between parts of candles or width of gaps */
        /* "far" means ">= 60% of the average of the 5 previous candles' high-low range" */
        { TA_Far, TA_RangeType_HighLow, 5, 0.6 },
        /* when measuring distance between parts of candles or width of gaps */
        /* "equal" means "<= 5% of the average of the 5 previous candles' high-low range" */
        { TA_Equal, TA_RangeType_HighLow, 5, 0.05 }
    };

    int i;
    if( settingType > TA_AllCandleSettings )
        return TA_BAD_PARAM;
    if( settingType == TA_AllCandleSettings )
        for( i = 0; i < TA_AllCandleSettings; ++i )
            TA_Globals->candleSettings[i] = TA_CandleDefaultSettings[i];
    else
        TA_Globals->candleSettings[settingType] = TA_CandleDefaultSettings[settingType];
    return TA_SUCCESS;
}

/**** Local functions definitions.     ****/
/* None */


