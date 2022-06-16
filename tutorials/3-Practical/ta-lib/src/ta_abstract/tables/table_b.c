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

/*********************************************************************
 * This file contains only TA functions starting with the letter 'B' *
 *********************************************************************/
#include <stddef.h>
#include "ta_abstract.h"
#include "ta_def_ui.h"

/* Follow the 3 steps defined below for adding a new TA Function to this
 * file.
 */


/****************************************************************************
 * Step 1 - Define here the interface to your TA functions with
 *          the macro DEF_FUNCTION.
 *
 ****************************************************************************/

/* BBANDS BEGIN */

/* Nb Deviation up/down is used for bollinger bands. */
const TA_OptInputParameterInfo TA_DEF_UI_NbDeviationUp =
{
   TA_OptInput_RealRange, /* type */
   "optInNbDevUp",        /* paramName */
   0,                     /* flags */

   "Deviations up",     /* displayName */
   (const void *)&TA_DEF_NbDeviation, /* dataSet */
   2.0, /* defaultValue */
   "Deviation multiplier for upper band", /* hint */

   NULL /* CamelCase name */
};

const TA_OptInputParameterInfo TA_DEF_UI_NbDeviationDn =
{
   TA_OptInput_RealRange, /* type */
   "optInNbDevDn",        /* paramName */
   0,                     /* flags */

   "Deviations down",          /* displayName */
   (const void *)&TA_DEF_NbDeviation, /* dataSet */
   2.0, /* defaultValue */
   "Deviation multiplier for lower band", /* hint */

   NULL /* CamelCase name */
};

const TA_OutputParameterInfo TA_DEF_UI_Output_Real_BBANDS_Middle =
                               { TA_Output_Real, "outRealMiddleBand", TA_OUT_LINE };

const TA_OutputParameterInfo TA_DEF_UI_Output_Real_BBANDS_Upper =
                               { TA_Output_Real, "outRealUpperBand", TA_OUT_UPPER_LIMIT };

const TA_OutputParameterInfo TA_DEF_UI_Output_Real_BBANDS_Lower =
                                { TA_Output_Real, "outRealLowerBand", TA_OUT_LOWER_LIMIT };

static const TA_InputParameterInfo    *TA_BBANDS_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_BBANDS_Outputs[]   =
{
  &TA_DEF_UI_Output_Real_BBANDS_Upper,
  &TA_DEF_UI_Output_Real_BBANDS_Middle,
  &TA_DEF_UI_Output_Real_BBANDS_Lower,
  NULL
};

static const TA_OptInputParameterInfo *TA_BBANDS_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_5_MINIMUM2,
  &TA_DEF_UI_NbDeviationUp,
  &TA_DEF_UI_NbDeviationDn,
  &TA_DEF_UI_MA_Method,
  NULL
};

DEF_FUNCTION( BBANDS,                    /* name */
              TA_GroupId_OverlapStudies, /* groupId */
              "Bollinger Bands",         /* hint */
              "Bbands",                  /* CamelCase name */
              TA_FUNC_FLG_OVERLAP        /* flags */
             );
/* BBANDS END */


/* BOP BEGIN */
static const TA_InputParameterInfo    *TA_BOP_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_OHLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_BOP_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_BOP_OptInputs[] = { NULL };

DEF_FUNCTION( BOP,                   /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Balance Of Power",         /* hint */
              "Bop",                      /* CamelCase name */
              0                          /* flags */
             );
/* BOP END */

/* BETA BEGIN */
static const TA_InputParameterInfo    *TA_BETA_Inputs[]    =
{
    &TA_DEF_UI_Input_Real0,
    &TA_DEF_UI_Input_Real1,
    NULL
};

static const TA_OutputParameterInfo   *TA_BETA_Outputs[]   =
{
    &TA_DEF_UI_Output_Real,
    NULL
};

static const TA_OptInputParameterInfo *TA_BETA_OptInputs[] =
{
  &TA_DEF_UI_TimePeriod_5,
  NULL
};

DEF_FUNCTION( BETA,                      /* name */
              TA_GroupId_Statistic,     /* groupId */
              "Beta", /* hint */
              "Beta",                /* CamelCase name */
              0                        /* flags */
            );
/* BETA END */

/****************************************************************************
 * Step 2 - Add your TA function to the table.
 *          Keep in alphabetical order. Must be NULL terminated.
 ****************************************************************************/
const TA_FuncDef *TA_DEF_TableB[] =
{
   ADD_TO_TABLE(BBANDS),
   ADD_TO_TABLE(BETA),
   ADD_TO_TABLE(BOP),
   NULL
};


/* Do not modify the following line. */
const unsigned int TA_DEF_TableBSize =
              ((sizeof(TA_DEF_TableB)/sizeof(TA_FuncDef *))-1);


/****************************************************************************
 * Step 3 - Make sure "gen_code" is executed for generating all other
 *          source files derived from this one.
 *          You can then re-compile the library as usual and you are done!
 ****************************************************************************/
