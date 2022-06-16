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
 * This file contains only TA functions starting with the letter 'U' *
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

/* ULTOSC BEGIN */
const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_7_PER1 =
{
   TA_OptInput_IntegerRange, /* type */
   "optInTimePeriod1",       /* paramName */
   0,                        /* flags */

   "First Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive, /* dataSet */
   7, /* defaultValue */
   "Number of bars for 1st period.", /* hint */

   NULL /* CamelCase name */
};

const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_14_PER2 =
{
   TA_OptInput_IntegerRange, /* type */
   "optInTimePeriod2",       /* paramName */
   0,                        /* flags */

   "Second Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive, /* dataSet */
   14, /* defaultValue */
   "Number of bars fro 2nd period", /* hint */

   NULL /* CamelCase name */
};

const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_28_PER3 =
{
   TA_OptInput_IntegerRange, /* type */
   "optInTimePeriod3",       /* paramName */
   0,                        /* flags */

   "Third Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive, /* dataSet */
   28, /* defaultValue */
   "Number of bars for 3rd period", /* hint */

   NULL /* CamelCase name */
};

static const TA_InputParameterInfo    *TA_ULTOSC_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_ULTOSC_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_ULTOSC_OptInputs[] =
{
  &TA_DEF_UI_TimePeriod_7_PER1,
  &TA_DEF_UI_TimePeriod_14_PER2,
  &TA_DEF_UI_TimePeriod_28_PER3,
  NULL
};

DEF_FUNCTION( ULTOSC,                         /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Ultimate Oscillator",          /* hint */
              "UltOsc",                       /* CamelCase name */
              0                               /* flags */
             );
/* ULTOSC END */

/****************************************************************************
 * Step 2 - Add your TA function to the table.
 *          Keep in alphabetical order. Must be NULL terminated.
 ****************************************************************************/
const TA_FuncDef *TA_DEF_TableU[] =
{
   ADD_TO_TABLE(ULTOSC),
   NULL
};


/* Do not modify the following line. */
const unsigned int TA_DEF_TableUSize =
              ((sizeof(TA_DEF_TableU)/sizeof(TA_FuncDef *))-1);


/****************************************************************************
 * Step 3 - Make sure "gen_code" is executed for generating all other
 *          source files derived from this one.
 *          You can then re-compile the library as usual and you are done!
 ****************************************************************************/
