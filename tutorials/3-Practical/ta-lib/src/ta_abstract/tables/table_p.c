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
 * This file contains only TA functions starting with the letter 'P' *
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

/* PLUS_DI BEGIN */
static const TA_InputParameterInfo    *TA_PLUS_DI_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_PLUS_DI_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_PLUS_DI_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_14,
  NULL
};

DEF_FUNCTION( PLUS_DI,                     /* name */
              TA_GroupId_MomentumIndicators,   /* groupId */
              "Plus Directional Indicator", /* hint */
              "PlusDI",                     /* CamelCase name */
              TA_FUNC_FLG_UNST_PER          /* flags */
             );

/* PLUS_DI END */

/* PLUS_DM BEGIN */
static const TA_InputParameterInfo    *TA_PLUS_DM_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HL,
  NULL
};

static const TA_OutputParameterInfo   *TA_PLUS_DM_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_PLUS_DM_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_14,
  NULL
};

DEF_FUNCTION( PLUS_DM,                       /* name */
              TA_GroupId_MomentumIndicators, /* groupId */
              "Plus Directional Movement",   /* hint */
              "PlusDM",                      /* CamelCase name */
              TA_FUNC_FLG_UNST_PER           /* flags */
             );

/* PLUS_DM END */

/* PPO BEGIN */
static const TA_InputParameterInfo *TA_PPO_Inputs[] =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_PPO_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_PPO_OptInputs[] =
{ &TA_DEF_UI_Fast_Period,
  &TA_DEF_UI_Slow_Period,
  &TA_DEF_UI_MA_Method,
  NULL
};

DEF_FUNCTION( PPO,                           /* name */
              TA_GroupId_MomentumIndicators, /* groupId */
              "Percentage Price Oscillator", /* hint */
              "Ppo",                         /* CamelCase name */
              0                              /* flags */
             );
/* PPO END */

#if 0
Will be implemented later
/* PVI BEGIN */
static const TA_InputParameterInfo    *TA_PVI_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_CV,
  NULL
};

static const TA_OutputParameterInfo   *TA_PVI_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_PVI_OptInputs[] =
{
  NULL
};

DEF_FUNCTION( PVI,                         /* name */
              TA_GroupId_VolumeIndicators, /* groupId */
              "Positive Volume Index",     /* hint */
              "Pvi",                       /* CamelCase name */
              0                            /* flags */
             );

/* PVI END */
#endif

/****************************************************************************
 * Step 2 - Add your TA function to the table.
 *          Keep in alphabetical order. Must be NULL terminated.
 ****************************************************************************/
const TA_FuncDef *TA_DEF_TableP[] =
{
   ADD_TO_TABLE(PLUS_DI),
   ADD_TO_TABLE(PLUS_DM),
   ADD_TO_TABLE(PPO),
   /* ADD_TO_TABLE(PVI),*/
   NULL
};


/* Do not modify the following line. */
const unsigned int TA_DEF_TablePSize =
              ((sizeof(TA_DEF_TableP)/sizeof(TA_FuncDef *))-1);


/****************************************************************************
 * Step 3 - Make sure "gen_code" is executed for generating all other
 *          source files derived from this one.
 *          You can then re-compile the library as usual and you are done!
 ****************************************************************************/
