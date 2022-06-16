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
#ifndef TA_COMMON_H
#define TA_COMMON_H

/* The following macros are used to return internal errors.
 * The Id can be from 1 to 999 and translate to the user 
 * as the return code 5000 to 5999.
 *
 * Everytime you wish to add a new fatal error code,
 * use the "NEXT AVAILABLE NUMBER" and increment the
 * number in this file.
 *
 * NEXT AVAILABLE NUMBER: 181
 */
#define TA_INTERNAL_ERROR(Id) ((TA_RetCode)(TA_INTERNAL_ERROR+Id))

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <limits.h>
#include <float.h>

#ifndef TA_DEFS_H
   #include "ta_defs.h"
#endif

/* Some functions to get the version of TA-Lib.
 *
 * Format is "Major.Minor.Patch (Month Day Year Hour:Min:Sec)"
 * 
 * Example: "1.2.0 (Jan 17 2004 23:59:59)"
 *
 * Major increments indicates an "Highly Recommended" update.
 * 
 * Minor increments indicates arbitrary milestones in the
 * development of the next major version.
 *
 * Patch are fixes to a "Major.Minor" release.
 */
const char *TA_GetVersionString( void );

/* Get individual component of the Version string */
const char *TA_GetVersionMajor ( void );
const char *TA_GetVersionMinor ( void );
const char *TA_GetVersionBuild ( void );
const char *TA_GetVersionDate  ( void );
const char *TA_GetVersionTime  ( void );

/* Misc. declaration used throughout the library code. */
typedef double TA_Real;
typedef int    TA_Integer;

/* General purpose structure containing an array of string. 
 *
 * Example of usage:
 *    void printStringTable( TA_StringTable *table )
 *    {
 *       int i;
 *       for( i=0; i < table->size; i++ )
 *          cout << table->string[i] << endl;
 *    }
 *
 */
typedef struct TA_StringTable
{
    unsigned int size;    /* Number of string. */
    const char **string;  /* Pointer to the strings. */

   /* Hidden data for internal use by TA-Lib. Do not modify. */
   void *hiddenData;
} TA_StringTable;
/* End-user can get additional information related to a TA_RetCode. 
 *
 * Example:
 *        TA_RetCodeInfo info;
 *
 *        retCode = TA_Initialize( ... );
 *
 *        if( retCode != TA_SUCCESS )
 *        {
 *           TA_SetRetCodeInfo( retCode, &info );
 *           printf( "Error %d(%s): %s\n",
 *                   retCode,
 *                   info.enumStr,
 *                   info.infoStr );
 *        }
 *
 * Would display:
 *        "Error 1(TA_LIB_NOT_INITIALIZE): TA_Initialize was not sucessfully called"
 */
typedef struct TA_RetCodeInfo
{
   const char *enumStr; /* Like "TA_IP_SOCKETERROR"     */
   const char *infoStr; /* Like "Error creating socket" */      
} TA_RetCodeInfo;

/* Info is always returned, even when 'theRetCode' is invalid. */
void TA_SetRetCodeInfo( TA_RetCode theRetCode, TA_RetCodeInfo *retCodeInfo );
 
/* TA_Initialize() initialize the ressources used by TA-Lib. This
 * function must be called once prior to any other functions declared in
 * this file.
 *
 * TA_Shutdown() allows to free all ressources used by TA-Lib. Following
 * a shutdown, TA_Initialize() must be called again for re-using TA-Lib.
 *
 * TA_Shutdown() should be called prior to exiting the application code.
 */
TA_RetCode TA_Initialize( void );
TA_RetCode TA_Shutdown( void );

#ifdef __cplusplus
}
#endif

#endif
