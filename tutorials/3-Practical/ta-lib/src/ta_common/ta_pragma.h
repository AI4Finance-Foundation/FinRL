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
 *  CM		 Craig Miller  (c-miller@users.sourceforge.net)
 *
 * Change history:
 *
 *  MMDDYY BY    Description
 *  -------------------------------------------------------------------
 *  011707 CM    First version.
 */

/* Description:
 * 
 * Visual Studio 2005 has extended the C Run-Time Library by including "secure"
 * runtime functions and deprecating the previous function prototypes.  Since
 * we need to use the previous prototypes to maintain compatibility with other
 * platform compilers we are going to disable the deprecation warnings when 
 * compiling with Visual Studio 2005.
 * 
 * Note: this header must be the first inclusion referenced by the code file 
 * needing these settings!!!!!
 * 
 */

#ifndef TA_PRAGMA_H
#define TA_PRAGMA_H

#if (_MSC_VER >= 1400)       // VC8+ nmake and VS2005
  
	#ifndef _CRT_SECURE_NO_DEPRECATE	//turn off MS 'safe' CRT library routines
		#define _CRT_SECURE_NO_DEPRECATE 1
	#endif
	
// There are additional macros that may be needed in the future, so we'll list them here
	//#ifndef _CRT_SECURE_NO_WARNINGS	//turn off MS 'safe' CRT library routines
	//	#define _CRT_SECURE_NO_WARNINGS 1
	//#endif
	//
	//#ifndef _SCL_SECURE_NO_DEPRECATE	//turn off MS 'safe' C++RT library routines
	//	#define _SCL_SECURE_NO_DEPRECATE 1
	//#endif
	//#ifndef _SCL_SECURE_NO_WARNINGS
	//	#define _SCL_SECURE_NO_WARNINGS 1
	//#endif
	//
	//#ifndef _CRT_NONSTDC_NO_DEPRECATE //turn off MS POSIX replacements library routines
	//	#define _CRT_NONSTDC_NO_DEPRECATE 1
	//#endif

#endif   // VC8+

#endif	//TA_PRAGMA_H
