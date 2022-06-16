/* TA-LIB Copyright (c) 1999-2006, Mario Fortier
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
 *  BT       Barry Tsung
 *
 * Change history:
 *
 *  MMDDYY BY    Description
 *  -------------------------------------------------------------------
 *  012706 BT    First version.
 */

/* Description:
 *       Do formating of Java source code files.
 */

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

public class PrettyCode {
   int line;
   int level;
   BufferedReader in;
   PrintWriter    out;
   
   String indentString = "   ";
   
   boolean insideOfComment;
   
   boolean unfinishedStatement;
   
   String lineBuffer;
   
   public PrettyCode(String inFileName, String outFileName) throws FileNotFoundException{
      this.in  = new BufferedReader(new FileReader(inFileName));
      this.out = new PrintWriter(outFileName);
   }
   
   public PrettyCode process() throws IOException{
      while( (lineBuffer=in.readLine())!=null ){
         processLine(lineBuffer);
         out.println();
         line++;
      }
      out.flush();
      return this;
   }
   
   public PrettyCode close() throws IOException{
      in.close();
      out.close();
      return this;
   }
   
   void processLine(String buffer){
      if( level > 0 ){
         buffer=buffer.trim();
      }
      if( insideOfComment ){
         appendIndents("");
         buffer = processComment(buffer);
         while((buffer=processBuffer(buffer))!=null);
      }else{
         appendIndents(buffer);
         while((buffer=processBuffer(buffer))!=null);
      }
   }
   
   String processBuffer(String buffer){
      if( buffer == null ){ return null; }
      if( buffer.startsWith("/*")){
         if( insideOfComment ){
            error("comment inside of comment");
         }
         insideOfComment=true;
         out.print(buffer.substring(0,2));
         buffer=processComment(substring(buffer,2));
      }else if(buffer.startsWith( "//")){
         out.print(buffer);
      }else if(buffer.startsWith("\"")){
         unfinishedStatement=true;
         int i=buffer.indexOf("\"");
         if( i == -1 ){
            error("no matching ending \"");
         }else{
            out.print(buffer.substring(0,i+1));
            return buffer.substring(i+1);
         }         
      }else{
         for(int i=0;i<buffer.length();i++){
            char c = buffer.charAt(i);
            char cc = (i == (buffer.length()-1))? 0:buffer.charAt(i+1);            
            if( c == '{' ){
               unfinishedStatement=false;
               level++;
               out.print(buffer.substring(0, i+1));
               return substring(buffer, i+1);
            }else if( c == '}' ){
               unfinishedStatement=false;
               level--;
               out.print(buffer.substring(0, i+1));
               return substring(buffer, i+1);         
            }else if( c == '"'){
               out.print(buffer.substring(0, i));
               return substring(buffer, i);
            }else if( c == '/' && cc == '/' ){
               out.print(buffer);
               return null;
            }else if( c == '/' && cc == '*' ){
               out.print(buffer.substring(0, i));
               return substring(buffer, i);
            }else if( c == ' ' ){
               
            }else if( c == ';' ){
               unfinishedStatement=false;
            }else if( c == ')' && cc == ',' ){
               unfinishedStatement=false;
               i++;
            }else{
               unfinishedStatement=true;
            }
         }
         out.print(buffer);
      }
      return null;
   }

   String processComment(String buf){
      int i=buf.indexOf("*/");
      if( i == -1 ){
         out.print(buf);
      }else{
         insideOfComment=false;
         out.print(buf.substring(0,i+2));
         return substring(buf, i+2);
      }
      return null;
   }
   
   void appendIndents(String buf){
      int n=level;
      if( unfinishedStatement ){
         if(!buf.startsWith("{")){
            n++;
         }
      }
      if( buf.startsWith("}") ){
         n--;
      }
      for(int i=0;i<n;i++){ out.print(indentString); }
   }
   
   void error(String msg){
      System.err.println("ERROR one line ["+line+"] : "+msg);
   }
   
   String substring(String s, int i){
      if( i>(s.length()-1)){return "";}
      return s.substring(i);
   }
   
   public static boolean verify(String fileName1, String fileName2) throws IOException{
      BufferedReader f1 = new BufferedReader(new FileReader(fileName1));
      BufferedReader f2 = new BufferedReader(new FileReader(fileName2));
      int line=0;
      String buf1, buf2;
      while((buf1=f1.readLine())!= null){
         if((buf2=f2.readLine())==null){
            System.err.println(fileName2+" is shorter than "+fileName1);
            return false;
         }
         if( !buf1.trim().equals(buf2.trim()) ){
            System.err.println("Error on line "+line);
            System.err.println(fileName1+"='"+buf1+"'");
            System.err.println(fileName2+"='"+buf2+"'");
            return false;
         }
         line++;
      }
      if( (buf2 = f2.readLine()) != null ){
         System.err.println(fileName1+" is shorter than "+fileName2);
         return false;         
      }
      return true;
   }   
}

