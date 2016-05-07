/*
 * (c) Copyright 2016 Hewlett Packard Enterprise Development LP
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package toolkit.neuralnetwork.operator

import libcog._

/**
  * Created by Dick Carter on 10/29/2014.
  *
  * Supplies helpful user-level routines for writing GPUOperators.
  */
private [operator] trait GPUOperatorHelper {

  // The statement:
  //
  // val foo = intVar(expr)
  //
  // can be used more concisely in place of:
  //
  // val foo = _intVar()
  // foo := expr
  //
  // Note that both these forms force a variable to be emitted in the OpenCL code, unlike:
  //
  // val foo = expr
  //
  // which creates a GPUExpression that is inlined everywhere it's used.  One might be tempted to use
  // intVar(expr) to eliminate repeated inlining, but the inlining does not seem to effect performance
  // since NVIDIA's LLVM-based compiler appears to perform common subexpression elimination.  A valid use
  // of intVar(expr) might be to make the OpenCL more readable, but the Cog compiler's use of varNNN names
  // makes this a toss-up.

  /** Make an integer GPUVariable and assign to it in one statement */
  def intVar(expr: GPUExpression): GPUVariable = {
    val gpuVariable = _intVar()
    gpuVariable := expr
    gpuVariable
  }
}
