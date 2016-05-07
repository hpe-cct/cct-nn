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

import cogdebugger.CogDebuggerApp
import libcog._

object BorderReduceSumDebugger extends CogDebuggerApp( new ComputeGraph{
  val inSize = 12
  val vecLen = 2
  val borderSize = 5
//  val x = VectorField.random(Shape(inSize,inSize),Shape(vecLen))
  val x = ScalarField.random(Shape(inSize,inSize))
  val y = borderReduceSum(x, borderSize)
  val y2 = ScalarField.random(Shape(inSize-2*borderSize, inSize-2*borderSize))
  val x2 = shiftCyclic(expand(y2, BorderClamp, Shape(inSize, inSize)), borderSize, borderSize)

  val rhs = reduceSum(fieldReduceSum(x*x2))
  val lhs = reduceSum(fieldReduceSum(y*y2))
  val diff = rhs - lhs
  probeAll
})
