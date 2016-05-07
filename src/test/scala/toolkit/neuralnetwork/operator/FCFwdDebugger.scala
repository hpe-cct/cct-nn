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
import cogdebugger._


object FCFwdDebugger extends CogDebuggerApp( new ComputeGraph{
  val batchSize = 12
  val inputShape = Shape(16,16)
  val inputLen = 64
  val outputLen = 20
  val x = VectorField.random(inputShape, Shape(inputLen*batchSize))
  val w = VectorField.random(inputShape, Shape(inputLen*outputLen))
  val y = forwardFC(x,w,batchSize)

  val y2 = VectorField.random(Shape(),Shape(outputLen*batchSize))
  val x2 = backFC(y2,w,batchSize)

  val w2 = weightGradFC(x, y2, batchSize)

  val rhs = reduceSum(fieldReduceSum(x*x2))
  val lhs = reduceSum(fieldReduceSum(y*y2))
  val l2 = reduceSum(fieldReduceSum(w*w2))

  probeAll
})
