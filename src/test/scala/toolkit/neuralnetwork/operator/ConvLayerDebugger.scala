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

/**
 * @author Matthew Pickett
 */
object ConvLayerDebugger extends CogDebuggerApp( new ComputeGraph{
  val inRows = 64
  val inShape = Shape(inRows, inRows)
  val filterSize = 9
  val filterShape = Shape(filterSize, filterSize)
  val filterNum = 128
  val inputLen = 64
  val batchSize = 128

  //good parameters for titan black, use 4,4 for 680?
  val filterSetSize = 4
  val batchSetSize = 8

  //generate random inputs and filter bank
  val x = VectorField.random(inShape, Shape(inputLen * batchSize))
  val f = VectorField.random(filterShape, Shape(filterNum * inputLen))
  val grad = VectorField.random(inShape, Shape(filterNum * batchSize))

  val fwd = fourierProject(x, f, batchSize, filterSetSize, batchSetSize)
  val back = fourierBackProject(grad, f, batchSize, filterSetSize, batchSetSize)
  val filterGrad = fourierFilterGrad(x, grad, filterShape, batchSize, filterSetSize, batchSetSize)


/*    val fwd = projectFFT(x, f, batchSize, filterSetSize, batchSetSize)
    val back = backProjectFFT(grad, f, batchSize, filterSetSize, batchSetSize)
    val filterGrad = filterGradientFFT(x, grad, filterShape, batchSize, filterSetSize, batchSetSize)*/
})
