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

/** A debugger test to manually check difference between Fourier approach and
  *  the crossCorrelateFilterAdjoint operator
  * @author Matthew Pickett
  */
object ConvGradDebugger extends CogDebuggerApp( new ComputeGraph{
  val inRows = 64
  val inShape = Shape(inRows, inRows)
  val filterSize = 9
  val filterShape = Shape(filterSize, filterSize)
  val filterNum = 36
  val inputLen = 35
  val batchSize = 5

  val inputSetSize = 5
  val filterSetSize = 3

  //generate random inputs and filter bank
  val x = VectorField.random(inShape, Shape(inputLen * batchSize))
  val grad = VectorField.random(inShape, Shape(filterNum * batchSize))
  val filterGrad = fourierFilterGrad(x, grad, filterShape, batchSize, inputSetSize, filterSetSize)
//  val filterGrad = filterGradientFFT(x, grad, filterShape, batchSize, filterSetSize, batchSetSize)

  //calculate the difference between the above Fourier approach and projectFrame
  val totalFilterGrad = IndexedSeq.tabulate(batchSize){i=>{
    val batchNx = VectorField(Vector(inputLen, j=>j+i*inputLen))
    val newShape = Shape(inRows + filterSize-1, inRows + filterSize-1)
    val xN = shiftCyclic(expand(vectorElements(x, batchNx), BorderCyclic, newShape), filterSize/2, filterSize/2)
    val batchNg = VectorField(Vector(filterNum, j=>j+i*filterNum))
    val gN = vectorElements(grad, batchNg)
    val result = crossCorrelateFilterAdjoint(xN, gN, BorderValid)
    result
  }}.reduce(_+_)
  probe(totalFilterGrad)

  probe(totalFilterGrad - filterGrad, "total diff")
})