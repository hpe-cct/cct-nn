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
  * projectFrame operator
  * @author Matthew Pickett
  */
object ConvFwdDebugger extends CogDebuggerApp( new ComputeGraph{
  val inRows = 64
  val inShape = Shape(inRows, inRows)
  val filterSize = 9
  val filterShape = Shape(filterSize, filterSize)
  val filterNum = 64
  val inputLen = 64
  val batchSize = 64

  val filterSetSize = 4
  val batchSetSize = 4

  //generate random inputs and filter bank
  val x = VectorField.random(inShape, Shape(inputLen * batchSize))
  val f = VectorField.random(filterShape, Shape(filterNum * inputLen)) - 0.5f
  val y = fourierProject(x, f, batchSize, filterSetSize, batchSetSize)
//  val y = projectFFT(x, f, batchSize, filterSetSize, batchSetSize)

  //calculate the difference between the above Fourier approach and projectFrame
  val totalDiffFwd = IndexedSeq.tabulate(batchSize){i=>{
    val batchN = VectorField(Vector(inputLen, j=>j+i*inputLen))
    val xN = vectorElements(x, batchN)
    val yN = blockReduceSum(projectFrame(xN,f,BorderCyclic),inputLen)
    val outputN = VectorField(Vector(filterNum, j=>j+i*filterNum))
    val singleDiff = yN - vectorElements(y, outputN)
    singleDiff
  }}.reduce(_+_)
  probe(totalDiffFwd, "total difference forward")
})
