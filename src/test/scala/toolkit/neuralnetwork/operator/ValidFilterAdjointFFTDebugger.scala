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
object ValidFilterAdjointFFTDebugger extends CogDebuggerApp(new ComputeGraph{
  val inRows = 64
  val inShape = Shape(inRows, inRows)
  val filterSize = 9
  val filterShape = Shape(filterSize, filterSize)
  val filterNum = 4
  val inputLen = 4
  val batchSize = 2

  val x = VectorField.random(inShape, Shape(1))
  val xF = fft(x)
  val xFC = conjugate(xF)
  val k = VectorField.random(filterShape, Shape(1))
  val kF = fft(shiftCyclic(expand(flip(k),BorderZero, inShape), -filterSize/2, -filterSize/2))

  val ccFFT = realPart(fftInverse(xF*kF))(filterSize/2 until inRows-filterSize/2, filterSize/2 until inRows-filterSize/2)

  val ccDirect = projectFrame(x,k,BorderValid)
  val ccDiff = ccDirect-ccFFT
  val g = ccDirect


  val g2 = VectorField.random(Shape(inRows-filterSize+1, inRows-filterSize+1), Shape(1))
  val k2Direct = crossCorrelateFilterAdjoint(x, g2, BorderValid)
  val g2F = fft(shiftCyclic(expand(g2, BorderZero, inShape), filterSize/2, filterSize/2))
  val k2FFT = flip(shiftCyclic(realPart(fftInverse(xFC*g2F)), filterSize/2, filterSize/2)(0 until filterSize, 0 until filterSize))
  val k2Diff = k2Direct-k2FFT

  val k2 = k2Direct
  probe(reduceSum(fieldReduceSum(g2*g)), "rhs")
  probe(reduceSum(fieldReduceSum(k2*k)), "lhs")



//  val conv = convolve(f, f1, BorderCyclic)

  probeAll
})
