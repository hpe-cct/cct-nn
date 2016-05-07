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
 * @author Matthew Pickett
 */
private [neuralnetwork] object fourierFilterGrad {
  def apply(x:VectorField, g:VectorField, filterShape:Shape, batchSize:Int, inputSetSize:Int, filterSetSize:Int) = {
    val xF = fftReal(x)
    val gF = fftReal(g)
    val mac = fourierFilterGradMAC(xF,gF,batchSize, inputSetSize, filterSetSize)
    val f = fftInverseReal(mac)
    cropFilters(f, filterShape)
//    val filterRows = filterShape(0)
//    val filterColumns = filterShape(1)
//    shiftCyclic(f, filterRows/2, filterColumns/2)(0 until filterRows, 0 until filterColumns)
  }
}
