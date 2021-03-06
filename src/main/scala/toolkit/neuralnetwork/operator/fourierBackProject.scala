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
private [neuralnetwork] object fourierBackProject {
  def apply(g:VectorField, f:VectorField, batchSize:Int, batchSetSize:Int, filterSetSize:Int) = {
    val fExpanded = expandFilters(f, g.fieldShape)
    val gF = fftReal(g)
    val fF = fftReal(fExpanded)
    val mac = fourierBackProjectMAC(gF, fF, batchSize, batchSetSize, filterSetSize)
    fftInverseReal(mac)
  }
}
