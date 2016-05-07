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

package toolkit.neuralnetwork.util

import libcog._


object NormalizedLowPass {
  def apply(input: Field, timeConstant: Float): Field = {
    require(timeConstant >= 0f && timeConstant <= 1f)

    // Step: input weight, recurrent weight
    // 0: 1.00 0.00
    // 1: 0.50 0.50
    // 2: 0.33 0.67
    // 3: 0.25 0.75
    // 4: 0.20 0.80

    val counter = ScalarField(1f)
    counter <== counter + 1f

    val inWeight = max(1f / counter, timeConstant)
    val recWeight = 1f - inWeight

    val state = Field(input.fieldType)
    state <== recWeight * state + inWeight * input
    state
  }
}
