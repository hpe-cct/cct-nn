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


object LowPass {
  def apply(input: Field, timeConstant: Float): Field = {
    require(timeConstant >= 0f && timeConstant <= 1f)
    val state = Field(input.fieldType)
    state <== (1f - timeConstant) * state + timeConstant * input
    state
  }
}
