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

package toolkit.neuralnetwork


object AdjointTestSource {
  def apply(forward: DifferentiableField, gradient: DifferentiableField, consumer: Boolean): DifferentiableField = {
    require(forward.forward.fieldType == gradient.forward.fieldType, "forward and gradient must have identical forward field types")
    require(forward.batchSize == gradient.batchSize, "forward and gradient must have identical batch sizes")

    val _forward = forward

    new DifferentiableField {
      override val gradientConsumer: Boolean = consumer
      override val batchSize: Int = _forward.batchSize
      override val forward: libcog.Field = _forward.forward

      forwardGradient = if (consumer) {
        Some(gradient.forward)
      } else {
        None
      }
    }
  }
}
