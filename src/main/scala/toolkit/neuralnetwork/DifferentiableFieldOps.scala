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

import toolkit.neuralnetwork.function._


trait DifferentiableFieldOps extends BasicOps { self: DifferentiableField =>
  def *(that: DifferentiableField) = multiply(this, that)
  def *(that: Float) = multiply(this, that)
  def /(that: DifferentiableField) = divide(this, that)
  def /(that: Float) = divide(this, that)
  def +(that: DifferentiableField) = add(this, that)
  def +(that: Float) = add(this, that)
  def -(that: DifferentiableField) = subtract(this, that)
  def -(that: Float) = subtract(this, that)
}
