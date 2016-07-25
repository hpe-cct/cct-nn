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

package toolkit.neuralnetwork.examples

import java.io.File

import cogio.fieldstate.FieldState
import libcog._
import toolkit.neuralnetwork.source.ByteDataSource
import toolkit.neuralnetwork.util.NormalizedLowPass


object ComputeMean extends App {
  val cg = new ComputeGraph {
    val prefix = "training"
    val batchSize = 1
    val dir = new File(System.getProperty("user.home"), "cog/data/cifar10")

    val raw = ByteDataSource(new File(dir, s"${prefix}_data.bin").toString, Shape(32, 32), 3, batchSize).forward
    val mean = NormalizedLowPass(raw, 0.0001f)

    probe(mean)
  }

  val mean = try {
    cg.step(50000)
    FieldState.read(cg.read(cg.mean))
  } finally {
    cg.release
  }

  mean.saveToFile(new File("cifar_mean.field"))
}
