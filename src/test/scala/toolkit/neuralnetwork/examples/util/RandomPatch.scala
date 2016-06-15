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

package toolkit.neuralnetwork.examples.util

import cogx.compiler.cpu_operator.Operator
import cogx.platform.cpumemory.readerwriter.{ScalarFieldReader, ScalarFieldWriter}
import cogx.utilities.Random

/**
 * Created by symonsj on 3/16/2016.
 *
 * generate a random guide for cropping based on input range
 * hard-coded to generate 0..26, 0..26 for now
 * 
 */


object RandomPatch extends Operator {
  def compute(in: ScalarFieldReader, out: ScalarFieldWriter): Unit = {
    out.setShape(in.fieldShape)
    //val range = in.read(0)
    //val r = range.toInt
    val rng = new Random()
    val x = Math.abs(rng.nextInt) % 27 // hard-coded for now, future use value from in
    val y = Math.abs(rng.nextInt) % 27
    out.write(0,x)
    out.write(1,y)
  }

}
