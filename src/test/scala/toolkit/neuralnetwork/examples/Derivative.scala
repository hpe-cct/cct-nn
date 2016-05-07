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

import libcog._
import cogdebugger._
import toolkit.neuralnetwork.DifferentiableField
import toolkit.neuralnetwork.function._
import toolkit.neuralnetwork.source.{ByteDataSource, RandomSource}
import toolkit.neuralnetwork.util.{Norm, NormalizedLowPass}


object Derivative extends CogDebuggerApp(new ComputeGraph {
  /*val dir = new File(System.getProperty("user.home"), "cog/data/MNIST")
  val batchSize = 1

  val in = ByteDataSource(new File(dir, "train-images.idx3-ubyte").toString,
    Shape(28, 28), 1, batchSize, headerLen = 16)

  val state = TrainableState(Shape(28, 28), Shape(10), 0.0f, 0.9f, 0.0005f)*/

  //val out = FullyConnected(in, state)

  //probe(in.forward)
  //probe(out.forward)
  //probe(out.totalDerivative(), "derivative")

  val dx = 0.01f
  val positiveXOnly = true
  val inputShapes = Seq(Shape(), Shape())
  val inputLens = Seq(10, 10)
  val batchSizes = Seq(1024, 1024)

  object OneHotSource {
    def apply(vectorLen: Int, batchSize: Int): DifferentiableField = {
      require(vectorLen > 0)
      val _batchSize = batchSize

      new DifferentiableField {
        val seed = Random.nextSeed
        val rng = new Random(seed)
        val fieldShape = Shape()
        val tensorShape = Shape(vectorLen * _batchSize)

        val nextValue = {
          () => {
            Some(
              new Iterator[Vector] {
                var empty = false
                val classes = Array.tabulate(batchSize) {
                  _ => rng.nextInt(vectorLen)
                }

                val v = Vector(vectorLen * batchSize, i => {
                  // Which sample in the batch are we on?
                  val s = i / vectorLen
                  // Which class index inside sample 's'?
                  val j = i % batchSize

                  // If the class number matches the class index for this
                  // sample, then 1f, else 0f
                  if (classes(s) == j) {
                    1f
                  } else {
                    0f
                  }
                })


                def next(): Vector = {
                  empty = true
                  v
                }

                def hasNext: Boolean = empty
              }
            )
          }
        }

        val resetHook = {
          () => rng.setSeed(seed)
        }

        override val gradientConsumer: Boolean = true
        override val batchSize: Int = _batchSize
        override val forward: libcog.Field = new VectorSensor(fieldShape, tensorShape, nextValue, resetHook)
      }
    }
  }

  def offset(i: DifferentiableField) = if (positiveXOnly) {
    i
  } else {
    i - 0.5f
  }

  def finiteDifferenceErr(f: Seq[DifferentiableField] => DifferentiableField): (Field, Field, Field) = {
    /*val x = Seq.tabulate(inputLens.length) {
      i => offset(RandomSource(inputShapes(i), inputLens(i), batchSizes(i),
        gradientConsumer = true, seed = Some(Random.nextSeed)))
    }*/

    val x =
      Seq(offset(RandomSource(inputShapes(0), inputLens(0),
        batchSizes(0), gradientConsumer = true, seed = Some(Random.nextSeed))),
        OneHotSource(inputLens(1), batchSizes(1)))

    val xPositive = x.map(f => f + dx)
    val xNegative = x.map(f => f - dx)

    val numeric = (f(xPositive).forward - f(xNegative).forward) / (2f * dx)
    val symbolic = f(x).totalDerivative()

    val err = Norm.L1(numeric, symbolic)
    (numeric, symbolic, NormalizedLowPass(err, 0.0001f))
  }

  def fn(s: Seq[DifferentiableField]) = {
    require(s.length == 2)
    CrossEntropySoftmax(s.head, s(1))
  }

  val (numeric, symbolic, err) = finiteDifferenceErr(fn)

  probe(numeric)
  probe(symbolic)
  probe(err)

  object TestSource {
    def apply(fieldShape: Shape, vectorLen: Int, batchSize: Int): DifferentiableField = {
      val _batchSize = batchSize

      new DifferentiableField {
        val b = RandomSource(fieldShape, vectorLen, _batchSize) - 0.5f
        val x = RandomSource(fieldShape, vectorLen, _batchSize) - 0.5f

        override val gradientConsumer: Boolean = true
        override val batchSize: Int = _batchSize
        override val forward: libcog.Field = b.forward

        forwardGradient = Some(x.forward)
      }
    }
  }

  // Pick a random starting point
  //val b = RandomSource(Shape(10, 10), 1, 1, gradientConsumer = true) - 0.5f
  // A random inbound gradient
  //val x = RandomSource(Shape(10, 10), 1, 1, gradientConsumer = true) - 0.5f
  // And a random top-down signal
  /*val y = RandomSource(Shape(10, 10), 1, 1) - 0.5f

  val b = TestSource(Shape(10, 10), 1, 1)

  // At point `b`, propagate x forward and y backwards
  val relu = ReLU(b)
  val fProp = relu.totalDerivative()
  relu.activateSGD(initField = y.forward, invokeCallbacks = false)
  val bProp = b.backward.get

  probe(relu.forward)
  probe(fProp)
  probe(bProp)

  // The forward stream dotted with `y` should be ~= the backward stream dotted with `x`
  val forward = dot(fProp, y.forward)
  val backward = dot(bProp, b.totalDerivative())

  probe(forward)
  probe(backward)
  probe(NormalizedLowPass(Norm.L1(forward, backward), 0.0001f), "adjointErr")*/
})
