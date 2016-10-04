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
import toolkit.neuralnetwork.function.Convolution

/**
 * @author Matthew Pickett
 */
private [neuralnetwork] object fourierFilterGradMAC extends GPUOperatorHelper {
  /** Should the workgroup thread neighbors along the "row dimension" be in the tensorElement dimension instead?
    *
    * This kernel has been shown to have higher performance when the threads of a workgroup are not in the same
    * tensor plane (i.e. they have different _tensorElement values, but the same _row and _column).  This is achieved
    * with a bit of a trick- the workField shape is based on a transposed version of the input, then the _row and
    * _tensorElement variables are effectively swapped.
    *
    * The performance improvement from the transpose results from better shared use of the L2 cache by consecutively
    * launched workgroups.
    */
  val Transpose = true
  /** Having workgroup columns a multiple of the NVIDIA warpsize of 32 is often best when doing coalesced loads. */
  val OptimizeWorkGroupShape = true

  def apply(x:(Field, Field), g:(Field, Field), batchSize:Int, inputSetSizeDesired:Int, filterSetSizeDesired:Int) = {
    val xR = x._1
    val xI = x._2
    val gR = g._1
    val gI = g._2
    val inputShape = xR.fieldShape
    require(xR.fieldType == xI.fieldType)
    require(gR.fieldType == gI.fieldType)
    require(xR.fieldShape == gR.fieldShape)
    require(xR.tensorShape(0)%batchSize == 0)
    val inputLen = xR.tensorShape(0) / batchSize
    require(gR.tensorShape(0)%batchSize == 0)
    val filterNum = gR.tensorShape(0)/batchSize

    // This operator is described with variants based the following choices for (inputSetSize, filterSetSize)
    val inputSetSizes = Seq(1, 2, 3, 4, 5, 6, 8, 12, 16)
    val filterSetSizes = Seq(1, 2, 3, 4, 5, 6, 8, 12, 16)

    // NVIDIA PASCAL bricks (and earlier) do horribly above this miniTile size
    // threshold, so we exclude these to speed up the tuning process.
    val maxMiniTileElements = 128

    def clipSetSizes(inputSetSize: Int, filterSetSize: Int) = {
      val clippedInputSetSize = math.min(inputLen, inputSetSize)
      val clippedFilterSetSize = math.min(filterNum, filterSetSize)
      (clippedInputSetSize, clippedFilterSetSize)
    }

    val parameters =
      if (Convolution.useProfiler) {
        for (inputSetSize <- inputSetSizes;
             filterSetSize <- filterSetSizes
             if (inputSetSize * filterSetSize <= maxMiniTileElements)
        )
          yield clipSetSizes(inputSetSize, filterSetSize)
      }
      else {
        IndexedSeq(clipSetSizes(inputSetSizeDesired, filterSetSizeDesired))
      }

    val variantNames = parameters.toArray.map(p => s"fourierFilterGradMAC_${p._1}_${p._2}")

    val outputType = new FieldType(inputShape, Shape(filterNum*inputLen), Float32)
    val (outR, outI) = GPUOperator(outputType, outputType, variantNames){ i =>

      val (inputSetSize, filterSetSize) = parameters(i)

      // If (inputLen,filterNum) is not a multiple of (inputSetSize,filterSetSize), then round up.
      val inputSetNum = (inputLen + inputSetSize - 1)/inputSetSize
      val filterSetNum = (filterNum + filterSetSize - 1)/filterSetSize

      // Standard workfield parameters before a possible "transpose" of the assignment of threads
      // in the workfield to elements of the input fields
      val Rows = inputShape(0)
      val Columns = inputShape(1)
      val TensorElements = inputSetNum * filterSetNum

      val (rows, columns, tensorElements) =
        if (Transpose)
          (TensorElements, Columns, Rows)
        else
          (Rows, Columns, TensorElements)

      if (OptimizeWorkGroupShape) {
        val WorkGroupThreads = 256
        // If there are at least 32 columns, set that as localColumns since this matches the NVIDIA warp size
        val localColumns = math.min(32, columns)
        val localRows = WorkGroupThreads / localColumns
        require(localColumns * localRows == WorkGroupThreads, s"Expecting power-of-2 input columns, found $columns")
        _localThreads(Shape(localRows, localColumns))
      }

      // Set possibly transposed workfield
      _globalThreads(Shape(rows, columns), Shape(tensorElements))

      // Swap _row and _tensorElement variables if the transpose trick is employed
      val (row, column, tensorElement) =
        if (Transpose)
          (_tensorElement, _column, _row)
        else
          (_row, _column, _tensorElement)

      val filterSetIndex = tensorElement / inputSetNum
      val inputSetIndex = tensorElement % inputSetNum

      // Rather than put in tests for boundary conditions to prevent a thread from reading or writing out-of-bounds
      // field data if (inputLen,filterNum) is not a multiple of (inputSetSize,filterSetSize), we
      // shift the base input and filter indices of the last set in each dimension to overlap the work of the
      // prior set.  This means that multiple workgroups might be writing the same output locations, but with
      // the same values, which is OK.

      val baseInputIndex = intVar(inputSetIndex * inputSetSize)
      if (inputLen % inputSetSize != 0) {
        _if (inputSetIndex >= inputSetNum - 1) {
          baseInputIndex := inputLen - inputSetSize
        }
      }

      val baseFilterIndex = intVar(filterSetIndex * filterSetSize)
      if (filterNum % filterSetSize != 0) {
        _if(filterSetIndex >= filterSetNum - 1) {
          baseFilterIndex := filterNum - filterSetSize
        }
      }

      //allocate local memory and zero output accumulator
      //total memory required = 2*bSS + 2*fSS + 2*bSS*fSS
      val xRlocal = _floatArray(inputSetSize)
      val xIlocal = _floatArray(inputSetSize)
      val gRlocal = _floatArray(filterSetSize)
      val gIlocal = _floatArray(filterSetSize)
      val outRlocal = _floatArray(inputSetSize*filterSetSize)
      val outIlocal = _floatArray(inputSetSize*filterSetSize)
      val outputIndex = _intVar()
      _for(outputIndex := 0, outputIndex < inputSetSize*filterSetSize, outputIndex+=1 ){
        outRlocal(outputIndex) := 0
        outIlocal(outputIndex) := 0
      }

      val batchIndex = _intVar()
      val xIndex = _intVar()
      val gIndex = _intVar()
      _for(batchIndex :=0, batchIndex < batchSize, batchIndex +=1){
        //load input block into local memory
        _for(xIndex :=0, xIndex < inputSetSize, xIndex +=1){
          val readIndex = batchIndex*inputLen + baseInputIndex + xIndex
          xRlocal(xIndex) := _readTensorElement(xR, row, column, readIndex)
          xIlocal(xIndex) := _readTensorElement(xI, row, column, readIndex)
        }

        //load filter block into local memory
        _for(gIndex :=0, gIndex < filterSetSize, gIndex +=1){
          val readIndex = batchIndex*filterNum  + baseFilterIndex + gIndex
          gRlocal(gIndex) := _readTensorElement(gR, row, column, readIndex)
          gIlocal(gIndex) := _readTensorElement(gI, row, column, readIndex)
        }

        //accumulate outer product
        _for(gIndex :=0, gIndex < filterSetSize, gIndex +=1){
          _for(xIndex :=0, xIndex < inputSetSize, xIndex +=1){
            // This approach to accumulation uses 2 fma's, the use of '+=' uses only 1; results in a 4% perf gain.
            val idx = gIndex*inputSetSize+xIndex
            //Note: implicitly conjugating (negating) gIlocal
            outRlocal(idx) := outRlocal(idx) + xRlocal(xIndex)*gRlocal(gIndex) + xIlocal(xIndex)*gIlocal(gIndex)
            outIlocal(idx) := outIlocal(idx) + xIlocal(xIndex)*gRlocal(gIndex) - xRlocal(xIndex)*gIlocal(gIndex)
          }
        }
      }

      //write the accumulated output
      val filterSetOffset = baseFilterIndex*inputLen
      val inputSetOffset = baseInputIndex
      _for(outputIndex := 0, outputIndex < inputSetSize*filterSetSize, outputIndex+=1 ){
        val inputBlockOffset = (outputIndex/inputSetSize)*inputLen
        val curOutput = filterSetOffset + inputSetOffset + inputBlockOffset+ outputIndex%inputSetSize
        _writeTensorElement(_out0, outRlocal(outputIndex), row, column, curOutput)
        _writeTensorElement(_out1, outIlocal(outputIndex), row, column, curOutput)
      }
    }

    (outR, outI)
  }
}
