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
private [neuralnetwork] object fourierProjectMAC extends GPUOperatorHelper {
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

  def apply(x:(Field, Field), f:(Field, Field), batchSize:Int, batchSetSizeDesired:Int, filterSetSizeDesired:Int) = {
    val xR = x._1
    val xI = x._2
    val fR = f._1
    val fI = f._2
    val inputShape = xR.fieldShape
    require(xR.fieldType == xI.fieldType)
    require(fR.fieldType == fI.fieldType)
    require(xR.fieldShape == fR.fieldShape)
    require(xR.tensorShape(0)%batchSize == 0)
    val inputLen = xR.tensorShape(0) / batchSize
    require(fR.tensorShape(0)%inputLen == 0)
    val filterNum = fR.tensorShape(0)/inputLen

    // This operator is described with variants based the following choices for (batchSetSize, filterSetSize)
    val batchSetSizes = Seq(1, 2, 3, 4, 5, 6, 8, 12, 16)
    val filterSetSizes = Seq(1, 2, 3, 4, 5, 6, 8, 12, 16)

    // NVIDIA PASCAL bricks (and earlier) do horribly above this miniTile size
    // threshold, so we exclude these to speed up the tuning process.
    val maxMiniTileElements = 128

    def clipSetSizes(batchSetSize: Int, filterSetSize: Int) = {
      val clippedBatchSetSize = math.min(batchSize, batchSetSize)
      val clippedFilterSetSize = math.min(filterNum, filterSetSize)
      (clippedBatchSetSize, clippedFilterSetSize)
    }

    val parameters =
      if (Convolution.useProfiler) {
        for (batchSetSize <- batchSetSizes;
             filterSetSize <- filterSetSizes
             if (batchSetSize * filterSetSize <= maxMiniTileElements)
        )
          yield clipSetSizes(batchSetSize, filterSetSize)
      }
      else {
        IndexedSeq(clipSetSizes(batchSetSizeDesired, filterSetSizeDesired))
      }

    val variantNames = parameters.toArray.map(p => s"fourierProjectMAC_${p._1}_${p._2}")

    val outputType = new FieldType(inputShape, Shape(filterNum*batchSize), Float32)
    val (outR, outI) = GPUOperator(outputType, outputType, variantNames){ i =>
      val (batchSetSize, filterSetSize) = parameters(i)

      // If (batchSize,filterNum) is not a multiple of (batchSetSize,filterSetSize), then round up.
      val batchSetNum = (batchSize + batchSetSize - 1)/batchSetSize
      val filterSetNum = (filterNum + filterSetSize - 1)/filterSetSize

      // Set up initial workfield parameters before a possible "transpose" of the assignment of threads
      // in the workfield to elements of the input fields
      val Rows = inputShape(0)
      val Columns = inputShape(1)
      val TensorElements = batchSetNum * filterSetNum

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

      val batchSetIndex = tensorElement / filterSetNum
      val filterSetIndex = tensorElement % filterSetNum

      // Rather than put in tests for boundary conditions to prevent a thread from reading or writing out-of-bounds
      // field data if (batchSize,filterNum) is not a multiple of (batchSetSize,filterSetSize), we
      // shift the base batch and filter indices of the last set in each dimension to overlap the work of the
      // prior set.  This means that multiple workgroups might be writing the same output locations, but with
      // the same values, which is OK.

      val baseBatchIndex = intVar(batchSetIndex * batchSetSize)
      if (batchSize % batchSetSize != 0) {
        _if (batchSetIndex >= batchSetNum - 1) {
          baseBatchIndex := batchSize - batchSetSize
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
      val xRlocal = _floatArray(batchSetSize)
      val xIlocal = _floatArray(batchSetSize)
      val fRlocal = _floatArray(filterSetSize)
      val fIlocal = _floatArray(filterSetSize)
      val outRlocal = _floatArray(batchSetSize*filterSetSize)
      val outIlocal = _floatArray(batchSetSize*filterSetSize)
      val outputIndex = _intVar()
      _for(outputIndex := 0, outputIndex < batchSetSize*filterSetSize, outputIndex+=1 ){
        outRlocal(outputIndex) := 0
        outIlocal(outputIndex) := 0
      }

      val inputIndex = _intVar()
      val xIndex = _intVar()
      val fIndex = _intVar()
      _for(inputIndex :=0, inputIndex < inputLen, inputIndex +=1){
        //load input block into local memory
        _for(xIndex :=0, xIndex < batchSetSize, xIndex +=1){
          val inputOffset = baseBatchIndex*inputLen + xIndex*inputLen + inputIndex
          xRlocal(xIndex) := _readTensorElement(xR, row, column, inputOffset)
          xIlocal(xIndex) := _readTensorElement(xI, row, column, inputOffset)
        }

        //load filter block into local memory
        _for(fIndex :=0, fIndex < filterSetSize, fIndex +=1){
          val filterOffset = baseFilterIndex*inputLen + fIndex*inputLen
          fRlocal(fIndex) := _readTensorElement(fR, row, column, filterOffset+inputIndex)
          fIlocal(fIndex) := _readTensorElement(fI, row, column, filterOffset+inputIndex)
        }

        //accumulate outer product
        _for(xIndex :=0, xIndex < batchSetSize, xIndex +=1){
          _for(fIndex :=0, fIndex < filterSetSize, fIndex +=1){
            // This approach to accumulation uses 2 fma's, the use of '+=' uses only 1; results in a 4% perf gain.
            val idx = xIndex*filterSetSize+fIndex
            //Note: implicitly conjugating (negating) fIlocal since projection is defined as cross corrleation
            outRlocal(idx) := outRlocal(idx) + xRlocal(xIndex)*fRlocal(fIndex) + xIlocal(xIndex)*fIlocal(fIndex)
            outIlocal(idx) := outIlocal(idx) + xIlocal(xIndex)*fRlocal(fIndex) - xRlocal(xIndex)*fIlocal(fIndex)
          }
        }
      }

      //write the accumulated output
      val batchSetOffset = baseBatchIndex*filterNum
      val filterSetOffset = baseFilterIndex
      _for(outputIndex := 0, outputIndex < batchSetSize*filterSetSize, outputIndex+=1 ){
        val filterBlockOffset = (outputIndex/filterSetSize)*filterNum
        val curOutput = batchSetOffset+filterSetOffset + filterBlockOffset+ outputIndex%filterSetSize
        _writeTensorElement(_out0, outRlocal(outputIndex), row, column, curOutput)
        _writeTensorElement(_out1, outIlocal(outputIndex), row, column, curOutput)
      }
    }
    (outR, outI)
  }
}
