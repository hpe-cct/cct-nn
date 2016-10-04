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
private[neuralnetwork] object fourierBackProjectMAC extends GPUOperatorHelper {
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

  def apply(g:(Field, Field), f:(Field, Field), batchSize:Int, batchSetSizeDesired:Int, inputSetSizeDesired:Int) = {
    val gR = g._1
    val gI = g._2
    val fR = f._1
    val fI = f._2
    val gradShape = gR.fieldShape
    require(gR.fieldType == gI.fieldType)
    require(fR.fieldType == fI.fieldType)
    require(gR.fieldShape == gR.fieldShape)
    require(gR.tensorShape(0)%batchSize == 0)
    val filterNum = gR.tensorShape(0) / batchSize
    require(fR.tensorShape(0)%filterNum == 0)
    val inputLen = fR.tensorShape(0)/filterNum

    // This operator is described with variants based the following choices for (batchSetSize, inputSetSize)
    val batchSetSizes = Seq(1, 2, 3, 4, 5, 6, 8, 12, 16)
    val inputSetSizes = Seq(1, 2, 3, 4, 5, 6, 8, 12, 16)

    // NVIDIA PASCAL bricks (and earlier) do horribly above this miniTile size
    // threshold, so we exclude these to speed up the tuning process.
    val maxMiniTileElements = 128

    def clipSetSizes(batchSetSize: Int, inputSetSize: Int) = {
      val clippedBatchSetSize = math.min(batchSize, batchSetSize)
      val clippedInputSetSize = math.min(inputLen, inputSetSize)
      (clippedBatchSetSize, clippedInputSetSize)
    }

    val parameters =
      if (Convolution.useProfiler) {
        for (batchSetSize <- batchSetSizes;
             inputSetSize <- inputSetSizes
             if (batchSetSize * inputSetSize <= maxMiniTileElements)
        )
          yield clipSetSizes(batchSetSize, inputSetSize)
      }
      else {
        IndexedSeq(clipSetSizes(batchSetSizeDesired, inputSetSizeDesired))
      }

    val variantNames = parameters.toArray.map(p => s"fourierBackProjectMAC_${p._1}_${p._2}")

    val outputType = new FieldType(gradShape, Shape(inputLen*batchSize), Float32)
    val (outR, outI) = GPUOperator(outputType, outputType, variantNames){ i =>

      val (batchSetSize, inputSetSize) = parameters(i)

      // If (batchSize,inputLen) is not a multiple of (batchSetSize,inputSetSize), then round up.
      val batchSetNum = (batchSize + batchSetSize - 1)/batchSetSize
      val inputSetNum = (inputLen + inputSetSize - 1)/inputSetSize

      // Set up initial workfield parameters before a possible "transpose" of the assignment of threads
      // in the workfield to elements of the input fields
      val Rows = gradShape(0)
      val Columns = gradShape(1)
      val TensorElements = batchSetNum * inputSetNum

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

      val batchSetIndex = tensorElement / inputSetNum
      val inputSetIndex = tensorElement % inputSetNum

      // Rather than put in tests for boundary conditions to prevent a thread from reading or writing out-of-bounds
      // field data if (batchSize,inputLen) is not a multiple of (batchSetSize,inputSetSize), we
      // shift the base batch and input indices of the last set in each dimension to overlap the work of the
      // prior set.  This means that multiple workgroups might be writing the same output locations, but with
      // the same values, which is OK.

      val baseBatchIndex = intVar(batchSetIndex * batchSetSize)
      if (batchSize % batchSetSize != 0) {
        _if (batchSetIndex >= batchSetNum - 1) {
          baseBatchIndex := batchSize - batchSetSize
        }
      }

      val baseInputIndex = intVar(inputSetIndex * inputSetSize)
      if (inputLen % inputSetSize != 0) {
        _if (inputSetIndex >= inputSetNum - 1) {
          baseInputIndex := inputLen - inputSetSize
        }
      }

      //allocate local memory and zero output accumulator
      //total memory required = 2*bSS + 2*fSS + 2*bSS*fSS
      val gRlocal = _floatArray(batchSetSize)
      val gIlocal = _floatArray(batchSetSize)
      val fRlocal = _floatArray(inputSetSize)
      val fIlocal = _floatArray(inputSetSize)
      val outRlocal = _floatArray(batchSetSize*inputSetSize)
      val outIlocal = _floatArray(batchSetSize*inputSetSize)
      val outputIndex = _intVar()
      _for(outputIndex := 0, outputIndex < batchSetSize*inputSetSize, outputIndex+=1 ){
        outRlocal(outputIndex) := 0
        outIlocal(outputIndex) := 0
      }

      val filterIndex = _intVar()
      val gIndex = _intVar()
      val fIndex = _intVar()
      _for(filterIndex :=0, filterIndex < filterNum, filterIndex +=1){
        //load grad block into local memory
        _for(gIndex :=0, gIndex < batchSetSize, gIndex +=1){
          val readIndex = baseBatchIndex*filterNum + gIndex*filterNum + filterIndex
          gRlocal(gIndex) := _readTensorElement(gR, row, column, readIndex)
          gIlocal(gIndex) := _readTensorElement(gI, row, column, readIndex)
        }

        //load filter block into local memory
        _for(fIndex :=0, fIndex < inputSetSize, fIndex +=1){
          val readIndex = baseInputIndex + filterIndex*inputLen+fIndex
          fRlocal(fIndex) := _readTensorElement(fR, row, column, readIndex)
          fIlocal(fIndex) := _readTensorElement(fI, row, column, readIndex)
        }

        //accumulate outer product
        _for(gIndex :=0, gIndex < batchSetSize, gIndex +=1){
          _for(fIndex :=0, fIndex < inputSetSize, fIndex +=1){
            // This approach to accumulation uses 2 fma's, the use of '+=' uses only 1; results in a 4% perf gain.
            val idx = gIndex*inputSetSize+fIndex
            outRlocal(idx) := outRlocal(idx) + gRlocal(gIndex)*fRlocal(fIndex) - gIlocal(gIndex)*fIlocal(fIndex)
            outIlocal(idx) := outIlocal(idx) + gIlocal(gIndex)*fRlocal(fIndex) + gRlocal(gIndex)*fIlocal(fIndex)
          }
        }
      }

      //write the accumulated output
      val batchSetOffset = baseBatchIndex*inputLen
      val inputSetOffset = baseInputIndex
      _for(outputIndex := 0, outputIndex < batchSetSize*inputSetSize, outputIndex+=1 ){
        val inputBlockOffset = (outputIndex/inputSetSize)*inputLen
        val curOutput = batchSetOffset+inputSetOffset + inputBlockOffset + outputIndex%inputSetSize
        _writeTensorElement(_out0, outRlocal(outputIndex), row, column, curOutput)
        _writeTensorElement(_out1, outIlocal(outputIndex), row, column, curOutput)
      }
    }
    (outR, outI)
  }
}
