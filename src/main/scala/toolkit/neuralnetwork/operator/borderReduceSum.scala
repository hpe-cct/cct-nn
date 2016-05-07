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

/**
 * @author Matthew Pickett
 */
private [neuralnetwork] object borderReduceSum {
  def apply(x:Field, borderSize:Int) = {
    require(x.fieldShape.dimensions == 2)
    val inRows = x.fieldShape(0)
    val inCols = x.fieldShape(1)
    require(inRows > 2*borderSize)
    require(inCols > 2*borderSize)


    val rows = inRows - 2*borderSize
    val cols = inCols - 2*borderSize
    val outputType = new FieldType(Shape(rows,cols), x.tensorShape, x.fieldType.elementType)
    GPUOperator(outputType, "BorderReduceSum"){
      _globalThreads(outputType.fieldShape, outputType.tensorShape)
      val accum = _tensorElementVar(x)
      accum := _readTensorElement(x, _row+borderSize, _column+borderSize, _tensorElement)
      val i = _intVar()
      val j = _intVar()

      def accumCorner(r0:Int, r1:Int, c0:Int, c1:Int) =
        _for(i := r0, i < r1, i +=1){
          _for(j := c0, j < c1, j +=1){
            accum += _readTensorElement(x, i, j,_tensorElement)
          }
        }

      //top left corner
      _if(_row === 0 && _column === 0){
        accumCorner(0, borderSize, 0, borderSize)
      }

      //top right corner
      _if(_row === 0 && _column === cols-1){
        accumCorner(0,borderSize, inCols-borderSize, inCols)
      }

      //bottom left corner
      _if(_row === rows-1 && _column === 0){
        accumCorner(inRows-borderSize,inRows, 0, borderSize)
      }

      //bottom right corner
      _if(_row === rows-1 && _column === cols-1){
        accumCorner(inRows-borderSize,inRows, inCols-borderSize, inCols)
      }

      //top side
      _if(_row === 0){
        _for(i := 0, i < borderSize, i +=1){
          accum += _readTensorElement(x, i, _column+borderSize ,_tensorElement)
        }
      }

      //left side
      _if(_column === 0){
        _for(j := 0, j < borderSize, j +=1){
          accum += _readTensorElement(x, _row+borderSize, j ,_tensorElement)
        }
      }

      //right side
      _if(_column === cols-1){
        _for(j := inCols-borderSize, j < inCols, j +=1){
          accum += _readTensorElement(x, _row+borderSize, j ,_tensorElement)
        }
      }

      //bottom side
      _if(_row === rows-1){
        _for(i := inRows-borderSize, i < inRows, i +=1){
          accum += _readTensorElement(x, i, _column+borderSize ,_tensorElement)
        }
      }

      _writeTensorElement(_out0, accum, _tensorElement)

    }
  }
}
