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

package toolkit.neuralnetwork.source

import toolkit.neuralnetwork.source.SaveParameters.Param

import scala.collection.mutable.ArrayBuffer

/**
  * Helper class to manage the parameters packed into the single Sensor save/restore parameter string.
  *
  * Parameters are saved as comma-separated key=value parameter definitions.  The advantage of
  * using this class is that values that are null strings or contain spaces are OK.  Values should
  * not contain the "=" or "," character however.
  * @author Dick Carter
  */
class SaveParameters private[SaveParameters](params: ArrayBuffer[Param]) {
  /** Convert parameters to a single string, as in "key1=value1,key2=value2" */
  override def toString() = params.toArray.map(param => param.key + "=" + param.value).mkString(",")
  /** Add a String-valued parameter to the parameter list. */
  def addParam(key: String, value: String) { params += Param(key, value) }
  /** Add an Int-valued parameter to the parameter list. */
  def addIntParam(key: String, value: Int) { addParam(key, value.toString) }
  /** Add a Long-valued parameter to the parameter list. */
  def addLongParam(key: String, value: Long) { addParam(key, value.toString) }
  /** Add a Float-valued parameter to the parameter list. */
  def addFloatParam(key: String, value: Float) { addParam(key, value.toString) }
  /** Add a Boolean-valued parameter to the parameter list. */
  def addBooleanParam(key: String, value: Boolean) { addParam(key, value.toString) }
  /** Get a string-valued parameter from the parameter list. */
  def getParam(key: String) = {
    var answer: Option[String] = None
    for (param <- params)
      if (key == param.key)
        answer = Some(param.value)
    answer match {
      case Some(s) => s
      case None => throw new RuntimeException("parameter list missing key " + key)
    }
  }
  /** Get an Int-valued parameter from the parameter list. */
  def getIntParam(key: String) = getParam(key).toInt
  /** Get a Long-valued parameter from the parameter list. */
  def getLongParam(key: String) = getParam(key).toLong
  /** Get a Float-valued parameter from the parameter list. */
  def getFloatParam(key: String) = getParam(key).toFloat
  /** Get a Boolean-valued parameter from the parameter list. */
  def getBooleanParam(key: String) = if (getParam(key) == "true") true else false
}

object SaveParameters {
  /** Simple container class for the key-value pair. */
  case class Param(key: String, value: String)
  /** Helper class to convert from the packed parameter string format to an ArrayBuffer[Param]. */
  private def fromString(initString: String) = {
    val paramStrings = initString.split(",")
    val params = new ArrayBuffer[Param]()
    for (param <- paramStrings) {
      val keyValue = param.split("=")
      keyValue.length match {
        case 1 => params += Param(keyValue(0),"")
        case 2 => params += Param(keyValue(0), keyValue(1))
        case x => throw new RuntimeException("Could not parse parameter " + keyValue)
      }
    }
    params
  }
  /** Construct a SaveParameter based on a packed parameter string. */
  def apply(initString: String) = new SaveParameters(fromString(initString))
  /** Construct a SaveParameter with no initial parameters. */
  def apply() = new SaveParameters(new ArrayBuffer[Param]())
}
