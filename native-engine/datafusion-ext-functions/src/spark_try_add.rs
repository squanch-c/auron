// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use arrow::{
    array::{Array, Int8Array, Int16Array, Int32Array, Int64Array},
    datatypes::DataType,
};
use datafusion::{
    common::{Result, ScalarValue, exec_err},
    logical_expr::ColumnarValue,
};
use datafusion_ext_commons::downcast_any;

pub fn spark_try_add(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    use ColumnarValue::{Array as Column, Scalar};

    let [left, right] = args else {
        return exec_err!("Spark_TryAdd requires exactly two arguments");
    };

    macro_rules! try_add {
        ($int:ident, $array:ty) => {{
            use ScalarValue::$int;

            match (left, right) {
                (Scalar($int(left)), Scalar($int(right))) => {
                    let sum = left.zip(*right).and_then(|(a, b)| a.checked_add(b));
                    Ok(Scalar($int(sum)))
                }
                (Column(array), Scalar($int(value))) | (Scalar($int(value)), Column(array)) => {
                    let array = downcast_any!(array, $array)?;
                    let mut output = <$array>::builder(array.len());
                    match *value {
                        Some(constant) => {
                            for value in array.iter() {
                                output.append_option(value.and_then(|v| constant.checked_add(v)));
                            }
                        }
                        None => output.append_nulls(array.len()),
                    }
                    Ok(Column(Arc::new(output.finish())))
                }
                (Column(left), Column(right)) => {
                    let left = downcast_any!(left, $array)?;
                    let right = downcast_any!(right, $array)?;
                    if left.len() != right.len() {
                        return exec_err!("Spark_TryAdd arrays must have equal lengths");
                    }
                    let mut output = <$array>::builder(left.len());
                    for (a, b) in left.iter().zip(right.iter()) {
                        output.append_option(a.zip(b).and_then(|(a, b)| a.checked_add(b)));
                    }
                    Ok(Column(Arc::new(output.finish())))
                }
                _ => exec_err!("Spark_TryAdd arguments must have the same integer type"),
            }
        }};
    }

    match left.data_type() {
        DataType::Int8 => try_add!(Int8, Int8Array),
        DataType::Int16 => try_add!(Int16, Int16Array),
        DataType::Int32 => try_add!(Int32, Int32Array),
        DataType::Int64 => try_add!(Int64, Int64Array),
        other => exec_err!("Spark_TryAdd does not support {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_add_columns() -> Result<()> {
        let left = Int32Array::from(vec![Some(i32::MAX), Some(i32::MIN), None, Some(3)]);
        let right = Int32Array::from(vec![1, -1, 2, 4]);
        let result = spark_try_add(&[
            ColumnarValue::Array(Arc::new(left)),
            ColumnarValue::Array(Arc::new(right)),
        ])?;
        let ColumnarValue::Array(result) = result else {
            return exec_err!("expected an array from two columns");
        };
        let expected = Int32Array::from(vec![None, None, None, Some(7)]);
        assert_eq!(downcast_any!(result, Int32Array)?, &expected);
        Ok(())
    }
}
