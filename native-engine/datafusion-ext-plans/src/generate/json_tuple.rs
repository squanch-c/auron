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

use std::{any::Any, sync::Arc};

use arrow::{
    array::{Array, ArrayRef},
    record_batch::RecordBatch,
};
use datafusion::{common::Result, physical_expr::PhysicalExprRef};
use datafusion_ext_commons::downcast_any;
use datafusion_ext_functions::spark_get_json_object::{
    spark_get_parsed_json_simple_field, spark_parse_json,
};

use crate::generate::{GenerateState, GeneratedRows, Generator};

#[derive(Debug)]
pub struct JsonTuple {
    child: PhysicalExprRef,
    json_paths: Vec<String>,
}

impl JsonTuple {
    pub fn new(child: PhysicalExprRef, json_paths: Vec<String>) -> Self {
        Self { child, json_paths }
    }
}

impl Generator for JsonTuple {
    fn exprs(&self) -> Vec<PhysicalExprRef> {
        vec![self.child.clone()]
    }

    fn with_new_exprs(&self, exprs: Vec<PhysicalExprRef>) -> Result<Arc<dyn Generator>> {
        Ok(Arc::new(Self::new(
            exprs[0].clone(),
            self.json_paths.clone(),
        )))
    }

    fn eval_start(&self, batch: &RecordBatch) -> Result<Box<dyn GenerateState>> {
        let num_rows = batch.num_rows();
        let json_values = spark_parse_json(&[self.child.evaluate(batch)?], num_rows)?;
        let parsed_json_array = json_values.into_array(num_rows)?;
        Ok(Box::new(JsonTupleGenerateState {
            parsed_json_array,
            cur_row_id: 0,
        }))
    }

    fn eval_loop(&self, state: &mut Box<dyn GenerateState>) -> Result<Option<GeneratedRows>> {
        let state = downcast_any!(state, mut JsonTupleGenerateState)?;
        if state.cur_row_id >= state.parsed_json_array.len() {
            return Ok(None);
        }

        let evaluated: Vec<ArrayRef> = self
            .json_paths
            .iter()
            .map(|json_path| {
                spark_get_parsed_json_simple_field(&state.parsed_json_array, json_path)
            })
            .collect::<Result<_>>()?;

        let generated = GeneratedRows {
            row_ids: (state.cur_row_id..state.parsed_json_array.len())
                .map(|i| i as i32)
                .collect(),
            cols: evaluated,
        };
        state.cur_row_id = state.parsed_json_array.len();
        Ok(Some(generated))
    }
}

struct JsonTupleGenerateState {
    parsed_json_array: ArrayRef,
    cur_row_id: usize,
}

impl GenerateState for JsonTupleGenerateState {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn cur_row_id(&self) -> usize {
        self.cur_row_id
    }
}

#[cfg(test)]
mod json_tuple_scalar_regression {
    use std::sync::Arc;

    use arrow::{
        array::{ArrayRef, AsArray, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::{RecordBatch, RecordBatchOptions},
    };
    use datafusion::{
        common::{Result, ScalarValue},
        physical_expr::expressions::{Column, Literal},
    };

    use crate::generate::{Generator, json_tuple::JsonTuple};

    fn assert_output(
        generator: JsonTuple,
        batch: &RecordBatch,
        expected: &[Option<&str>],
    ) -> Result<()> {
        assert_eq!(batch.num_rows(), expected.len());
        let mut state = generator.eval_start(batch)?;
        let output = generator.eval_loop(&mut state)?;
        if expected.is_empty() {
            assert!(output.is_none());
        } else {
            let output = output.expect("nonempty input must generate rows");
            let row_ids: Vec<_> = output.row_ids.iter().collect();
            let expected_ids: Vec<_> = (0..expected.len()).map(|i| Some(i as i32)).collect();
            assert_eq!(row_ids, expected_ids);
            assert_eq!(output.cols.len(), 1);
            assert_eq!(
                output.cols[0].as_string::<i32>().iter().collect::<Vec<_>>(),
                expected
            );
        }
        assert_eq!(state.cur_row_id(), expected.len());
        assert!(generator.eval_loop(&mut state)?.is_none());
        Ok(())
    }

    #[test]
    fn scalar_inputs_preserve_rows_and_row_ids() -> Result<()> {
        for value in [None, Some(r#"{"name":"Alice"}"#)] {
            for rows in [0, 1, 2, 8] {
                let batch = RecordBatch::try_new_with_options(
                    Arc::new(Schema::empty()),
                    vec![],
                    &RecordBatchOptions::new().with_row_count(Some(rows)),
                )?;
                let generator = JsonTuple::new(
                    Arc::new(Literal::new(ScalarValue::Utf8(value.map(str::to_owned)))),
                    vec!["name".into()],
                );
                assert_output(generator, &batch, &vec![value.map(|_| "Alice"); rows])?;
            }
        }
        Ok(())
    }

    #[test]
    fn array_inputs_preserve_rows_and_row_ids() -> Result<()> {
        let values = [
            Some(r#"{"name":"Alice"}"#),
            None,
            Some("{"),
            Some(r#"{"other":1}"#),
            Some(r#"{"name":null}"#),
            Some(r#"{"name":"Bob"}"#),
        ];
        let expected = [Some("Alice"), None, None, None, None, Some("Bob")];
        for rows in [0, 1, 6] {
            let input: ArrayRef = Arc::new(StringArray::from(values[..rows].to_vec()));
            let schema = Arc::new(Schema::new(vec![Field::new("json", DataType::Utf8, true)]));
            let batch = RecordBatch::try_new(schema, vec![input])?;
            let generator = JsonTuple::new(Arc::new(Column::new("json", 0)), vec!["name".into()]);
            assert_output(generator, &batch, &expected[..rows])?;
        }
        Ok(())
    }
}
