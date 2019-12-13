# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test the annotate binary."""
from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow import annotate
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import test

FLAGS = test.FLAGS


###############################################################################
# Fixtures.
###############################################################################


@test.Fixture(
  scope="session", params=list(random_programl_generator.EnumerateTestSet()),
)
def real_proto(request) -> programl_pb2.ProgramGraph:
  """A test fixture which enumerates one of 100 "real" protos."""
  return request.param


@test.Fixture(scope="session")
def one_proto() -> programl_pb2.ProgramGraph:
  """A test fixture which enumerates a single real proto."""
  return next(random_programl_generator.EnumerateTestSet())


@test.Fixture(scope="session", params=list(programl.InputOutputFormat))
def stdin_fmt(request) -> programl.InputOutputFormat:
  """A test fixture which enumerates stdin formats."""
  return request.param


@test.Fixture(scope="session", params=list(programl.InputOutputFormat))
def stdout_fmt(request) -> programl.InputOutputFormat:
  """A test fixture which enumerates stdout formats."""
  return request.param


@test.Fixture(scope="session", params=list(annotate.AVAILABLE_ANALYSES))
def analysis(request) -> programl.InputOutputFormat:
  """A test fixture which yields all analysis names."""
  return request.param


@test.Fixture(scope="session", params=(1, 3))
def n(request) -> int:
  """A test fixture enumerate values for `n`."""
  return request.param


###############################################################################
# Tests.
###############################################################################


def test_invalid_analysis(one_proto: programl_pb2.ProgramGraph, n: int):
  """Test that error is raised if the input is invalid."""
  with test.Raises(ValueError) as e_ctx:
    annotate.Annotate("invalid_analysis", one_proto, n)
  assert str(e_ctx.value).startswith("Unknown analysis: invalid_analysis. ")
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/annotate_test.py
=======


@test.XFail(reason="Empty-graph check appears broken.")
def test_invalid_input(analysis: str, n: int):
  """Test that error is raised if the input is invalid."""
  invalid_input = programl_pb2.ProgramGraph()
  with test.Raises(IOError) as e_ctx:
    annotate.Annotate(analysis, invalid_input, n)
  assert str(e_ctx.value) == "Failed to serialize input graph"
>>>>>>> ec6e48210... Raise more clear exceptions.:deeplearning/ml4pl/graphs/labelled/dataflow/annotate_binary_test.py


def test_timeout(one_proto: programl_pb2.ProgramGraph):
  """Test that error is raised if the analysis times out."""
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/annotate_test.py
  with test.Raises(data_flow_graphs.AnalysisTimeout):
=======
  with test.Raises(annotate.AnalysisTimeout):
>>>>>>> ec6e48210... Raise more clear exceptions.:deeplearning/ml4pl/graphs/labelled/dataflow/annotate_binary_test.py
    annotate.Annotate("test_timeout", one_proto, timeout=1)


def test_binary_graph_input(one_proto: programl_pb2.ProgramGraph):
  """Test that a binary-encoded graph is acceptable."""
  binary_graph = programl.ToBytes(one_proto, programl.InputOutputFormat.PB)
  assert annotate.Annotate("reachability", binary_graph, n=3, binary_graph=True)


def test_annotate(analysis: str, real_proto: programl_pb2.ProgramGraph, n: int):
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/annotate_test.py
  """Test all annotators over all real protos."""
=======
  """Test annotating real program graphs."""
>>>>>>> ec6e48210... Raise more clear exceptions.:deeplearning/ml4pl/graphs/labelled/dataflow/annotate_binary_test.py
  try:
    # Use a lower timeout for testing.
    annotated = annotate.Annotate(analysis, real_proto, n, timeout=30)

    # Check that up to 'n' annotated graphs were generated.
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/annotate_test.py
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/annotate_test.py
    assert 0 <= len(annotated.protos) <= n

    # Check that output graphs have the same shape as the input graphs.
    for graph in annotated.protos:
      assert len(graph.node) == len(real_proto.node)
      assert len(graph.edge) == len(real_proto.edge)
  except data_flow_graphs.AnalysisTimeout:
=======
    assert 0 <= len(annotated.graph) <= n
=======
    assert 0 <= len(annotated.protos) <= n
>>>>>>> aff9d1d05... Fix annotator tests.:deeplearning/ml4pl/graphs/labelled/dataflow/annotate_binary_test.py

    # Check that output graphs have the same shape as the input graphs.
    for graph in annotated.protos:
      assert len(graph.node) == len(real_proto.node)
      assert len(graph.edge) == len(real_proto.edge)
  except annotate.AnalysisTimeout:
>>>>>>> ec6e48210... Raise more clear exceptions.:deeplearning/ml4pl/graphs/labelled/dataflow/annotate_binary_test.py
    # A timeout error is acceptable.
    pass


if __name__ == "__main__":
  test.Main()
