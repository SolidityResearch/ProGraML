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
"""Module for labelling program graphs with polyhedral SCoPs."""
import typing

import networkx as nx
import numpy as np
import pydot

from compilers.llvm import opt
from compilers.llvm import opt_util
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra/polyhedra.py
from deeplearning.ml4pl.graphs.unlabelled.llvm2graph import graph_builder
from deeplearning.ml4pl.graphs.unlabelled.llvm2graph.cfg import llvm_util
from labm8.py import app
from labm8.py import decorators

=======
from deeplearning.ml4pl.graphs.unlabelled.cdfg import control_and_data_flow_graph as cdfg
from deeplearning.ml4pl.graphs.unlabelled.cfg import llvm_util
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra/polyhedra.py
>>>>>>> edb8c21d9... Automated code format.:deeplearning/ml4pl/graphs/labelled/polyhedra/polyhedra.py
=======
from labm8.py import app
from labm8.py import decorators
>>>>>>> 8be094257... Move //labm8 to //labm8/py.:deeplearning/ml4pl/graphs/labelled/polyhedra/polyhedra.py

FLAGS = app.FLAGS


<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra/polyhedra.py
def RecursePydot(
  subgraph: pydot.Dot,
  func: typing.Callable[[pydot.Dot, typing.Any], None],
  state: typing.Any,
):
=======
def RecursePydot(subgraph: pydot.Dot,
                 func: typing.Callable[[pydot.Dot, typing.Any], None],
                 state: typing.Any):
>>>>>>> edb8c21d9... Automated code format.:deeplearning/ml4pl/graphs/labelled/polyhedra/polyhedra.py
  func(subgraph, state)
  for ss in subgraph.get_subgraphs():
    RecursePydot(ss, func, state)


def SubNodes(subgraph: pydot.Dot, nodes: typing.List[typing.Any]):
  nodes.extend(subgraph.get_nodes())


def GetSubgraph(subgraph: pydot.Dot, state: typing.Dict[pydot.Dot, typing.Any]):
  if subgraph.get("style") == "filled":
    nodes = []
    RecursePydot(subgraph, SubNodes, nodes)
    state[subgraph] = nodes


class PolyhedralRegionAnnotator(llvm_util.TagHook):
  """Tag hook that annotates polyhedral regions on the nodes (with the attribute
  `polyhedral=True`)"""

  def OnGraphBegin(self, dot: pydot.Dot):
    # Get polyhedral basic blocks from Polly and pydot
    # Obtain all basic blocks in polyhedral region (need to recurse into sub-subgraphs)
    self.regions = {}
    RecursePydot(dot, GetSubgraph, self.regions)

  def OnNode(self, node: pydot.Node) -> typing.Dict[str, typing.Any]:
    for region in self.regions.values():
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra/polyhedra.py
      if node.get_name() in [
        str(r)[:-1] for r in region
      ]:  # Need to cut off semicolon
        return {"polyhedral": True}

    return {"polyhedral": False}

  def OnInstruction(
    self, node_attrs: typing.Dict[str, typing.Any], instruction: str
  ) -> typing.Dict[str, typing.Any]:
    return {"polyhedral": node_attrs.get("polyhedral", False)}

  def OnIdentifier(
    self,
    stmt_node: typing.Dict[str, typing.Any],
    identifier_node: typing.Dict[str, typing.Any],
    definition_type: str,
  ) -> typing.Dict[str, typing.Any]:
    if definition_type == "def":
      if "polyhedral" in stmt_node:
        return {"polyhedral": stmt_node["polyhedral"]}
=======
      if node.get_name() in [str(r)[:-1] for r in region
                            ]:  # Need to cut off semicolon
        return {'polyhedral': True}

    return {'polyhedral': False}

  def OnInstruction(self, node_attrs: typing.Dict[str, typing.Any],
                    instruction: str) -> typing.Dict[str, typing.Any]:
    return {'polyhedral': node_attrs.get('polyhedral', False)}

  def OnIdentifier(self, stmt_node: typing.Dict[str, typing.Any],
                   identifier_node: typing.Dict[str, typing.Any],
                   definition_type: str) -> typing.Dict[str, typing.Any]:
    if definition_type == 'def':
      if 'polyhedral' in stmt_node:
        return {'polyhedral': stmt_node['polyhedral']}
>>>>>>> edb8c21d9... Automated code format.:deeplearning/ml4pl/graphs/labelled/polyhedra/polyhedra.py

    # TODO(talbn): Perhaps no need for definition_type == 'use' (may come from outside region)

    return {}


@decorators.timeout(seconds=60)
def BytecodeToPollyCanonicalized(source: str) -> str:
  process = opt.Exec(
    ["-polly-canonicalize", "-S", "-", "-o", "-"], stdin=source
  )
  if process.returncode:
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra/polyhedra.py
    raise opt.OptException(
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra/polyhedra.py
      "Error in canonicalization opt execution (%d)" % process.returncode
    )
=======
        'Error in canonicalization opt execution (%d)' % process.returncode)
>>>>>>> edb8c21d9... Automated code format.:deeplearning/ml4pl/graphs/labelled/polyhedra/polyhedra.py
=======
    raise opt.OptException('Error in canonicalization opt execution (%d)' %
                           process.returncode[:120])
>>>>>>> 085ca75b4... Export fix ups.:deeplearning/ml4pl/graphs/labelled/polyhedra/polyhedra.py
  return process.stdout


@decorators.timeout(seconds=60)
def CreateCDFG(bytecode: str) -> nx.MultiDiGraph:
  builder = graph_builder.ProGraMLGraphBuilder()
  return builder.Build(bytecode)


@decorators.timeout(seconds=60)
def AnnotatePolyhedra(
  g: nx.MultiDiGraph,
  annotated_cdfgs: typing.List[nx.MultiDiGraph],
  x_label: str = "x",
  y_label: str = "y",
  false=False,
  true=True,
) -> None:
  """

  Args:
    g: The graph.
    annotated_cdfgs: CDFGs with nodes that polly marked as "polyhedral" (green in input dot).
    x_label: The graph 'x' attribute property attribute name.
    y_label: The graph 'y' attribute property attribute name.
    false: The value to set for nodes that are not polyhedral.
    true: The value to set for nodes that are polyhedral.
  """

  # Set all of the nodes as not-polyhedral at first.
  # X labels are a list which concatenates the original graph 'x'
  # embedding indices with a [0,1] value for false/true, respectively.
  for _, data in g.nodes(data=True):
    data[x_label] = [data[x_label], 0]
    data[y_label] = false

  # Obtain nodes in g
  entities = 0
  mismatched_entities = 0
  for cdfg in annotated_cdfgs:
    # Mark the nodes in the polyhedral regions
    for node, ndata in cdfg.nodes(data=True):
      if not ndata.get("polyhedral"):
        continue

      entities += 1
      if node not in g.nodes:
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra/polyhedra.py
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra/polyhedra.py
        mismatched_entities += 1
        continue
      g.nodes[node][y_label] = true

  if mismatched_entities > 0:
    app.Warning(
      "%d (%f%%) mismatched entities in code",
      mismatched_entities,
      mismatched_entities / entities * 100,
    )


@decorators.timeout(seconds=120)
=======
        raise ValueError(
            f"Entity `{node}` not found in graph, {g.nodes(data=True)}")
=======
        raise ValueError(f"Entity `{node}` not found in graph")
>>>>>>> 085ca75b4... Export fix ups.:deeplearning/ml4pl/graphs/labelled/polyhedra/polyhedra.py
      g.nodes[node][y_label] = true


>>>>>>> edb8c21d9... Automated code format.:deeplearning/ml4pl/graphs/labelled/polyhedra/polyhedra.py
def MakePolyhedralGraphs(
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra/polyhedra.py
  bytecode: str, n: typing.Optional[int] = None, false=False, true=True,
=======
    bytecode: str,
    n: typing.Optional[int] = None,
    false=False,
    true=True,
>>>>>>> 7a1801574... Add support for generating polyhedra datasets.:deeplearning/ml4pl/graphs/labelled/polyhedra/polyhedra.py
) -> typing.Iterable[nx.MultiDiGraph]:
  """Create an annotated graph from a bytecode that potentially contains
     polyhedral loops.

  Args:
    bytecode: The bytecode which produced the input graph.
    n: The maximum number of graphs to produce. This value is ignored and one graph
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra/polyhedra.py
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra/polyhedra.py
=======
>>>>>>> d9a9c98fc... Add cross-references to issue tracker.:deeplearning/ml4pl/graphs/labelled/polyhedra/polyhedra.py
      will be produced with all polyhedral regions annotated.
    false: TODO(github.com/ChrisCummins/ProGraML/issues/2): Unused. This method
      is hardcoded to use 2-class 1-hots.
    true: TODO(github.com/ChrisCummins/ProGraML/issues/2): Unused. This method
      is hardcoded to use 2-class 1-hots.
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra/polyhedra.py
=======
       will be produced with all polyhedral regions annotated.
    false: TODO(cec): Unused. This method is hardcoded to use 2-class 1-hots.
    true: TODO(cec): Unused. This method is hardcoded to use 2-class 1-hots.
>>>>>>> 085ca75b4... Export fix ups.:deeplearning/ml4pl/graphs/labelled/polyhedra/polyhedra.py
=======
>>>>>>> d9a9c98fc... Add cross-references to issue tracker.:deeplearning/ml4pl/graphs/labelled/polyhedra/polyhedra.py

  Returns:
    A generator of annotated graphs, where each graph has 'x' and 'y' labels on
    the statement nodes, and additionally a 'data_flow_max_steps_required'
    attribute which is set to the largest number of statements in a polyhedral block.
  """
  # TODO(github.com/ChrisCummins/ProGraML/issues/2): Replace true/false args
  # with a list of class values for all graph annotator functions.
  del false
  del true
  del n

  # One-hot encoding
  false = np.array([1, 0], np.int64)
  true = np.array([0, 1], np.int64)

  # Canonicalize input graph (see http://polly.llvm.org/docs/Architecture.html)
  bytecode = BytecodeToPollyCanonicalized(bytecode)
  g = CreateCDFG(bytecode)

  # Build the polyhedral building blocks
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra/polyhedra.py
  scop_graphs, _ = opt_util.DotGraphsFromBytecode(
    bytecode,
    [
      "-O1",
      "-polly-process-unprofitable",
      "-polly-optimized-scops",
      "-polly-dot",
      "-polly-optimizer=none",
    ],
  )
=======
  scop_graphs, _ = opt_util.DotGraphsFromBytecode(bytecode, [
      '-O1', '-polly-process-unprofitable', '-polly-optimized-scops',
      '-polly-dot', '-polly-optimizer=none'
  ])
>>>>>>> edb8c21d9... Automated code format.:deeplearning/ml4pl/graphs/labelled/polyhedra/polyhedra.py

  # Loop over each function
  max_steps = 0
  cdfgs = []
  for i, graph in enumerate(scop_graphs):
    graph_annotator = PolyhedralRegionAnnotator()
    dot = graph
    cfg = llvm_util.ControlFlowGraphFromDotSource(dot, tag_hook=graph_annotator)
    builder = graph_builder.ProGraMLGraphBuilder()
    annotated_cdfg = builder.BuildFromControlFlowGraph(cfg)

<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra/polyhedra.py
    steps = sum(
      1
      for nid, node in annotated_cdfg.nodes(data=True)
      if node.get("polyhedral")
    )
=======
    steps = sum(1 for nid, node in annotated_cdfg.nodes(data=True)
                if node.get('polyhedral'))
>>>>>>> edb8c21d9... Automated code format.:deeplearning/ml4pl/graphs/labelled/polyhedra/polyhedra.py
    max_steps = max(max_steps, steps)
    cdfgs.append(annotated_cdfg)

  labelled = g.copy()
  labelled.data_flow_max_steps_required = max_steps
  AnnotatePolyhedra(labelled, cdfgs, false=false, true=true)
  yield labelled
