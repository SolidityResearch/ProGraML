"""This module defines an LSTM for node-level classification."""
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional
<<<<<<< HEAD

=======
>>>>>>> 33703e4de... Split the LSTM into multiple modules.
import numpy as np
import tensorflow as tf

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
<<<<<<< HEAD
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models.lstm import lstm_base
from deeplearning.ml4pl.models.lstm import lstm_utils as utils
=======
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.models.lstm import lstm_utils as utils
from deeplearning.ml4pl.models.lstm import lstm_base
>>>>>>> 33703e4de... Split the LSTM into multiple modules.
from deeplearning.ml4pl.seq import graph2seq
from labm8.py import app
from labm8.py import progress


FLAGS = app.FLAGS

app.DEFINE_integer(
<<<<<<< HEAD
  "padded_node_sequence_length",
  5000,
  "For node-level models, the padded/truncated length of encoded node "
  "sequences.",
=======
    "padded_nodes_sequence_length",
    5000,
    "For node-level models, the padded/truncated length of encoded node "
    "sequences.",
>>>>>>> 33703e4de... Split the LSTM into multiple modules.
)


class NodeLstmBatch(NamedTuple):
  """The data for an LSTM batch."""

  # Shape (batch_size, padded_sequence_length, 1), dtype np.int32
  encoded_sequences: np.array
  # Shape (batch_size, padded_sequence_length, 1), dtype np.int32
  segment_ids: np.array
<<<<<<< HEAD
  # Shape (batch_size, padded_node_sequence_length, 2), dtype np.int32
  selector_vectors: np.array
  # Shape (batch_size, padded_node_sequence_length, node_y_dimensionality),
  # dtype np.float32
  node_y: np.array

  # An array of shape (batch_node_count, node_y_dimensionality) which
  # concatenates the true node labels for each of the graphs in the batch,
  # without padding or truncation.
  targets: np.array

  # An array of shape (batch_size * padded_node_sequence_length, 1) which
  # indicate the indices into the targets array that the model produced outputs
  # for. Nodes which were not used have a -1 padding value. All other values
  # are in the range [0, batch_node_count - 1].
  node_indices: np.array

  @property
  def x(self) -> List[np.array]:
    """Get the model 'x' inputs."""
    return [self.encoded_sequences, self.segment_ids, self.selector_vectors]

  @property
  def y(self) -> List[np.array]:
    """Get the model 'y' inputs."""
    return [self.node_y]

  def GetPredictions(
    self, model_output, ctx: progress.ProgressContext = progress.NullContext
  ) -> np.array:
    """Reshape the model outputs to an array of predictions of same shape as
    targets."""
    if model_output.shape != self.node_y.shape:
      raise ValueError(
        f"Model produced output with shape {model_output.shape}, but expected "
        f"outputs with shape {self.node_y.shape}"
      )
    predictions = np.zeros(self.targets.shape)
    flattened_model_output = np.vstack(model_output)
    if len(self.node_indices) != len(flattened_model_output):
      raise ValueError(
        f"Model produced output with shape {flattened_model_output.shape} but "
        f"expected outputs with shape {self.node_indices.shape}"
      )

    # Strip the padding from node indices.
    active_nodes = np.where(self.node_indices != -1)
    node_indices = self.node_indices[active_nodes]
    flattened_model_output = flattened_model_output[active_nodes]

    predictions[node_indices] = flattened_model_output
    ctx.Log(
      4,
      "Reshaped model outputs from %s to %s",
      model_output.shape,
      predictions.shape,
    )
    return predictions

=======
  # Shape (batch_size, padded_nodes_sequence_length, 2), dtype np.int32
  selector_vectors: np.array
  # Shape (batch_size, padded_nodes_sequence_length, node_y_dimensionality),
  # dtype np.float32
  node_y: np.array

  # A list of indices into the graph nodes for the nodes that are encoded.
  # Shape (batch_size, ?),
  node_indices: List[np.array]

  @property
  def x(self):
    return [self.encoded_sequences, self.segment_ids, self.selector_vectors]

  @property
  def y(self):
    return [self.node_y]

>>>>>>> 33703e4de... Split the LSTM into multiple modules.

class NodeLstm(lstm_base.LstmBase):
  """An LSTM model for node-level classification."""

  def __init__(
<<<<<<< HEAD
    self,
    *args,
    padded_node_sequence_length: Optional[int] = None,
    proto_db: Optional[unlabelled_graph_database.Database] = None,
    **kwargs,
=======
      self,
      *args,
      padded_nodes_sequence_length: Optional[int] = None,
      proto_db: Optional[unlabelled_graph_database.Database] = None,
      **kwargs,
>>>>>>> 33703e4de... Split the LSTM into multiple modules.
  ):
    if not proto_db and not FLAGS.proto_db:
      raise app.UsageError("--proto_db is required for node level models")
    self._proto_db = proto_db or FLAGS.proto_db()

    # Determine the maximum node sequence length.
<<<<<<< HEAD
    self._padded_node_sequence_length = padded_node_sequence_length

    super(NodeLstm, self).__init__(*args, **kwargs)

=======
    self._padded_nodes_sequence_length = padded_nodes_sequence_length

    super(NodeLstm, self).__init__(*args, **kwargs)


>>>>>>> 33703e4de... Split the LSTM into multiple modules.
  @property
  def padded_node_sequence_length(self) -> int:
    """Get the length of padded node sequences."""
    return min(
<<<<<<< HEAD
      self.padded_sequence_length,
      self._padded_node_sequence_length or FLAGS.padded_node_sequence_length,
=======
        self.padded_sequence_length,
        self._padded_nodes_sequence_length or FLAGS.padded_nodes_sequence_length,
>>>>>>> 33703e4de... Split the LSTM into multiple modules.
    )

  @property
  def warm_up_batch_size(self) -> int:
    """This model has fixed sized batches, so the warm-up iteration requires
    a full batch_size number of graphs."""
    return self.batch_size

  def CreateKerasModel(self) -> tf.compat.v1.keras.Model:
    """Construct the tensorflow computation graph."""
    sequence_input = tf.compat.v1.keras.layers.Input(
<<<<<<< HEAD
      batch_shape=(self.batch_size, self.padded_sequence_length,),
      dtype="int32",
      name="sequence_in",
    )
    segment_ids = tf.compat.v1.keras.layers.Input(
      batch_shape=(self.batch_size, self.padded_sequence_length,),
      dtype="int32",
      name="segment_ids",
    )
    selector_vector = tf.compat.v1.keras.layers.Input(
      batch_shape=(self.batch_size, None, 2),
      dtype="float32",
      name="selector_vector",
=======
        batch_shape=(self.batch_size, self.padded_sequence_length,),
        dtype="int32",
        name="sequence_in",
    )
    segment_ids = tf.compat.v1.keras.layers.Input(
        batch_shape=(self.batch_size, self.padded_sequence_length,),
        dtype="int32",
        name="segment_ids",
    )
    selector_vector = tf.compat.v1.keras.layers.Input(
        batch_shape=(self.batch_size, None, 2),
        dtype="float32",
        name="selector_vector",
>>>>>>> 33703e4de... Split the LSTM into multiple modules.
    )

    # Embed the sequence inputs and sum the embeddings by nodes.
    embedded_inputs = tf.compat.v1.keras.layers.Embedding(
<<<<<<< HEAD
      input_dim=self.padded_vocabulary_size,
      input_length=self.padded_sequence_length,
      output_dim=FLAGS.lang_model_hidden_size,
      name="embedding",
    )(sequence_input)

    segmented_input = utils.SegmentSumLayer(
      encoded_sequences=embedded_inputs,
      segment_ids=segment_ids,
      batch_size=self.batch_size,
      max_sequence_length=self.padded_sequence_length,
      max_output_sequence_length=self.graph_db.node_count_max,
    )
    # Because padded values in segment_ids have value, the segment sum emits a
    # tensor of shape [B, padded_node_sequence_length + 1, 64] IFF something
    # was padded and [B, padded_node_sequence_length, 64] otherwise. We want
    # to discard the + 1 guy because that is just summed padded tokens anyway.
    segmented_input = utils.SliceToSizeLayer(
      segmented_input=segmented_input, selector_vector=selector_vector
    )
    lang_model_input = tf.compat.v1.keras.layers.Concatenate(
      axis=2, name="segmented_inputs_and_selector_vectors"
=======
        input_dim=self.padded_vocabulary_size,
        input_length=self.padded_sequence_length,
        output_dim=FLAGS.lang_model_hidden_size,
        name="embedding",
    )(sequence_input)

    segmented_input = utils.SegmentSumLayer(
        encoded_sequences=embedded_inputs,
        segment_ids=segment_ids,
        batch_size=self.batch_size,
        max_sequence_length=self.padded_sequence_length,
        max_output_sequence_length=self.graph_db.node_count_max,
    )
    # Because padded values in segment_ids have value, the segment sum emits a
    # tensor of shape [B, padded_nodes_sequence_length + 1, 64] IFF something
    # was padded and [B, padded_nodes_sequence_length, 64] otherwise. We want
    # to discard the + 1 guy because that is just summed padded tokens anyway.
    segmented_input = utils.SliceToSizeLayer(
        segmented_input=segmented_input, selector_vector=selector_vector
    )
    lang_model_input = tf.compat.v1.keras.layers.Concatenate(
        axis=2, name="segmented_inputs_and_selector_vectors"
>>>>>>> 33703e4de... Split the LSTM into multiple modules.
    )([segmented_input, selector_vector],)

    # Make the language model.
    lang_model = utils.LstmLayer(
<<<<<<< HEAD
      FLAGS.lang_model_hidden_size, return_sequences=True, name="lstm_1"
    )(lang_model_input)
    lang_model = utils.LstmLayer(
      FLAGS.lang_model_hidden_size,
      return_sequences=True,
      return_state=False,
      name="lstm_2",
    )(lang_model)

    node_out = tf.compat.v1.keras.layers.Dense(
      self.graph_db.node_y_dimensionality,
      activation="sigmoid",
      name="node_out",
    )(lang_model)

    model = tf.compat.v1.keras.Model(
      inputs=[sequence_input, segment_ids, selector_vector], outputs=[node_out],
    )
    model.compile(
      optimizer="adam",
      metrics=["accuracy"],
      loss=["categorical_crossentropy"],
      loss_weights=[1.0],
=======
        FLAGS.lang_model_hidden_size, return_sequences=True, name="lstm_1"
    )(lang_model_input)
    lang_model = utils.LstmLayer(
        FLAGS.lang_model_hidden_size,
        return_sequences=True,
        return_state=False,
        name="lstm_2",
    )(lang_model)

    node_out = tf.compat.v1.keras.layers.Dense(
        self.graph_db.node_y_dimensionality,
        activation="sigmoid",
        name="node_out",
    )(lang_model)

    model = tf.compat.v1.keras.Model(
        inputs=[sequence_input, segment_ids, selector_vector], outputs=[node_out],
    )
    model.compile(
        optimizer="adam",
        metrics=["accuracy"],
        loss=["categorical_crossentropy"],
        loss_weights=[1.0],
>>>>>>> 33703e4de... Split the LSTM into multiple modules.
    )

    return model

  def GetEncoder(self) -> graph2seq.EncoderBase:
    """Construct the graph encoder."""
    if not (
<<<<<<< HEAD
      self.graph_db.node_y_dimensionality
      and self.graph_db.node_x_dimensionality == 2
      and self.graph_db.graph_y_dimensionality == 0
    ):
      raise app.UsageError(
        f"Unsupported graph dimensionalities: {self.graph_db}"
      )
    return graph2seq.StatementEncoder(
      graph_db=self.graph_db,
      proto_db=self._proto_db,
      max_encoded_length=self.padded_sequence_length,
      max_nodes=self.padded_node_sequence_length,
    )

  def MakeBatch(
    self,
    epoch_type: epoch.Type,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
=======
        self.graph_db.node_y_dimensionality
        and self.graph_db.node_x_dimensionality == 2
        and self.graph_db.graph_y_dimensionality == 0
    ):
      raise app.UsageError(f"Unsupported graph dimensionalities: {self.graph_db}")
    return graph2seq.StatementEncoder(
        graph_db=self.graph_db, proto_db=self._proto_db
    )

  def MakeBatch(
      self,
      epoch_type: epoch.Type,
      graphs: Iterable[graph_tuple_database.GraphTuple],
      ctx: progress.ProgressContext = progress.NullContext,
>>>>>>> 33703e4de... Split the LSTM into multiple modules.
  ) -> batches.Data:
    """Create a mini-batch of LSTM data."""
    del epoch_type  # Unused.

    graphs = self.GetBatchOfGraphs(graphs)
    # For node classification we require all batches to be of batch_size.
    # In the future we could work around this by padding an incomplete
    # batch with arrays of zeros.
    if not graphs or len(graphs) != self.batch_size:
<<<<<<< HEAD
      return batches.EndOfBatches()
=======
      return batches.Data(graph_ids=[], data=None)
>>>>>>> 33703e4de... Split the LSTM into multiple modules.

    # Encode the graphs in the batch.

    # A list of arrays of shape (node_count, 1)
    encoded_sequences: List[np.array] = []
    # A list of arrays of shape (node_count, 1)
    segment_ids: List[np.array] = []
    # A list of arrays of shape (node_mask_count, 2)
    selector_vectors: List[np.array] = []
    # A list of arrays of shape (node_mask_count, node_y_dimensionality)
    node_y: List[np.array] = []
<<<<<<< HEAD
    all_node_indices: List[np.array] = []
    targets: List[np.array] = []

    try:
      encoded_graphs = self.encoder.Encode(graphs, ctx=ctx)
    except ValueError as e:
      ctx.Error("%s", e)
      # TODO(github.com/ChrisCummins/ProGraML/issues/38): to debug a possible
      # error in the LSTM I have temporarily made the batch construction
      # resilient to encoder errors by returning an empty batch.
      # Graph encoding failed, so return an empty batch. This is probably
      # not a good idea to keep, as it means the LSTM will silently skip
      # data.
      return batches.Data(graph_ids=[], data=None)

    node_offset = 0
    # Convert ProgramGraphSeq protos to arrays of numeric values.
    for graph, seq in zip(graphs, encoded_graphs):
=======
    node_indices: List[np.array] = []

    # Convert ProgramGraphSeq protos to arrays of numeric values.
    for graph, seq in zip(graphs, self.encoder.Encode(graphs, ctx=ctx)):
>>>>>>> 33703e4de... Split the LSTM into multiple modules.
      # Skip empty graphs.
      if not seq.encoded:
        continue

      encoded_sequences.append(np.array(seq.encoded, dtype=np.int32))

      # Construct a list of segment IDs using the encoded node lengths,
      # e.g. for encoded node lengths [2, 3, 1], produce segment IDs:
      # [0, 0, 1, 1, 1, 2].
<<<<<<< HEAD
      out_of_range_segment = self.padded_node_sequence_length - 1
      segment_ids.append(
        np.concatenate(
          [
            np.ones(encoded_node_length, dtype=np.int32)
            * min(segment_id, out_of_range_segment)
            for segment_id, encoded_node_length in enumerate(
              seq.encoded_node_length
            )
          ]
        )
=======
      out_of_range_segment = self.padded_nodes_sequence_length - 1
      segment_ids.append(
          np.concatenate(
              [
                np.ones(encoded_node_length, dtype=np.int32)
                * min(segment_id, out_of_range_segment)
                for segment_id, encoded_node_length in enumerate(
                  seq.encoded_node_length
              )
              ]
          )
>>>>>>> 33703e4de... Split the LSTM into multiple modules.
      )

      # Get the list of graph node indices that produced the serialized encoded
      # graph representation. We use this to construct predictions for the
      # "full" graph through padding.
<<<<<<< HEAD
      node_indices = np.array(seq.node, dtype=np.int32)

      # Offset the node index list and concatenate.
      all_node_indices.append(node_indices + node_offset)

      # Sanity check that the node indices are in-range for this graph.
      assert len(graph.tuple.node_x) >= max(node_indices)

      # Use only the 'binary selector' feature and convert to an array of
      # 1 hot binary vectors.
      node_selectors = graph.tuple.node_x[:, 1][node_indices]
=======
      seq_node_indices = np.array(seq.node, dtype=np.int32)
      node_indices.append(seq_node_indices)

      # Sanity check that the node indices are in-range for this graph.
      assert len(graph.tuple.node_x) >= max(seq_node_indices)

      # Use only the 'binary selector' feature and convert to an array of
      # 1 hot binary vectors.
      node_selectors = graph.tuple.node_x[:, 1][seq_node_indices]
>>>>>>> 33703e4de... Split the LSTM into multiple modules.
      node_selector_vectors = np.zeros((node_selectors.size, 2), dtype=np.int32)
      node_selector_vectors[np.arange(node_selectors.size), node_selectors] = 1
      selector_vectors.append(node_selector_vectors)

      # Select the node targets for only the active nodes.
<<<<<<< HEAD
      node_y.append(graph.tuple.node_y[node_indices])
      targets.append(graph.tuple.node_y)

      # Increment out node offset for concatenating the list of node indices.
      node_offset += graph.node_count

    # Pad and truncate encoded sequences.
    encoded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
      encoded_sequences,
      maxlen=self.padded_sequence_length,
      dtype="int32",
      padding="pre",
      truncating="post",
      value=self.padding_element,
=======
      node_y.append(graph.tuple.node_y[seq_node_indices])

    # Pad and truncate encoded sequences.
    encoded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_sequences,
        maxlen=self.padded_sequence_length,
        dtype="int32",
        padding="pre",
        truncating="post",
        value=self.padding_element,
>>>>>>> 33703e4de... Split the LSTM into multiple modules.
    )

    # Determine an out-of-range segment ID to pad the segment IDs to.
    segment_id_padding_element = (
<<<<<<< HEAD
      max(max(s) if s.size else 0 for s in segment_ids) + 1
    )

    segment_ids = tf.keras.preprocessing.sequence.pad_sequences(
      segment_ids,
      maxlen=self.padded_sequence_length,
      dtype="int32",
      padding="pre",
      truncating="post",
      value=segment_id_padding_element,
    )

    padded_node_sequence_length = min(
      self.padded_node_sequence_length, max(len(s) for s in selector_vectors)
=======
        max(max(s) if s.size else 0 for s in segment_ids) + 1
    )

    segment_ids = tf.keras.preprocessing.sequence.pad_sequences(
        segment_ids,
        maxlen=self.padded_sequence_length,
        dtype="int32",
        padding="pre",
        truncating="post",
        value=segment_id_padding_element,
    )

    padded_nodes_sequence_length = min(
        self.padded_nodes_sequence_length, max(len(s) for s in selector_vectors)
>>>>>>> 33703e4de... Split the LSTM into multiple modules.
    )

    # Pad the selector vectors to the same shape as the segment IDs.)
    selector_vectors = tf.keras.preprocessing.sequence.pad_sequences(
<<<<<<< HEAD
      selector_vectors,
      maxlen=padded_node_sequence_length,
      dtype="int32",
      padding="pre",
      truncating="post",
      value=np.array((0, 0), dtype=np.int32),
    )

    node_y = tf.keras.preprocessing.sequence.pad_sequences(
      node_y,
      maxlen=padded_node_sequence_length,
      dtype="int32",
      padding="pre",
      truncating="post",
      value=np.zeros(self.graph_db.node_y_dimensionality, dtype=np.int64),
    )

    all_node_indices = tf.keras.preprocessing.sequence.pad_sequences(
      all_node_indices,
      maxlen=padded_node_sequence_length,
      dtype="int32",
      padding="pre",
      truncating="post",
      value=-1,
    )

    return batches.Data(
      graph_ids=[graph.id for graph in graphs],
      data=NodeLstmBatch(
        encoded_sequences=encoded_sequences,
        segment_ids=segment_ids,
        selector_vectors=selector_vectors,
        node_y=node_y,
        node_indices=np.concatenate(all_node_indices),
        targets=np.vstack(targets),
      ),
    )
=======
        selector_vectors,
        maxlen=padded_nodes_sequence_length,
        dtype="int32",
        padding="pre",
        truncating="post",
        value=np.array((0, 0), dtype=np.int32),
    )

    node_y = tf.keras.preprocessing.sequence.pad_sequences(
        node_y,
        maxlen=padded_nodes_sequence_length,
        dtype="int32",
        padding="pre",
        truncating="post",
        value=np.zeros(self.graph_db.node_y_dimensionality, dtype=np.int64),
    )

    return batches.Data(
        graph_ids=[graph.id for graph in graphs],
        data=NodeLstmBatch(
            encoded_sequences=encoded_sequences,
            segment_ids=segment_ids,
            selector_vectors=selector_vectors,
            node_y=node_y,
            node_indices=node_indices,
        ),
    )

  def ReshapeTargets(self, targets):
    # Stack the manually-batched targets list.
    return np.vstack(targets)
>>>>>>> 33703e4de... Split the LSTM into multiple modules.
