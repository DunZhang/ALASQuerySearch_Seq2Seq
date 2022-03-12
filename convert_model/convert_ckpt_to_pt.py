from roformer.convert_roformer_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
from transformers.convert_graph_to_onnx import convert
if __name__ == "__main__":
    convert_tf_checkpoint_to_pytorch(
        tf_checkpoint_path="../output/seq2seq_nosim/bert_model.ckpt",
        bert_config_file="../output/seq2seq_nosim/bert_config.json",
        pytorch_dump_path="../output/seq2seq_nosim/pt/pytorch_model.bin")
