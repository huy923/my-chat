# import sentencepiece as spm

# sp = spm.SentencePieceProcessor()
# sp.load("tokenizer.model")

# text = "Xin chào, đây là mô hình của tôi!"
# tokens = sp.encode(text, out_type=str)
# decoded_text = sp.decode(sp.encode(text))

# print("Tokens:", tokens)
# print("Decoded:", decoded_text)


# import sentencepiece as spm

# sp = spm.SentencePieceProcessor()
# sp.load("tokenizer.model")

# print("Tokens:", sp.encode("Xin chào! Đây là một tập dữ liệu mẫu", out_type=str))
# print("Decoded:", sp.decode(sp.encode("Xin chào! Đây là một tập dữ liệu mẫu", out_type=int)))


# from transformers import LlamaTokenizer

# tokenizer = LlamaTokenizer(vocab_file="tokenizer.model")
# tokenizer.save_pretrained(".")
