# import sentencepiece as spm
# # create file model and vocab
# spm.SentencePieceTrainer.train(
#     input="data_train.txt",
#     model_prefix="tokenizer",
#     vocab_size=32000,
#     # model_type="bpe",
#     character_coverage=0.9995,
#     pad_id=0,
#     unk_id=1,
#     bos_id=2,
#     eos_id=3,
#     pad_piece='[PAD]',
#     unk_piece='[UNK]',
#     bos_piece='[CLS]',
#     eos_piece='[SEP]',
#     user_defined_symbols='[MASK]',
#     model_type='unigram'
# )

from tokenizers import Tokenizer, trainers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
import sentencepiece as spm

# Khởi tạo Tokenizer với mô hình BPE
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
special_tokens = ["<unk>", "<s>", "</s>", "[PAD]", "<|im_start|>", "<|im_end|>"]
tokenizer.add_special_tokens(special_tokens)

# Thêm bước tiền xử lý
from tokenizers.normalizers import Sequence, Prepend, Replace
tokenizer.normalizer = Sequence([
    Prepend("_"),
    Replace(" ", "__")
])


trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=32000)
tokenizer.train(["data_train.txt"], trainer)

# Cấu hình hậu xử lý để thêm token đặc biệt vào các chuỗi
tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s>:0 <s> $B </s>:1",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("</s>", tokenizer.token_to_id("</s>"))
    ]
)

# Chuyển đổi sang Tokenizer tương thích với HuggingFace
awesome_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    pad_token="[PAD]",
    mask_token="<|im_end|>",
)

# Lưu Tokenizer
awesome_tokenizer.save_pretrained("awesome_tokenizer")

# Huấn luyện SentencePiece Tokenizer
spm.SentencePieceTrainer.train(
    input="data_train.txt",
    model_prefix="tokenizer",
    vocab_size=32000,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="[PAD]",
    unk_piece="<unk>",
    bos_piece="<s>",
    eos_piece="</s>",
    user_defined_symbols=["<|im_start|>", "<|im_end|>"]
)
# spm.SentencePieceTrainer.train(
#     input="data_train.txt",
#     model_prefix="tokenizer",
#     vocab_size=32000,
#     # model_type="bpe",
#     character_coverage=0.9995,
#     pad_id=0,
#     unk_id=1,
#     bos_id=2,
#     eos_id=3,
#     pad_piece='[PAD]',
#     unk_piece='[UNK]',
#     bos_piece='[CLS]',
#     eos_piece='[SEP]',
#     user_defined_symbols='[MASK]',
#     model_type='unigram'
# )
# In thông tin cấu hình
print("Tokenizer configuration:")
print("- Vocabulary size:", len(awesome_tokenizer))
print("- Special tokens:", special_tokens)
