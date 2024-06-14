Chapters:
00:00:00 intro: Let’s reproduce GPT-2 (124M)
00:03:39 exploring the GPT-2 (124M) OpenAI checkpoint
00:13:47 SECTION 1: implementing the GPT-2 nn.Module
00:28:08 loading the huggingface/GPT-2 parameters
00:31:00 implementing the forward pass to get logits
00:33:31 sampling init, prefix tokens, tokenization
00:37:02 sampling loop
00:41:47 sample, auto-detect the device
00:45:50 let’s train: data batches (B,T) → logits (B,T,C)
00:52:53 cross entropy loss
00:56:42 optimization loop: overfit a single batch
01:02:00 data loader lite
01:06:14 parameter sharing wte and lm_head
01:13:47 model initialization: std 0.02, residual init
01:22:18 SECTION 2: Let’s make it fast. GPUs, mixed precision, 1000ms
01:28:14 Tensor Cores, timing the code, TF32 precision, 333ms
01:39:38 float16, gradient scalers, bfloat16, 300ms
01:48:15 torch.compile, Python overhead, kernel fusion, 130ms
02:00:18 flash attention, 96ms
02:06:54 nice/ugly numbers. vocab size 50257 → 50304, 93ms
02:14:55 SECTION 3: hyperpamaters, AdamW, gradient clipping
02:21:06 learning rate scheduler: warmup + cosine decay
02:26:21 batch size schedule, weight decay, FusedAdamW, 90ms
02:34:09 gradient accumulation
02:46:52 distributed data parallel (DDP)
03:10:21 datasets used in GPT-2, GPT-3, FineWeb (EDU)
03:23:10 validation data split, validation loss, sampling revive
03:28:23 evaluation: HellaSwag, starting the run
03:43:05 SECTION 4: results in the morning! GPT-2, GPT-3 repro
03:56:21 shoutout to llm.c, equivalent but faster code in raw C/CUDA
03:59:39 summary, phew, build-nanogpt github repo


notes on new karpathy video (https://www.youtube.com/watch?v=l8pRSuU81PU&ab_channel=AndrejKarpathy):

- cross attention is the encoder ⇒ decoder (non-masked attn) part in the attn is all you need architecture. “cross attention”
- lots of the mods including learnable pos enc, post/pre norm in my LLM course are directly from the GPT-2 paper (https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- GPT-3 paper (https://arxiv.org/pdf/2005.14165)
- clean residual pathway is desirable (no layernorm computation graph leafs in the residual stream)
- causualattn in the new lecture is more pytorch efficient than the nanoGPT version
- gpt-2 uses GELU activation
- we take the printed out architecture from "gpt-2" huggingface and map its dimensions to a pytorch architecture to train from scratch
- we treat AdamW as a blackbox since its more involved
- the embedding matrix & logit-level linear classifer (nn.Linear before softmax) are identical. paper ref -> https://arxiv.org/pdf/1608.05859. since the tokens should be semantically similar during the embedding phase and output classification phase, we can hack and place the same tensors on completely opposite sides of the architecture and have it work better (save memory because its just a data ptr to the same tensor at the end of the day). we save significant memory from this approach when doing the math on gpt-2 total weights. (50257 * 768 = ~40M params = ~33% of gpt-2 total weights). we can hardcode these or find some trick to make it faster, by default these are weights we don't have to include in the torch graph OR optimizer state 
- we apparently scale by 1/sqrt(N) where N is n_layer in the residual stream. by the time we have accumulated all the sums at the end of the stream,  our stddev remains stable
- mixed precision is in the middle of the lecture. good to review float32, tensorfloat 32, float 16, bfloat16 (brain float). the gradient scalars are a complexity issue in fp16 so industry std is bf16.
- mixed precision on torch docs (https://pytorch.org/docs/stable/notes/amp_examples.html)
- its unclear what is converted to diff precision and when so take it for granted :D
- what is `torch.compile`?
    - `model = torch.compile(model)`
    - copy paste from torch docs: "Speedup mainly comes from reducing Python overhead and GPU read/writes, and so the observed speedup may vary on factors such as model architecture and batch size. For example, if a model’s architecture is simple and the amount of data is large, then the bottleneck would be GPU compute and the observed speedup may be less significant."
    - removes the python interpreter entirely
    - takes advantage of kernel fusion. instead of doing all these expensive reads and writes back and forth between HBM and registers, we do all operations on chip by having the compiler know the lifetime of an iteration and the internal variables we are moving around. the python interpreter is stupid so we take up some additional time during the init to make sure this model runs really fast.
    - karpathy uses compiled GELU to illusrate this @ ~1:50:00
- flash attention @ ~1:56:00
    - HBM is off chip, shared & registers on chip (faster)
    - pytorch `scaled_dot_product_attention` to call custom fused attention kernel
- turning ugly numbers to nice numbers is a hack (50257 vocab size -> 50304). 50304 is divisible by 2, 4, 8, 16, 32, 64, 128
- we have the industry standard kernels to do the nice "powers of 2" part and the less efficient kernels will do the cleanup job, hence taking up a stupid amount of compute time 
- skipped parts from 2:18:00 - 3:10:00
- for data mining standards -> https://huggingface.co/datasets/HuggingFaceFW/fineweb
- lecture repo ->  https://github.com/karpathy/build-nanogpt
- what is `torchrun`?
- eleuther harness is an eval tool for researching LMs
-  
