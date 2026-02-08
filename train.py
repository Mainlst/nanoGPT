"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText

# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = False  # disabled by default
wandb_project = "owt"
wandb_run_name = "gpt2"  # 'run' + str(time.time())

# data
dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, ~= learning_rate/10 per Chinchilla

# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.

# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', or try 'mps' on macbooks
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16'; float16 uses GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
vocab_source = "gpt2"  # added to fix ValueError when running with --vocab_source

# -----------------------------------------------------------------------------

config_keys = [
    key
    for key, value in globals().items()
    if not key.startswith("_") and isinstance(value, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {key: globals()[key] for key in config_keys}  # useful for logging

# -----------------------------------------------------------------------------
# various inits, derived attributes, I/O setup

ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process handles logging/checkpointing
    seed_offset = ddp_rank  # each process gets a different seed

    # world_size processes train simultaneously, so scale down per-process
    # gradient accumulation iterations proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = (
    gradient_accumulation_steps * ddp_world_size * batch_size * block_size
)
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for torch.autocast

# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# poor man's data loader
data_dir = os.path.join("data", dataset)


def get_batch(split):
    # Recreate np.memmap every batch to avoid a memory leak, as per:
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )

    if device_type == "cuda":
        # pin arrays x,y so transfer to GPU can be async (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


# init these up here, can override if init_from='resume' (checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = {
    "n_layer": n_layer,
    "n_head": n_head,
    "n_embd": n_embd,
    "block_size": block_size,
    "bias": bias,
    "vocab_size": None,
    "dropout": dropout,
}

required_model_keys = (
    "n_layer",
    "n_head",
    "n_embd",
    "block_size",
    "bias",
    "vocab_size",
)

if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")

    # determine vocab size for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up)")
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == "resume":
    print(f"Resuming training from {out_dir}")

    # resume training from a checkpoint
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]

    # force these config attributes to be equal or resume will fail
    # other attributes (e.g. dropout) can remain as set on command line
    for key in required_model_keys:
        model_args[key] = checkpoint_model_args[key]

    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint["model"]

    # Fix state dict keys when this prefix appears unexpectedly.
    unwanted_prefix = "_orig_mod."
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")

    # initialize from OpenAI GPT-2 weights
    override_args = {"dropout": dropout}
    model = GPT.from_pretrained(init_from, override_args)

    # read off created config params so checkpoints have correct values
    for key in required_model_keys:
        model_args[key] = getattr(model.config, key)

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = block_size  # checkpoint should store the cropped value

model.to(device)

# initialize a GradScaler. If enabled=False, scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for idx in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[idx] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)

    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr

    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# training loop
X, Y = get_batch("train")  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
            )

        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    if iter_num == 0 and eval_only:
        break

    # forward/backward/update with optional grad accumulation
    # and using GradScaler when dtype is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # In DDP training we only need to sync gradients at the last micro step.
            # The official way is model.no_sync(), but this direct flag change is
            # equivalent and avoids duplicating the forward/backward code path.
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )

        with ctx:
            logits, loss = model(X, Y)
            loss = (
                loss / gradient_accumulation_steps
            )  # scale loss for gradient accumulation

        # async prefetch next batch while GPU is running current pass
        X, Y = get_batch("train")

        # backward pass (with grad scaling in fp16)
        scaler.scale(loss).backward()

    # clip gradients
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # step optimizer/scaler
    scaler.step(optimizer)
    scaler.update()

    # flush gradients quickly; no need to keep this memory around
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        # get loss as float; this is a CPU-GPU sync point
        # scale up to undo division above, approximating true total loss
        lossf = loss.item() * gradient_accumulation_steps

        if local_iter_num >= 5:  # let training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        print(
            f"iter {iter_num}: loss {lossf:.4f}, "
            f"time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
        )

    iter_num += 1
    local_iter_num += 1

    # termination condition
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
