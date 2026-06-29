import argparse
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.preprocess.dataset import PackedTokenDataset
from models.LFM2.model import LFM2, LFM2_5_350M_Config


@dataclass
class TrainingConfig:
    name: str = "local"
    manifest_path: str = "/home/yuxin/workspace/data/datasets/llmfactory_pretrain_v0/manifest.json"
    checkpoint_dir: str = "./checkpoints/LFM2_5"
    context_length: int = 2048
    epochs: int = 1
    train_batch_size: int = 1
    val_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 1_000
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    activation_checkpointing: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    eval_interval: int = 10_000
    log_interval: int = 10
    max_steps: int | None = None
    max_eval_steps: int | None = None
    resume_from: str | None = None
    peak_bf16_tflops_per_gpu: float | None = None
    compile_model: bool = False
    compile_mode: str = "default"
    wandb_enabled: bool = True
    wandb_project: str = "llmfactory"
    wandb_run_name: str | None = None
    seed: int = 42


@dataclass
class H100TrainingConfig(TrainingConfig):
    name: str = "h100"
    train_batch_size: int = 8
    val_batch_size: int = 8
    activation_checkpointing: bool = True
    peak_bf16_tflops_per_gpu: float = 989.0


@dataclass
class H200TrainingConfig(TrainingConfig):
    name: str = "h200"
    train_batch_size: int = 16
    val_batch_size: int = 16
    activation_checkpointing: bool = True
    peak_bf16_tflops_per_gpu: float = 989.0


@dataclass
class B200TrainingConfig(TrainingConfig):
    name: str = "b200"
    manifest_path: str = "/workspace/data/datasets/llmfactory_pretrain_v0/manifest.json"
    train_batch_size: int = 16
    val_batch_size: int = 16
    activation_checkpointing: bool = True
    peak_bf16_tflops_per_gpu: float = 2250.0


@dataclass
class B300TrainingConfig(TrainingConfig):
    name: str = "b300"
    train_batch_size: int = 36
    val_batch_size: int = 36
    activation_checkpointing: bool = True
    peak_bf16_tflops_per_gpu: float = 2500.0


TRAINING_CONFIGS = {
    "local": TrainingConfig,
    "h100": H100TrainingConfig,
    "h200": H200TrainingConfig,
    "b200": B200TrainingConfig,
    "b300": B300TrainingConfig,
}


def setup_ddp():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return local_rank, rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain LFM2.5 with DDP.")
    parser.add_argument("--config", choices=sorted(TRAINING_CONFIGS), default="local")
    parser.add_argument("--manifest-path", type=str)
    parser.add_argument("--checkpoint-dir", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--val-batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-eval-steps", type=int)
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--peak-bf16-tflops", type=float)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", default=None)
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-run-name", type=str)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def build_config(args) -> TrainingConfig:
    config = TRAINING_CONFIGS[args.config]()
    overrides = {
        "manifest_path": args.manifest_path,
        "checkpoint_dir": args.checkpoint_dir,
        "epochs": args.epochs,
        "train_batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "learning_rate": args.lr,
        "min_learning_rate": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "eval_interval": args.eval_interval,
        "log_interval": args.log_interval,
        "max_steps": args.max_steps,
        "max_eval_steps": args.max_eval_steps,
        "resume_from": args.resume,
        "peak_bf16_tflops_per_gpu": args.peak_bf16_tflops,
        "compile_mode": args.compile_mode,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    if args.disable_wandb:
        config.wandb_enabled = False
    if args.activation_checkpointing is not None:
        config.activation_checkpointing = args.activation_checkpointing
    if args.compile is not None:
        config.compile_model = args.compile
    return config


def build_lr_scheduler(optimizer, config: TrainingConfig, total_steps: int):
    min_lr_ratio = config.min_learning_rate / config.learning_rate
    warmup_steps = min(config.warmup_steps, max(total_steps - 1, 0))

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_resume_path(config: TrainingConfig, latest_checkpoint_path: Path) -> Path | None:
    if config.resume_from is None:
        return None
    if config.resume_from == "latest":
        return latest_checkpoint_path
    return Path(config.resume_from)


def load_checkpoint(path: Path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    unwrapped_model = unwrap_model(model)
    unwrapped_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return {
        "epoch": int(checkpoint.get("epoch", 0)),
        "step": int(checkpoint.get("step", 0)),
        "next_micro_step": int(checkpoint.get("next_micro_step", 0)),
        "best_val_loss": float(checkpoint.get("best_val_loss", float("inf"))),
    }


def init_wandb(config: TrainingConfig, rank: int, world_size: int):
    if rank != 0 or not config.wandb_enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed. Install project dependencies or pass --disable-wandb.") from exc

    run_name = config.wandb_run_name or f"LFM2_5-{config.name}-{world_size}gpu"
    return wandb.init(
        project=config.wandb_project,
        name=run_name,
        config={**asdict(config), "world_size": world_size},
    )


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    value = value.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / world_size


@torch.no_grad()
def evaluate(model, val_loader, loss_fn, device, config: TrainingConfig):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0, dtype=torch.long, device=device)

    for eval_step, batch in enumerate(val_loader):
        if config.max_eval_steps is not None and eval_step >= config.max_eval_steps:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        compile_step_begin(config)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = loss_fn(logits.float().transpose(1, 2), labels)

        tokens = labels.numel()
        total_loss += loss.detach() * tokens
        total_tokens += tokens

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    mean_loss = total_loss / total_tokens.clamp_min(1)
    model.train()
    return mean_loss.item()


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    config: TrainingConfig,
    epoch: int,
    step: int,
    next_micro_step: int,
    val_loss: float,
    best_val_loss: float,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    unwrapped_model = unwrap_model(model)
    torch.save(
        {
            "model": unwrapped_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": asdict(config),
            "epoch": epoch,
            "step": step,
            "next_micro_step": next_micro_step,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def log_metrics(wandb_run, metrics: dict[str, float], step: int):
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def unwrap_model(model):
    if isinstance(model, DDP):
        model = model.module
    return getattr(model, "_orig_mod", model)


def compile_step_begin(config: TrainingConfig):
    if config.compile_model and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()


def estimate_mfu(tokens_per_second: float, num_parameters: int, world_size: int, peak_bf16_tflops_per_gpu: float | None):
    achieved_tflops = 6 * num_parameters * tokens_per_second / 1e12
    if peak_bf16_tflops_per_gpu is None:
        return achieved_tflops, None
    mfu = achieved_tflops / (peak_bf16_tflops_per_gpu * world_size)
    return achieved_tflops, mfu


def main():
    args = parse_args()
    config = build_config(args)
    local_rank, rank, world_size = setup_ddp()

    device = torch.device("cuda", local_rank)
    torch.manual_seed(config.seed + rank)
    torch.cuda.manual_seed(config.seed + rank)

    model = LFM2(LFM2_5_350M_Config())
    model.set_gradient_checkpointing(config.activation_checkpointing)
    model.to(device)

    if config.compile_model:
        print(f"Compiling model with mode {config.compile_mode}...")
        model.warmup_caches(config.context_length, device, torch.bfloat16)
        model = torch.compile(model, mode=config.compile_mode)

    model = DDP(model, device_ids=[local_rank])
    num_parameters = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    train_dataset = PackedTokenDataset(
        manifest_path=config.manifest_path,
        split="train",
        context_length=config.context_length,
    )
    val_dataset = PackedTokenDataset(
        manifest_path=config.manifest_path,
        split="val",
        context_length=config.context_length,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    wandb_run = init_wandb(config, rank, world_size)
    checkpoint_dir = Path(config.checkpoint_dir)
    latest_checkpoint_path = checkpoint_dir / "latest.pt"
    best_checkpoint_path = checkpoint_dir / "best.pt"
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_training_steps = config.max_steps or (updates_per_epoch * config.epochs)
    scheduler = build_lr_scheduler(optimizer, config, total_training_steps)
    best_val_loss = float("inf")
    global_step = 0
    last_eval_step = -1
    start_epoch = 0
    resume_next_micro_step = 0

    resume_path = resolve_resume_path(config, latest_checkpoint_path)
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        resume_state = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        start_epoch = max(resume_state["epoch"] - 1, 0)
        global_step = resume_state["step"]
        resume_next_micro_step = resume_state["next_micro_step"]
        if resume_next_micro_step >= len(train_loader):
            start_epoch += 1
            resume_next_micro_step = 0
        best_val_loss = resume_state["best_val_loss"]
        last_eval_step = global_step if global_step % config.eval_interval == 0 else -1
        if rank == 0:
            print(f"Resumed from {resume_path} at step {global_step}")

    if rank == 0:
        tokens_per_optimizer_step = (
            config.context_length
            * config.train_batch_size
            * world_size
            * config.gradient_accumulation_steps
        )
        print(f"Using training config: {config.name}")
        print(f"Tokens per optimizer step: {tokens_per_optimizer_step:,}")
        print(f"Total planned optimizer steps: {total_training_steps:,}")
        print(f"Activation checkpointing: {config.activation_checkpointing}")
        print(f"torch.compile: {config.compile_model} ({config.compile_mode})")

    try:
        optimizer.zero_grad(set_to_none=True)
        last_log_time = time.perf_counter()
        last_log_tokens = global_step * config.context_length * config.train_batch_size * world_size * config.gradient_accumulation_steps
        stop_training = False

        for epoch in range(start_epoch, config.epochs):
            train_sampler.set_epoch(epoch)
            skip_micro_steps = resume_next_micro_step if epoch == start_epoch else 0

            for micro_step, batch in enumerate(train_loader):
                if micro_step < skip_micro_steps:
                    continue

                if config.max_steps is not None and global_step >= config.max_steps:
                    stop_training = True
                    break

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                should_step = (
                    (micro_step + 1) % config.gradient_accumulation_steps == 0
                    or micro_step + 1 == len(train_loader)
                )

                compile_step_begin(config)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids)
                    loss = loss_fn(logits.float().transpose(1, 2), labels)

                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {global_step}: {loss.item()}")

                sync_context = nullcontext() if should_step else model.no_sync()
                with sync_context:
                    (loss / config.gradient_accumulation_steps).backward()

                if not should_step:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite grad norm at step {global_step}: {grad_norm.item()}")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                train_loss = reduce_mean(loss, world_size).item()
                tokens_seen = global_step * config.context_length * config.train_batch_size * world_size * config.gradient_accumulation_steps

                if rank == 0 and global_step % config.log_interval == 0:
                    now = time.perf_counter()
                    elapsed = max(now - last_log_time, 1e-9)
                    tokens_per_second = (tokens_seen - last_log_tokens) / elapsed
                    achieved_tflops, mfu = estimate_mfu(
                        tokens_per_second,
                        num_parameters,
                        world_size,
                        config.peak_bf16_tflops_per_gpu,
                    )
                    train_ppl = math.exp(min(train_loss, 20.0))
                    print(
                        f"epoch {epoch + 1} step {global_step}/{total_training_steps} "
                        f"loss {train_loss:.4f} ppl {train_ppl:.2f} grad_norm {grad_norm.item():.4f} "
                        f"tok/s {tokens_per_second:.0f}"
                    )
                    metrics = {
                        "train/loss": train_loss,
                        "train/perplexity": train_ppl,
                        "train/grad_norm_before_clip": grad_norm.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/tokens_seen": tokens_seen,
                        "throughput/tokens_per_second": tokens_per_second,
                        "throughput/achieved_tflops": achieved_tflops,
                        "epoch": epoch + 1,
                    }
                    if mfu is not None:
                        metrics["throughput/mfu"] = mfu
                    log_metrics(
                        wandb_run,
                        metrics,
                        global_step,
                    )
                    last_log_time = now
                    last_log_tokens = tokens_seen

                if global_step % config.eval_interval == 0:
                    val_loss = evaluate(model, val_loader, loss_fn, device, config)
                    val_ppl = math.exp(min(val_loss, 20.0))
                    last_eval_step = global_step

                    if rank == 0:
                        print(f"eval step {global_step} val_loss {val_loss:.4f} val_ppl {val_ppl:.2f}")
                        is_best = val_loss < best_val_loss
                        best_val_loss = min(best_val_loss, val_loss)
                        save_checkpoint(latest_checkpoint_path, model, optimizer, scheduler, config, epoch + 1, global_step, micro_step + 1, val_loss, best_val_loss)
                        if is_best:
                            save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, config, epoch + 1, global_step, micro_step + 1, val_loss, best_val_loss)
                        log_metrics(
                            wandb_run,
                            {
                                "val/loss": val_loss,
                                "val/perplexity": val_ppl,
                                "val/best_loss": best_val_loss,
                            },
                            global_step,
                        )
                    dist.barrier()
                    last_log_time = time.perf_counter()
                    last_log_tokens = tokens_seen

            if stop_training:
                break

            val_loss = evaluate(model, val_loader, loss_fn, device, config)
            val_ppl = math.exp(min(val_loss, 20.0))
            last_eval_step = global_step

            if rank == 0:
                print(f"epoch {epoch + 1} done step {global_step} val_loss {val_loss:.4f} val_ppl {val_ppl:.2f}")
                is_best = val_loss < best_val_loss
                best_val_loss = min(best_val_loss, val_loss)
                epoch_checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
                epoch_end_micro_step = len(train_loader)
                save_checkpoint(latest_checkpoint_path, model, optimizer, scheduler, config, epoch + 1, global_step, epoch_end_micro_step, val_loss, best_val_loss)
                save_checkpoint(epoch_checkpoint_path, model, optimizer, scheduler, config, epoch + 1, global_step, epoch_end_micro_step, val_loss, best_val_loss)
                if is_best:
                    save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, config, epoch + 1, global_step, epoch_end_micro_step, val_loss, best_val_loss)
                log_metrics(
                    wandb_run,
                    {
                        "val/loss": val_loss,
                        "val/perplexity": val_ppl,
                        "val/best_loss": best_val_loss,
                    },
                    global_step,
                )
            dist.barrier()
            last_log_time = time.perf_counter()
            last_log_tokens = global_step * config.context_length * config.train_batch_size * world_size * config.gradient_accumulation_steps

        if global_step != last_eval_step:
            val_loss = evaluate(model, val_loader, loss_fn, device, config)
            val_ppl = math.exp(min(val_loss, 20.0))
            if rank == 0:
                is_best = val_loss < best_val_loss
                best_val_loss = min(best_val_loss, val_loss)
                save_checkpoint(latest_checkpoint_path, model, optimizer, scheduler, config, config.epochs, global_step, 0, val_loss, best_val_loss)
                if is_best:
                    save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, config, config.epochs, global_step, 0, val_loss, best_val_loss)
                log_metrics(
                    wandb_run,
                    {
                        "val/loss": val_loss,
                        "val/perplexity": val_ppl,
                        "val/best_loss": best_val_loss,
                    },
                    global_step,
                )
            dist.barrier()

    finally:
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_ddp()


if __name__ == "__main__":
    main()