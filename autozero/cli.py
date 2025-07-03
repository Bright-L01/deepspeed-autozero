#!/usr/bin/env python3
"""
Simplified CLI interface for DeepSpeed AutoZero configuration generator.
Based on design document requirements with evidence-based safety factors.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="autozero",
    help="DeepSpeed ZeRO configuration generator - streamlines ZeRO stage configuration",
    add_completion=False,  # Simplified
)

console = Console()


@app.command()
def main(
    model_path: Path = typer.Argument(
        ...,
        help="Path to Python file containing the model definition (nn.Module)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    gpus: int = typer.Option(
        1,
        "--gpus",
        "-g",
        min=1,
        help="Number of GPUs to use for training",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        min=1,
        help="Training batch size per GPU",
    ),
    dtype: str = typer.Option(
        "fp16",
        "--dtype",
        "-d",
        help="Data type: fp32, fp16, or bf16",
    ),
    cpu_offload: bool = typer.Option(
        False,
        "--cpu-offload",
        help="Enable CPU offloading for large models",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (defaults to stdout)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    """
    Generate optimal DeepSpeed configuration for your model.

    Analyzes model architecture and hardware to recommend the best ZeRO stage
    and configuration parameters with evidence-based safety factors.

    Example:
        autozero model.py --gpus=4 --batch-size=8 --dtype=bf16
    """
    try:
        # Import modules dynamically to avoid circular imports
        from .model_analyzer import ModelAnalyzer
        from .memory_estimator import MemoryEstimator
        from .config_generator import ConfigGenerator
        from .utils import validate_model_file, get_dtype_from_string, get_system_info

        if verbose:
            console.print("[bold blue]DeepSpeed AutoZero[/bold blue]")
            console.print(f"Model: {model_path}")
            console.print(f"GPUs: {gpus}, Batch Size: {batch_size}, Type: {dtype}")
            console.print()

        # Validate inputs
        validate_model_file(model_path)
        if dtype not in ["fp32", "fp16", "bf16"]:
            raise ValueError(f"Invalid dtype: {dtype}. Must be fp32, fp16, or bf16")

        # Initialize components
        analyzer = ModelAnalyzer()
        estimator = MemoryEstimator(verbose=verbose)
        generator = ConfigGenerator()

        # Load and analyze model
        if verbose:
            console.print("Loading model...")
        model = analyzer.load_model_from_file(model_path)
        model_info = analyzer.analyze_model(model, get_dtype_from_string(dtype))

        if verbose:
            console.print(f"Model parameters: {model_info.total_params:,}")
            console.print(f"Model size: {model_info.model_size_mb:.1f} MB")

        # Estimate memory and select stage
        if verbose:
            console.print("Estimating memory requirements...")
        stage, offload_needed, estimate = estimator.select_optimal_stage(
            model, num_gpus=gpus, batch_size=batch_size, force_offload=cpu_offload
        )

        if verbose:
            console.print(f"Recommended ZeRO stage: {stage}")
            console.print(f"GPU memory needed: {estimate.gpu_memory_gb:.2f} GB")
            if offload_needed:
                console.print("[yellow]CPU offloading enabled[/yellow]")
            for warning in estimate.warnings:
                console.print(f"[yellow]Warning:[/yellow] {warning}")

        # Generate configuration
        config = generator.generate_config(
            model_info=model_info,
            stage=stage,
            batch_size=batch_size,
            num_gpus=gpus,
            dtype=dtype,
            cpu_offload=offload_needed,
        )

        # Output
        import json

        config_json = json.dumps(config, indent=2)

        if output:
            output.write_text(config_json)
            console.print(f"[green]✓[/green] Configuration saved to: {output}")
        else:
            print(config_json)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(code=1)


@app.command()
def validate(
    config_file: Path = typer.Argument(
        ...,
        help="Path to DeepSpeed configuration JSON file to validate",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
):
    """Validate a DeepSpeed configuration file."""
    try:
        from .config_generator import ConfigGenerator
        import json

        generator = ConfigGenerator()

        # Load and validate configuration
        with open(config_file, "r") as f:
            config = json.load(f)

        errors = generator.validate_config(config)

        if errors:
            console.print(f"[red]Validation failed:[/red]")
            for error in errors:
                console.print(f"  - {error}")
            raise typer.Exit(code=1)
        else:
            console.print(f"[green]✓[/green] Valid configuration: {config_file}")
            if "zero_optimization" in config:
                stage = config["zero_optimization"].get("stage", 0)
                console.print(f"  ZeRO Stage: {stage}")

    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON:[/red] {str(e)}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def version():
    """Show version information."""
    from . import __version__

    console.print(f"DeepSpeed AutoZero v{__version__}")


if __name__ == "__main__":
    app()
