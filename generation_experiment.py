import sys
import run_generation
import click

END_OF_TEXT = "<|endoftext|>"


@click.command()
@click.option("--model_path", default="gpt2")
@click.option("--prompt", default=END_OF_TEXT)
@click.option("--num_generations", default=1)
@click.option("--batch_size", default=32)
@click.option("--length", default=200)
@click.option("--output_dir", default=None)
def generate(
        model_path: str,
        prompt: str,
        length: int,
        num_generations,
        batch_size,
        output_dir: str,
        model_type="gpt2",
        stop_token=END_OF_TEXT,
):
    sys.argv = [
        "run_generation.py",
        f"--model_type={model_type}",
        f"--model_name_or_path={model_path}",
        f"--prompt={prompt}",
        f"--num_generations={num_generations}",
        f"--batch_size={batch_size}",
        f"--length={length}",
        f"--stop_token={stop_token}"
    ]
    if output_dir:
        sys.argv.append(f"output_dir {output_dir}")

    run_generation.main()


if __name__ == '__main__':
    generate()
