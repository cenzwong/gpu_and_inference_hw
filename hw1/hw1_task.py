"""HW1 task entrypoint."""

from hw1_runtime import (
    measure_roofline_points,
    plot_roofline,
    print_header,
    save_roofline_data,
)


def run_hw1(
    lowest_ai_fn,
    make_compute_fn,
    benchmark_fn,
    compute_elementwise_metrics,
):
    print_header()
    print("Running benchmarks (first run compiles kernels via torch.compile)...")
    results = measure_roofline_points(
        lowest_ai_fn=lowest_ai_fn,
        make_compute_fn=make_compute_fn,
        benchmark_fn=benchmark_fn,
        compute_elementwise_metrics=compute_elementwise_metrics,
    )
    save_roofline_data(results)
    print("\nGenerating plots...")
    plot_roofline(results)
    print("\nDone! Check the results/ directory for plots.")


def main():
    from hw1_task_impl import (
        benchmark_fn,
        compute_elementwise_metrics,
        lowest_ai_fn,
        make_compute_fn,
    )

    run_hw1(
        lowest_ai_fn=lowest_ai_fn,
        make_compute_fn=make_compute_fn,
        benchmark_fn=benchmark_fn,
        compute_elementwise_metrics=compute_elementwise_metrics,
    )


if __name__ == "__main__":
    main()
