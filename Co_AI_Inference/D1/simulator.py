"""A small, deterministic-capable simulator for a single LLM server.

Requests arrive over time, wait while the server is busy, and are processed in
first-in, first-out (FIFO) order.  Running this module prints one row per
request and a short summary.

the queuing policy is rudimentry, all requests are put into one queue and to be 
processed one by one( aka, only one batch of requests). it is to show end-to-end 
latency is queuing time + llm process time, where llm process time is a function of 
requests tokoen, prefill cost , output token, decode cost.

further developed queuing policies could be designed so that each batch of requests 
latency meets the latency budget.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Iterable, Sequence


PROMPT_TOKEN_CHOICES = (64, 128, 256, 512, 1024, 2048, 4096)
INTERARRIVAL_TIME_CHOICES = (
    0.001,
    0.002,
    0.003,
    0.005,
    0.006,
    0.007,
    0.008,
    0.009,
    0.010,
)

DEFAULT_LATENCY_BUDGET = 0.2


@dataclass(slots=True)
class MetadataRequest:
    """Metadata and measured timings for one inference request."""

    id: int
    arrival_time: float
    prompt_tokens: int
    expected_output_tokens: int | None
    deadline: float
    queue_waiting_time: float = 0.0
    llm_response_time: float | None = None
    completion_time: float | None = None

    @property
    def met_deadline(self) -> bool | None:
        """Return whether the request met its deadline, if it has completed."""
        if self.completion_time is None:
            return None
        return self.completion_time <= self.deadline

    @property
    def total_latency(self) -> float | None:
        """Return end-to-end latency, if the request has completed."""
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time


# Backwards-compatible alias for the name used in the original skeleton.
Meta_data_request = MetadataRequest


def expected_token_given_prompt_token(prompt_token: int) -> int:
    """Estimate output length as half the prompt length. just a simple func"""
    if prompt_token < 0:
        raise ValueError("prompt_token must be non-negative")
    return round(prompt_token / 2)



def estimate_service_time(
    prompt_tokens: int,
    output_tokens: int,
) -> float:
    
    """
    llm response time being function of below factors
    Prompt length
    Expected output length
    Prefill cost
    Decode cost
    """

    PREFILL_SECONDS_PER_TOKEN = 0.000002
    DECODE_SECONDS_PER_TOKEN = 0.000050

    prefill_time = prompt_tokens * PREFILL_SECONDS_PER_TOKEN
    decode_time = output_tokens * DECODE_SECONDS_PER_TOKEN
    return prefill_time + decode_time


def requests_sim(
    request_id: int = 0,
    current_time: float = 0.0,
    latency_budget: float = DEFAULT_LATENCY_BUDGET,
    rng: random.Random | None = None,
) -> MetadataRequest:
    """Generate one request arriving after ``current_time``."""
    if latency_budget <= 0:
        raise ValueError("latency_budget must be positive")

    generator = rng or random
    arrival_time = current_time + generator.choice(INTERARRIVAL_TIME_CHOICES)
    prompt_tokens = generator.choice(PROMPT_TOKEN_CHOICES)
    expected_output_tokens = expected_token_given_prompt_token(prompt_tokens)

    return MetadataRequest(
        id=request_id,
        arrival_time=arrival_time,
        prompt_tokens=prompt_tokens,
        expected_output_tokens=expected_output_tokens,
        deadline=arrival_time + latency_budget,
    )


def queue_requests(
    request_count: int = 15,
    latency_budget: float = DEFAULT_LATENCY_BUDGET,
    rng: random.Random | None = None,
) -> list[MetadataRequest]:
    """Generate a FIFO queue with monotonically increasing arrival times."""
    if request_count < 0:
        raise ValueError("request_count must be non-negative")

    generator = rng or random
    requests: list[MetadataRequest] = []
    current_time = 0.0

    for request_id in range(request_count):
        request = requests_sim(
            request_id=request_id,
            current_time=current_time,
            latency_budget=latency_budget,
            rng=generator,
        )
        requests.append(request)
        current_time = request.arrival_time

    return requests


def consume_queue(
    queued_requests: Sequence[MetadataRequest],
    rng: random.Random | None = None,
) -> list[MetadataRequest]:
    """Process requests on one server in FIFO order.

    Queueing time is derived from server availability rather than sampled
    independently, so a request waits only when an earlier request is still
    being processed.
    """
    generator = rng or random
    server_available_at = 0.0

    for request in queued_requests:
        response_time =estimate_service_time(request.prompt_tokens,
                                            request.expected_output_tokens)
        service_start = max(request.arrival_time, server_available_at)

        request.queue_waiting_time = service_start - request.arrival_time
        request.llm_response_time = response_time

        server_available_at = service_start + response_time

    return list(queued_requests)


def calculate_completion_time(
    processed_requests: Sequence[MetadataRequest],
) -> list[MetadataRequest]:
    """Calculate completion time for every processed request."""
    for request in processed_requests:
        if request.llm_response_time is None:
            raise ValueError(
                f"request {request.id} has not been processed by consume_queue"
            )
        request.completion_time = (
            request.arrival_time
            + request.queue_waiting_time
            + request.llm_response_time
        )

    return list(processed_requests)


def run_simulation(
    request_count: int = 15,
    latency_budget: float = DEFAULT_LATENCY_BUDGET,
    seed: int | None = None,
) -> list[MetadataRequest]:
    """Run all simulation stages and return the completed requests."""
    rng = random.Random(seed)
    requests = queue_requests(request_count, latency_budget, rng)
    consume_queue(requests, rng)
    return calculate_completion_time(requests)


def print_report(requests: Iterable[MetadataRequest]) -> None:
    """Print request-level results and aggregate performance statistics."""
    completed = list(requests)
    header = (
        f"{'id':>3} {'arrival':>8} {'prompt':>7} {'output':>7} "
        f"{'queued':>8} {'service':>8} {'complete':>9} {'deadline':>9} {'met':>4}"
    )
    print(header)
    print("-" * len(header))

    for request in completed:
        output_tokens = (
            "-" if request.expected_output_tokens is None
            else str(request.expected_output_tokens)
        )
        response_time = (
            float("nan")
            if request.llm_response_time is None
            else request.llm_response_time
        )
        completion_time = (
            float("nan") if request.completion_time is None
            else request.completion_time
        )
        met_deadline = "-" if request.met_deadline is None else (
            "yes" if request.met_deadline else "no"
        )
        print(
            f"{request.id:>3} {request.arrival_time:>8.3f} "
            f"{request.prompt_tokens:>7} {output_tokens:>7} "
            f"{request.queue_waiting_time:>8.3f} {response_time:>8.3f} "
            f"{completion_time:>9.3f} {request.deadline:>9.3f} "
            f"{met_deadline:>4}"
        )

    completed_requests = [
        request
        for request in completed
        if request.completion_time is not None
        and request.llm_response_time is not None
        and request.total_latency is not None
    ]
    completed_count = len(completed_requests)
    met_count = sum(request.met_deadline is True for request in completed_requests)
    miss_count = completed_count - met_count

    if completed_count:
        average_queueing_time = sum(
            request.queue_waiting_time for request in completed_requests
        ) / completed_count
        average_service_time = sum(
            request.llm_response_time for request in completed_requests
        ) / completed_count
        average_end_to_end_latency = sum(
            request.total_latency for request in completed_requests
        ) / completed_count

        first_arrival = min(
            request.arrival_time for request in completed_requests
        )
        last_completion = max(
            request.completion_time for request in completed_requests
        )
        elapsed_time = last_completion - first_arrival
        throughput = completed_count / elapsed_time if elapsed_time > 0 else 0.0
        deadline_miss_rate = miss_count / completed_count
    else:
        average_queueing_time = 0.0
        average_service_time = 0.0
        average_end_to_end_latency = 0.0
        throughput = 0.0
        deadline_miss_rate = 0.0

    print(
        f"\nCompleted: {completed_count} | "
        f"Deadlines met: {met_count}/{completed_count}"
    )
    print(f"Average queueing time:      {average_queueing_time:.6f} s")
    print(f"Average service time:       {average_service_time:.6f} s")
    print(f"Average end-to-end latency: {average_end_to_end_latency:.6f} s")
    print(f"Throughput:                 {throughput:.3f} requests/s")
    print(f"Deadline-miss rate:         {deadline_miss_rate:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--requests",
        type=int,
        default=15,
        help="number of requests to simulate (default: 15)",
    )
    parser.add_argument(
        "--latency-budget",
        type=float,
        default=DEFAULT_LATENCY_BUDGET,
        help="per-request latency budget in seconds (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for a reproducible simulation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requests = run_simulation(
        request_count=args.requests,
        latency_budget=args.latency_budget,
        seed=args.seed,
    )
    print_report(requests)


if __name__ == "__main__":
    main()
